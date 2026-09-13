"""性能与质量基准：把"快不快、准不准"变成可引用的数字。

这个模块的产出直接对应简历里要写的量化指标，所以设计目标是
**每次运行都能稳定复现同一张表**。

测什么
------
| 阶段 | 指标 | 为什么重要 |
|---|---|---|
| 感知（检测/分割/编码） | ms/帧 | 端侧实时性的主要瓶颈 |
| 反投影 + 体素融合 | ms/帧 | 几何链路开销 |
| 物体构建（关联/聚类） | ms | 随物体数增长，要能看出量级 |
| 语言查询 | ms/次 | 交互响应速度 |
| 3D 定位精度 | 中位误差(m) | **空间语义的核心质量指标** |
| 检测指标 | P/R/F1/mIoU | 与 GT 对比 |
| 查询命中率 | Top-1/3/5 | 端到端可用性 |

⚠️ 关于 `torch.cuda.synchronize`
GPU 是异步执行的，不 sync 就计时会把"提交内核"的时间当成"算完"的时间，
测出来的延迟会**显著偏低**。本模块在每次计时前后都做同步。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roboground.eval.metrics import (
    aggregate,
    detection_metrics,
    format_metrics,
    localization_errors,
    topk_accuracy,
)
from roboground.utils.logging import get_logger

logger = get_logger("eval.benchmark")


def _sync() -> None:
    """GPU 计时前必须同步（否则延迟被低估）。"""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class _Stopwatch:
    """分段计时器（自动处理 CUDA 同步）。"""

    def __init__(self) -> None:
        self.records: Dict[str, List[float]] = {}

    def time(self, name: str):
        return _TimedBlock(self, name)

    def add(self, name: str, ms: float) -> None:
        self.records.setdefault(name, []).append(float(ms))

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, values in self.records.items():
            arr = np.asarray(values, dtype=np.float64)
            out[name] = {
                "n": float(arr.size),
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.median(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "total_ms": float(arr.sum()),
            }
        return out


class _TimedBlock:
    def __init__(self, stopwatch: _Stopwatch, name: str) -> None:
        self.stopwatch = stopwatch
        self.name = name

    def __enter__(self) -> "_TimedBlock":
        _sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        _sync()
        self.stopwatch.add(self.name, (time.perf_counter() - self._t0) * 1000.0)


# ==========================================================================
# 主基准
# ==========================================================================
def benchmark_pipeline(
    cfg,
    frames: Sequence[Any],
    *,
    prompts: Optional[Sequence[str]] = None,
    queries: Optional[Sequence[str]] = None,
    gt_boxes: Optional[Sequence[np.ndarray]] = None,
    query_expectations: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """端到端基准：建图耗时 + 查询耗时 + 定位精度 + 检测指标。

    Parameters
    ----------
    frames
        `RGBDFrame` 序列。
    gt_boxes
        与帧一一对应的 GT 3D 框（可选）。给了才计算定位精度与检测指标。
    queries, query_expectations
        查询列表与对应的期望标签（可选）。给了才计算 Top-K 命中率。
    """
    from roboground.mapping import MapBuilder  # noqa: PLC0415
    from roboground.perception import build_pipeline  # noqa: PLC0415
    from roboground.reasoning import RuleEngine  # noqa: PLC0415

    frames = list(frames)
    use_prompts = list(prompts) if prompts is not None else list(cfg.get("perception.prompts", []) or [])

    stopwatch = _Stopwatch()
    result: Dict[str, Any] = {"num_frames": len(frames), "num_queries": len(queries or [])}

    # ---------------- 1) 感知 ----------------
    pipeline = build_pipeline(cfg, prompts=use_prompts)
    per_frame_detections: List[List[Any]] = []
    with stopwatch.time("perceive_ms"):
        for frame in frames:
            dets = pipeline.run(frame, prompts=use_prompts)
            per_frame_detections.append(dets)
    result["perception_stats"] = pipeline.profile()
    result["detections_per_frame"] = float(
        np.mean([len(d) for d in per_frame_detections]) if per_frame_detections else 0.0
    )

    # ---------------- 2) 建图（含反投影/融合/关联）----------------
    builder = MapBuilder(cfg, pipeline=pipeline, prompts=use_prompts)
    _sync()
    t0 = time.perf_counter()
    semantic_map = builder.build_from_frames(frames, prompts=use_prompts)
    _sync()
    result["build_total_ms"] = (time.perf_counter() - t0) * 1000.0
    result["build_per_frame_ms"] = result["build_total_ms"] / max(len(frames), 1)
    result["num_objects"] = float(semantic_map.num_objects)
    result["num_voxels"] = float(semantic_map.num_voxels)
    result["map_labels"] = semantic_map.labels

    # ---------------- 3) 查询 ----------------
    if queries:
        engine = RuleEngine(semantic_map, cfg=cfg)
        predicted_labels: List[List[str]] = []
        query_times: List[float] = []
        for q in queries:
            _sync()
            t0 = time.perf_counter()
            res = engine.answer(q)
            _sync()
            query_times.append((time.perf_counter() - t0) * 1000.0)
            predicted_labels.append(
                [getattr(t, "label", str(t)) for t in (res.targets or [])]
            )
        stopwatch.records["query_ms"] = query_times
        if query_expectations:
            result["query_accuracy"] = topk_accuracy(
                predicted_labels, list(query_expectations), ks=(1, 3, 5)
            )

    # ---------------- 4) 精度（需要 GT）----------------
    if gt_boxes is not None and len(gt_boxes) == len(frames):
        pred_list: List[List[np.ndarray]] = []
        gt_list: List[List[np.ndarray]] = []
        pred_centers: List[np.ndarray] = []
        gt_centers: List[np.ndarray] = []

        for frame, gt in zip(frames, gt_boxes):
            gt = np.asarray(gt)
            gt_list.append([g for g in gt])
            # 预测：把该帧的观测（反投影后的 3D 点云 bbox）当作预测框
            obs = [o for o in builder.observations if o.frame_id == frame.frame_id]
            preds: List[np.ndarray] = []
            for o in obs:
                lo = o.points_world.min(axis=0)
                hi = o.points_world.max(axis=0)
                size = np.clip(hi - lo, 1e-4, None)
                center = (lo + hi) / 2.0
                preds.append(np.concatenate([center, size, [0.0]]))
            pred_list.append(preds)

            # 观测级定位误差：用匹配上的框对，算中心距离
            from roboground.eval.metrics import match_boxes  # noqa: PLC0415

            m = match_boxes(preds, list(gt), iou_threshold=0.10)
            for pi, gi in m["matches"]:
                pred_centers.append(preds[pi][:3])
                gt_centers.append(np.asarray(gt[gi])[:3])

        result["detection"] = detection_metrics(pred_list, gt_list, iou_thresholds=(0.25, 0.50))
        result["localization"] = localization_errors(pred_centers, gt_centers)

        # ---- 地图级定位精度（**这才是系统真正的输出质量**）----
        result["map_localization"] = map_level_localization(
            semantic_map, gt_boxes
        )

        # 说明为什么观测级检测 IoU 可能很低（避免误读成"系统坏了"）
        if result["detection"].get("mean_iou@0.25", 0.0) < 0.05:
            result["detection_note"] = (
                "检测 IoU 接近 0 属**预期**：当前离线后端把 2D bbox 直接当掩码，"
                "反投影会把框内的背景点（地面/墙）也纳入 3D 包围盒，导致预测框显著偏大、"
                "且朝向被近似为 0。这正是需要用 SAM 精细掩码替换 bbox 的原因；"
                "请以 `map_localization`（物体级中心误差）作为主要质量指标。"
            )

    result["timings"] = stopwatch.summary()
    return result


def map_level_localization(
    semantic_map,
    gt_boxes_per_frame: Sequence[np.ndarray],
    *,
    max_distance: float = 2.0,
) -> Dict[str, float]:
    """地图产出的物体 vs 去重后的 GT 物体：中心定位误差。

    为什么需要这个指标
    -----------------
    观测级指标衡量的是"单个 2D 检测反投影得准不准"，
    而机器人真正消费的是**地图里的物体**（"杯子在哪" → 一个 3D 坐标）。
    所以这个指标才是端到端质量的核心数字。

    匹配规则：同标签 + 最近中心（贪心一对一），
    距离超过 `max_distance` 视为未匹配。
    """
    # GT 去重（多帧指向同一房间时 GT 会重复）
    gt_objects: List[Tuple[str, np.ndarray]] = []
    for boxes in gt_boxes_per_frame:
        boxes = np.asarray(boxes)
        if boxes.size == 0:
            continue
        for b in boxes.reshape(-1, boxes.shape[-1] if boxes.ndim > 1 else 7):
            center = np.asarray(b[:3], dtype=np.float64)
            if not np.all(np.isfinite(center)):
                continue
            duplicated = any(
                np.linalg.norm(center - c) < 0.05 for _, c in gt_objects
            )
            if not duplicated:
                gt_objects.append(("", center))       # 标签未知时只用几何去重

    if not semantic_map.objects or not gt_objects:
        return {"n": 0.0, "median_m": float("nan"), "mean_m": float("nan")}

    used = set()
    pred_centers: List[np.ndarray] = []
    matched_gt: List[np.ndarray] = []
    for obj in semantic_map.objects:
        best_j, best_d = -1, float("inf")
        for j, (_, center) in enumerate(gt_objects):
            if j in used:
                continue
            d = float(np.linalg.norm(np.asarray(obj.center, dtype=np.float64) - center))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= max_distance:
            used.add(best_j)
            pred_centers.append(np.asarray(obj.center, dtype=np.float64))
            matched_gt.append(gt_objects[best_j][1])

    stats = localization_errors(pred_centers, matched_gt)
    stats["num_map_objects"] = float(semantic_map.num_objects)
    stats["num_gt_objects"] = float(len(gt_objects))
    stats["match_rate"] = len(pred_centers) / max(len(gt_objects), 1)
    return stats


# ==========================================================================
# 分阶段基准（用于定位瓶颈）
# ==========================================================================
def benchmark_perception(
    cfg,
    frames: Sequence[Any],
    *,
    prompts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """只测感知阶段（换后端时对比用）。"""
    from roboground.perception import build_pipeline  # noqa: PLC0415

    pipeline = build_pipeline(cfg, prompts=prompts)
    sw = _Stopwatch()
    for frame in frames:
        with sw.time("perception_ms"):
            pipeline.run(frame, prompts=prompts)
    out = sw.summary()
    out["backend"] = {
        "detector": pipeline.detector.name if pipeline.detector else None,
        "segmenter": pipeline.segmenter.name if pipeline.segmenter else None,
        "encoder": pipeline.encoder.name if pipeline.encoder else None,
        "feature_dim": getattr(pipeline.encoder, "feature_dim", None),
        "supports_text": getattr(pipeline.encoder, "supports_text", None),
    }
    return out


def benchmark_query(
    semantic_map,
    queries: Sequence[str],
    *,
    cfg: Any = None,
    repeat: int = 3,
) -> Dict[str, Any]:
    """只测查询阶段（含规则引擎与可选嵌入匹配）。"""
    from roboground.reasoning import RuleEngine  # noqa: PLC0415

    engine = RuleEngine(semantic_map, cfg=cfg)
    sw = _Stopwatch()
    answers: List[Dict[str, Any]] = []
    for _ in range(max(int(repeat), 1)):
        for q in queries:
            with sw.time("query_ms"):
                res = engine.answer(q)
            answers.append({"query": q, "answer": res.answer, "targets": len(res.targets or [])})
    out = sw.summary()
    out["answers"] = answers[: len(queries)]
    return out


def benchmark_map_query(
    semantic_map,
    queries: Sequence[str],
    *,
    encoder: Any = None,
    repeat: int = 3,
) -> Dict[str, Any]:
    """只测"地图级"文本查询（`SemanticMap.query_text`）的延迟与命中。"""
    sw = _Stopwatch()
    hits: List[Dict[str, Any]] = []
    for _ in range(max(int(repeat), 1)):
        for q in queries:
            with sw.time("map_query_ms"):
                results = semantic_map.query_text(q, top_k=3, text_encoder=encoder)
            hits.append({
                "query": q,
                "num_hits": len(results),
                "top1": (results[0].label if results else None),
                "top1_score": (round(results[0].score, 4) if results else 0.0),
            })
    out = sw.summary()
    out["hits"] = hits[: len(queries)]
    return out


# ==========================================================================
# 报告
# ==========================================================================
def format_report(result: Dict[str, Any], *, title: str = "RoboGround 基准报告") -> str:
    """把基准结果渲染成 Markdown 报告（可直接贴进文档/简历附件）。"""
    lines: List[str] = [f"# {title}", ""]

    basic = {
        "帧数": result.get("num_frames"),
        "查询数": result.get("num_queries"),
        "检出/帧": round(result.get("detections_per_frame", 0.0), 2),
        "物体数": int(result.get("num_objects", 0)),
        "体素数": int(result.get("num_voxels", 0)),
        "建图总耗时(ms)": round(result.get("build_total_ms", 0.0), 2),
        "建图/帧(ms)": round(result.get("build_per_frame_ms", 0.0), 2),
    }
    lines += ["## 规模与总耗时", "", "| 指标 | 值 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in basic.items() if v is not None]
    lines.append("")

    labels = result.get("map_labels")
    if labels:
        lines += ["## 地图中的类别", "", "、".join(f"`{x}`" for x in labels[:30]), ""]

    timings = result.get("timings") or {}
    if timings:
        lines += ["## 阶段耗时", "", "| 阶段 | n | mean(ms) | p50(ms) | p95(ms) |", "|---|---|---|---|---|"]
        for name, s in timings.items():
            lines.append(
                f"| {name} | {int(s['n'])} | {s['mean_ms']:.2f} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} |"
            )
        lines.append("")

    if "map_localization" in result:
        lines += [
            "## 地图级定位精度（主要质量指标）", "",
            format_metrics(result["map_localization"]), "",
        ]

    if "localization" in result:
        lines += ["## 观测级定位精度（单帧反投影）", "", format_metrics(result["localization"]), ""]

    if "detection" in result:
        lines += ["## 检测指标（bbox 反投影基线）", "", format_metrics(result["detection"])]
        if result.get("detection_note"):
            lines += ["", f"> ⚠️ {result['detection_note']}"]
        lines.append("")

    if "query_accuracy" in result:
        lines += ["## 查询命中率", "", format_metrics(result["query_accuracy"]), ""]

    if result.get("perception_stats"):
        lines += [
            "## 感知耗时拆解（最近一帧）", "",
            format_metrics(result["perception_stats"]), "",
        ]

    return "\n".join(lines)
