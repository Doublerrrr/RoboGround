#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""09 · 真实开放词汇后端集成测试（需要已下载模型权重）。

这是把"离线能跑"验证成"真模型能跑"的关键一步。会依次验证：
1. 四个模型能否加载（显存 / 耗时）；
2. Grounding DINO 能否在真实 SUN RGB-D 帧上检出物体；
3. SAM 能否把检测框转成精细掩码（并对比 bbox 掩码的差异）；
4. CLIP 能否同时编码图像与文本（开放词汇的数学前提）；
5. 完整流水线：真实检测 → 反投影 → 3D 语义地图 → **文本查询 3D 位置**；
6. 端到端耗时与显存峰值。

用法::

    python scripts/09_test_real_backends.py
    python scripts/09_test_real_backends.py --scene 1236 --steps all
    python scripts/09_test_real_backends.py --steps load,detect     # 只跑部分步骤
    python scripts/09_test_real_backends.py --no-sam                # 跳过 SAM（最快）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# HF 离线/镜像由用户环境决定；这里只保证不因为缺 token 而失败
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from roboground.utils.logging import get_logger, set_verbosity  # noqa: E402

log = get_logger("real_backends")


def vram_gb() -> float:
    import torch

    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def peak_vram_gb() -> float:
    import torch

    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def step_load(cfg, args) -> dict:
    """1) 加载四个模型，测显存与耗时。"""
    import torch

    from roboground.perception import build_detector, build_encoder, build_segmenter

    results = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    log.info("— 步骤 1：加载模型 —")

    t0 = time.perf_counter()
    detector = build_detector(cfg, name="grounding_dino")
    detector.warmup()
    results["detector"] = {"name": detector.name, "load_s": time.perf_counter() - t0,
                           "vram_gb": round(vram_gb(), 3)}
    log.ok(f"  Grounding DINO 就绪  {results['detector']['load_s']:.1f}s  "
           f"VRAM={results['detector']['vram_gb']:.2f}GB")

    segmenter = None
    if not args.no_sam:
        t0 = time.perf_counter()
        segmenter = build_segmenter(cfg, name="sam")
        segmenter.warmup()
        results["segmenter"] = {"name": segmenter.name, "load_s": time.perf_counter() - t0,
                                "vram_gb": round(vram_gb(), 3)}
        log.ok(f"  SAM 就绪             {results['segmenter']['load_s']:.1f}s  "
               f"VRAM={results['segmenter']['vram_gb']:.2f}GB")

    t0 = time.perf_counter()
    encoder = build_encoder(cfg, name=args.encoder)
    # 用 local_dir 下载过的模型可以直接指向本地路径，避免 Windows 符号链接权限问题
    local = Path(rf"G:\RoboGround\weights\{args.encoder}-base-patch16-224") \
        if args.encoder.startswith("siglip") else Path(r"G:\RoboGround\weights\clip-vit-base-patch32")
    if local.exists() and getattr(encoder, "model_id", None) != str(local):
        encoder.model_id = str(local)
    encoder.warmup()
    results["encoder"] = {"name": encoder.name, "load_s": time.perf_counter() - t0,
                          "vram_gb": round(vram_gb(), 3),
                          "feature_dim": encoder.feature_dim,
                          "supports_text": encoder.supports_text,
                          "calibration": "native" if getattr(encoder, "is_calibrated", False) else "null_text",
                          "suggested_threshold": getattr(encoder, "suggested_pair_threshold", None)}
    log.ok(f"  {encoder.name} 就绪            {results['encoder']['load_s']:.1f}s  "
           f"VRAM={results['encoder']['vram_gb']:.2f}GB  dim={encoder.feature_dim}  "
           f"文本={encoder.supports_text}  校准={results['encoder']['calibration']}")

    results["_objects"] = {"detector": detector, "segmenter": segmenter, "encoder": encoder}
    results["total_vram_gb"] = round(vram_gb(), 3)
    results["peak_vram_gb"] = round(peak_vram_gb(), 3)
    return results


def load_scene(cfg, args):
    """加载一个真实 SUN RGB-D 场景。"""
    from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene

    index_path = args.index or str(
        Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz"
    )
    index = load_scene_index(index_path)

    if args.scene is not None:
        scene = load_sunrgbd_scene(
            index, args.scene, max_depth=8.0, resize=(args.width, args.height)
        )
        if scene is None:
            raise RuntimeError(f"场景 {args.scene} 加载失败")
        return scene

    # 自动挑一个类别与 prompts 匹配的场景
    prompts = {str(p).lower() for p in cfg.get("perception.prompts", [])}
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))
    for pos in np.argsort(-counts)[:3000]:
        i = int(pos)
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = {str(x).lower() for x in np.asarray(index["label_flat"][off0:off1]).ravel()}
        if len(labels & prompts) < 2:
            continue
        scene = load_sunrgbd_scene(index, i, max_depth=8.0, resize=(args.width, args.height))
        if scene is not None and scene.boxes_3d.shape[0] > 0:
            log.info(f"自动选中场景 #{i}：{scene.sequence}")
            log.info(f"  GT 类别：{sorted(set(scene.labels))}")
            return scene
    raise RuntimeError("没找到合适的场景")


def step_detect(cfg, scene, objs, args) -> dict:
    """2) Grounding DINO 在真实帧上检测。"""
    log.info("— 步骤 2：Grounding DINO 开放词汇检测 —")
    frame = scene.to_frame()
    prompts = list(cfg.get("perception.prompts", []))
    log.info(f"  prompts：{prompts}")

    t0 = time.perf_counter()
    dets = objs["detector"].detect(frame, prompts)
    elapsed = time.perf_counter() - t0

    labels = sorted({d.label for d in dets})
    log.ok(f"  检出 {len(dets)} 个目标，耗时 {elapsed * 1000:.0f} ms，类别：{labels}")
    for d in sorted(dets, key=lambda x: -x.score)[:8]:
        b = d.bbox
        log.info(f"    {d.label:<18} score={d.score:.3f}  "
                 f"bbox=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")

    return {
        "num_detections": len(dets),
        "labels": labels,
        "detect_ms": round(elapsed * 1000, 1),
        "detections": dets,
        "frame": frame,
    }


def step_segment(cfg, scene, objs, dets, args) -> dict:
    """3) SAM 把框转成精细掩码，并与 bbox 掩码对比。"""
    if objs.get("segmenter") is None:
        return {}
    log.info("— 步骤 3：SAM 精细掩码（对比 bbox 掩码）—")
    frame = scene.to_frame()

    # 3a) bbox 掩码
    from roboground.perception.segmenters.box import BoxSegmenter

    box_seg = BoxSegmenter()
    subsample = dets[: min(len(dets), args.max_boxes)]
    box_dets = box_seg.segment(frame, subsample)
    box_areas = [int(np.asarray(d.mask).sum()) for d in box_dets if d.mask is not None]

    # 3b) SAM 掩码
    t0 = time.perf_counter()
    sam_all = objs["segmenter"].segment(frame, subsample)
    elapsed = time.perf_counter() - t0

    # ⚠️ 重要：SAM 只对前 max_boxes 个框做精细分割（其余保持原掩码），
    # 但它**返回完整列表**。绝不能把检测列表截断成 subsample ——
    # 那样会把没轮到 SAM 的检测整个丢掉（实测：10 个检测只剩 6 个进流水线）。
    dets = list(sam_all) + list(dets[len(subsample):])

    # 逐检测统计"掩码内有效深度像素数"，这是反投影能否成功的关键
    valid_depth = (frame.depth_m > cfg.get("geometry.min_depth", 0.1)) & \
                  (frame.depth_m < cfg.get("geometry.max_depth", 8.0))
    per_box = []
    for d in subsample:
        m = np.asarray(d.mask, dtype=bool) if d.mask is not None else None
        h, w = frame.shape
        if m is None or m.shape != (h, w):
            x1, y1, x2, y2 = np.asarray(d.bbox, dtype=int)
            m = np.zeros((h, w), dtype=bool)
            m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
        per_box.append({
            "label": d.label,
            "mask_px": int(m.sum()),
            "mask_valid_px": int((m & valid_depth).sum()),
        })

    sam_areas = [int(np.asarray(d.mask).sum()) for d in subsample if d.mask is not None]

    ratios = [s / max(b, 1) for s, b in zip(sam_areas, box_areas)]
    log.ok(f"  SAM 处理 {len(subsample)} 个框，耗时 {elapsed * 1000:.0f} ms "
           f"（{elapsed / max(len(subsample), 1) * 1000:.0f} ms/框）")
    log.info(f"  掩码像素数  bbox均值={np.mean(box_areas):.0f}  SAM均值={np.mean(sam_areas):.0f}  "
             f"SAM/bbox 比={np.mean(ratios):.2f}")
    log.info("  （SAM/bbox 比 < 1 说明 SAM 剔除了框内的背景像素 —— 这正是反投影精度的来源）")
    log.info("  逐框「掩码内有效深度像素」统计（为 0 的检测无法反投影，只能丢弃）：")
    for i, row in enumerate(per_box):
        flag = "  ← 无有效深度，将被丢弃" if row["mask_valid_px"] == 0 else ""
        log.info(f"    [{i}] {row['label']:<14} mask={row['mask_px']:>6}px  "
                 f"其中有效深度={row['mask_valid_px']:>6}px{flag}")

    return {
        "sam_ms": round(elapsed * 1000, 1),
        "sam_ms_per_box": round(elapsed / max(len(subsample), 1) * 1000, 1),
        "num_boxes": len(subsample),
        "bbox_area_mean": float(np.mean(box_areas)) if box_areas else 0.0,
        "sam_area_mean": float(np.mean(sam_areas)) if sam_areas else 0.0,
        "sam_over_bbox": float(np.mean(ratios)) if ratios else 0.0,
        "background_removed_pct": round((1.0 - float(np.mean(ratios))) * 100, 1) if ratios else 0.0,
        "boxes_without_valid_depth": int(sum(1 for r in per_box if r["mask_valid_px"] == 0)),
        "per_box": per_box,
        "detections": dets,
    }


def step_clip(cfg, scene, objs, dets, args) -> dict:
    """4) CLIP 图文同空间验证（开放词汇的数学前提）。"""
    log.info("— 步骤 4：CLIP 图文同空间特征 —")
    frame = scene.to_frame()
    encoder = objs["encoder"]

    t0 = time.perf_counter()
    feats = encoder.encode_regions(frame, dets[: min(len(dets), 16)])
    img_ms = (time.perf_counter() - t0) * 1000

    texts = ["a photo of a cup", "a photo of a chair", "a photo of a table",
             "a photo of a monitor", "a photo of a door", "a photo of a window"]
    t0 = time.perf_counter()
    txt_feats = encoder.encode_text(texts)
    txt_ms = (time.perf_counter() - t0) * 1000

    log.ok(f"  图像特征 {feats.shape}（{img_ms:.0f}ms）  文本特征 {txt_feats.shape}（{txt_ms:.0f}ms）")
    assert feats.shape[1] == txt_feats.shape[1], "图文特征维度必须一致才能做开放词汇查询"

    # 交叉相似度矩阵（证明"文本能查图像"在数学上成立）
    sim = feats @ txt_feats.T
    log.info("  图文相似度矩阵（行=检测，列=文本）。"
             "若同一检测的 self-sim（与其自身标签文本的相似度）接近该行最大值，说明图文空间对齐良好：")
    for i in range(min(len(dets), 6)):
        if i >= sim.shape[0]:
            break
        label = dets[i].label
        best = int(np.argmax(sim[i]))
        self_text = f"a photo of a {label.lower()}"
        row = f"    检测[{i}] {label:<16} → 最佳文本 {texts[best]!r:<28} sim={sim[i, best]:.3f}"
        if self_text in texts:
            j = texts.index(self_text)
            row += f"   |  自身标签 sim={sim[i, j]:.3f}  排名={int(np.argsort(-sim[i]).tolist().index(j)) + 1}/{len(texts)}"
        log.info(row)

    # 开放词汇能力演示：用**没在 prompts 里出现**的词去查
    novel = ["a photo of a laptop", "a photo of a bottle", "a photo of a plant",
             "a photo of a trash can bin", "a photo of a person"]
    novel_feats = encoder.encode_text(novel)
    novel_sim = feats @ novel_feats.T
    log.info("  【开放词汇验证】用未出现在 prompts 中的词查询：")
    for j, t in enumerate(novel):
        best_i = int(np.argmax(novel_sim[:, j]))
        log.info(f"    {t!r:<32} → 最相似检测：{dets[best_i].label:<16} sim={novel_sim[best_i, j]:.3f}")

    return {
        "image_feat_shape": list(feats.shape),
        "text_feat_shape": list(txt_feats.shape),
        "encode_image_ms": round(img_ms, 1),
        "encode_text_ms": round(txt_ms, 1),
        "features": feats,
        "texts": texts,
        "text_features": txt_feats,
    }


def step_pipeline(cfg, scene, objs, dets, args) -> dict:
    """5) 完整流水线：真实检测 → 3D 语义地图 → 文本查询。"""
    log.info("— 步骤 5：完整流水线（检测 → 反投影 → 3D 地图 → 文本查询）—")
    from roboground.mapping import MapBuilder

    frame = scene.to_frame()
    # 把真实检测塞进 pipeline（跳过检测阶段，复用已算好的结果）
    from roboground.perception.base import PerceptionPipeline

    class _Precomputed:
        """把已算好的检测包成 pipeline 接口，避免重复推理。"""

        def __init__(self, detections, encoder):
            self._dets = detections
            self.encoder = encoder
            self.detector = objs["detector"]
            self.segmenter = objs.get("segmenter")
            self.stats = {}

        def run(self, frame, prompts=None):
            if self.encoder is not None and self._dets:
                feats = self.encoder.encode_regions(frame, self._dets)
                for d, f in zip(self._dets, feats):
                    d.feature = f
            return list(self._dets)

        def profile(self):
            return dict(self.stats)

    pipe = _Precomputed(dets, objs["encoder"])
    builder = MapBuilder(cfg, pipeline=pipe, prompts=list(cfg.get("perception.prompts", [])))

    t0 = time.perf_counter()
    smap = builder.build_from_frames([frame])
    build_ms = (time.perf_counter() - t0) * 1000

    log.kv("  地图", {
        "物体数": smap.num_objects, "体素数": smap.num_voxels,
        "特征维度": smap.feature_dim, "建图耗时(ms)": round(build_ms, 1),
        "类别": smap.labels,
    })
    for obj in sorted(smap.objects, key=lambda o: -o.confidence)[:10]:
        log.info("    " + obj.describe())

    # ---- 关键：用 CLIP 文本特征做真·开放词汇查询 ----
    log.info("  【开放词汇 3D 查询】用文本直接在地图里定位（CLIP 嵌入匹配）：")

    # 把 CLIP 挂到地图上，让 QueryEngine 走 embedding 路径
    smap.text_encoder = objs["encoder"]
    from roboground.mapping.query import QueryEngine

    engine = QueryEngine(smap, text_encoder=objs["encoder"])
    log.info(f"    查询引擎模式：{engine.mode}")

    queries = [
        "杯子", "椅子", "桌子", "显示器",
        "a cup", "a chair", "a table",
        "something to drink from",     # 从未出现过的表达（真开放词汇测试）
        "a place to sit",              # 同上
        "冰箱",                         # 地图里应该没有
    ]
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        hits = engine.query(q, top_k=3, min_score=0.05)
        qms = (time.perf_counter() - t0) * 1000
        if hits:
            top = hits[0]
            lab = top.label or (top.obj.label if top.obj else "?")
            pos = top.position
            log.info(f"    {q!r:32} → {lab:<16} @({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}) "
                     f"score={top.score:.3f} ({qms:.0f}ms, by={top.matched_by})")
            rows.append({"query": q, "label": lab, "score": round(top.score, 4),
                         "matched_by": top.matched_by, "latency_ms": round(qms, 2)})
        else:
            log.info(f"    {q!r:32} → （无命中）({qms:.0f}ms)")
            rows.append({"query": q, "label": None, "score": 0.0,
                         "matched_by": engine.mode, "latency_ms": round(qms, 2)})

    return {
        "num_objects": smap.num_objects,
        "num_voxels": smap.num_voxels,
        "build_ms": round(build_ms, 1),
        "query_mode": engine.mode,
        "query_rows": rows,
        "_map": smap,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=int, default=None)
    ap.add_argument("--index", default=None)
    # SUN RGB-D 原图是 730×530，用 640×480 做等比缩放到接近的长宽比
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max-boxes", type=int, default=8, help="送多少个框给 SAM")
    ap.add_argument("--no-sam", action="store_true", help="跳过 SAM（更快）")
    ap.add_argument("--encoder", default="siglip", choices=["siglip", "clip"],
                    help="图文编码器（siglip 实测更优，见 scripts/11）")
    ap.add_argument("--steps", default="all",
                    help="逗号分隔：load,detect,segment,clip,pipeline；all=全部")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    from roboground import load_config

    cfg = load_config("configs/perception_openvocab.yaml")
    cfg.set("project.device", "cuda")
    if args.prompts:
        cfg.set("perception.prompts", args.prompts)
    # CLIP 已用 local_dir 下载，指向本地路径避免重新触发缓存逻辑
    clip_local = Path(r"G:\RoboGround\weights\clip-vit-base-patch32")
    if clip_local.exists():
        cfg.set("perception.encoder_kwargs.model_id", str(clip_local))

    steps = set(args.steps.split(",")) if args.steps != "all" else {"load", "detect", "segment", "clip", "pipeline"}

    print()
    log.info("=" * 70)
    log.info("RoboGround 真实开放词汇后端集成测试")
    log.info("=" * 70)

    try:
        import torch

        if torch.cuda.is_available():
            log.info(f"GPU: {torch.cuda.get_device_name(0)}  "
                     f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        log.error("需要 torch")
        return 2

    report = {}
    objs = {}
    scene = None
    dets = []

    if "load" in steps:
        r = step_load(cfg, args)
        objs = r.pop("_objects")
        report["load"] = r
    else:
        from roboground.perception import build_detector, build_encoder, build_segmenter

        objs["detector"] = build_detector(cfg, name="grounding_dino")
        objs["encoder"] = build_encoder(cfg, name=getattr(args, "encoder", "siglip"))
        objs["segmenter"] = None if args.no_sam else build_segmenter(cfg, name="sam")

    if steps & {"detect", "segment", "clip", "pipeline"}:
        log.info("— 加载真实数据 —")
        scene = load_scene(cfg, args)

    if "detect" in steps:
        r = step_detect(cfg, scene, objs, args)
        dets = r.pop("detections")
        r.pop("frame", None)
        report["detect"] = r

    if dets and "segment" in steps:
        r = step_segment(cfg, scene, objs, dets, args)
        if r:
            dets = r.pop("detections", dets)
            report["segment"] = r

    if dets and "clip" in steps:
        r = step_clip(cfg, scene, objs, dets, args)
        for k in ("features", "text_features"):
            r.pop(k, None)
        report["clip"] = r

    if dets and "pipeline" in steps:
        r = step_pipeline(cfg, scene, objs, dets, args)
        r.pop("_map", None)
        report["pipeline"] = r

    # ---- 汇总 ----
    print()
    log.info("=" * 70)
    log.info("汇总")
    log.info("=" * 70)
    if "load" in report:
        log.kv("显存", {
            "全部加载后(GB)": report["load"].get("total_vram_gb"),
            "峰值(GB)": report["load"].get("peak_vram_gb"),
        })
    for key in ("detect", "segment", "clip", "pipeline"):
        if key in report:
            log.info(f"  {key}: " + ", ".join(
                f"{k}={v}" for k, v in report[key].items()
                if not isinstance(v, (list, dict))
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
