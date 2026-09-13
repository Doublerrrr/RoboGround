"""Stage 5：自动化标注 pipeline（数据闭环）。

对应 JD 的"协助构建多模态数据集训练 pipeline，设计数据清洗、增强及自动化标注策略"。

核心思路
--------
**"用模型产出标注，再用几何与一致性去筛掉不可信的标注"**，
从而把人工从"逐帧画框"降到"抽检纠错"。

四道质量闸门（这是本模块的算法含量所在）
--------------------------------------
1. **置信度闸门**：检测分数低于阈值直接丢；
2. **几何有效性闸门**：反投影后的 3D 点数太少（< min_points）说明深度无效
   （典型是玻璃/远处/无纹理），这类标注宁可不要；
3. **尺度合理性闸门**：3D 包围盒的尺寸超出物理常识（如 0.01m 的"桌子"
   或 8m 的"杯子"）则剔除 —— 用类别先验尺寸表判定；
4. **一致性闸门**（可选）：同一物体在多帧里的 3D 位置应一致，
   偏差过大的标注标记为"待人工复核"而不是直接采用。

输出格式
--------
- `coco.json`：2D 检测/分割标注（可直接喂 YOLO/Mask2Former 训练）；
- `boxes3d.json`：3D 框标注（喂 3D 检测/VLM 空间推理训练）；
- `review.json`：被闸门拦下、需要人工复核的样本（含原因）—— **不是丢弃**，
  这部分才是主动学习的价值所在。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roboground.geometry.projection import backproject_detection
from roboground.types import Detection2D, Observation, RGBDFrame
from roboground.utils.io import ensure_dir, save_json
from roboground.utils.logging import get_logger

logger = get_logger("data.auto_label")


#: 类别 → 合理的 3D 尺寸范围（米），用于尺度合理性检查。
#: 这些是**物理常识先验**（不是数据集统计），所以可以跨数据集使用。
PLAUSIBLE_SIZE_RANGES: Dict[str, Tuple[float, float]] = {
    "cup": (0.04, 0.30),
    "bottle": (0.05, 0.40),
    "book": (0.05, 0.40),
    "box": (0.05, 1.20),
    "bag": (0.10, 1.00),
    "lamp": (0.08, 0.80),
    "monitor": (0.15, 2.00),
    "laptop": (0.15, 0.60),
    "keyboard": (0.15, 0.80),
    "phone": (0.04, 0.25),
    "chair": (0.25, 1.50),
    "table": (0.30, 3.00),
    "desk": (0.30, 3.00),
    "sofa": (0.50, 4.00),
    "bed": (0.80, 3.50),
    "shelf": (0.20, 3.00),
    "cabinet": (0.20, 3.00),
    "door": (0.60, 3.00),
    "window": (0.30, 4.00),
    "toilet": (0.30, 1.20),
    "person": (0.80, 2.50),
    "trash can": (0.15, 1.00),
    "plant": (0.10, 2.00),
    "default": (0.02, 6.00),
}


@dataclass
class LabelItem:
    """一条自动标注。"""

    frame_id: str
    label: str
    score: float
    bbox: np.ndarray                 # (4,) xyxy 像素
    num_points: int
    points_world: Optional[np.ndarray] = None
    box3d: Optional[np.ndarray] = None      # (7,) cx,cy,cz,dx,dy,dz,heading
    accepted: bool = True
    reject_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_coco_annotation(self, image_id: int, ann_id: int) -> Dict[str, Any]:
        x1, y1, x2, y2 = [float(v) for v in self.bbox]
        return {
            "id": int(ann_id),
            "image_id": int(image_id),
            "category_id": -1,                       # 由调用方按类别表填充
            "bbox": [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)],
            "area": float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)),
            "iscrowd": 0,
            "score": float(self.score),
            "attributes": {"label": self.label, "auto": True},
        }

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "frame_id": self.frame_id,
            "label": self.label,
            "score": round(float(self.score), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "num_points": int(self.num_points),
            "accepted": bool(self.accepted),
        }
        if self.box3d is not None:
            payload["box3d"] = [round(float(v), 4) for v in self.box3d]
        if self.reject_reason:
            payload["reject_reason"] = self.reject_reason
        return payload


class AutoLabeler:
    """多模态自动化标注器。

    Examples
    --------
    >>> from roboground import load_config                  # doctest: +SKIP
    >>> labeler = AutoLabeler(load_config())                # doctest: +SKIP
    >>> report = labeler.label_frames(frames, out_dir="runs/labels")  # doctest: +SKIP
    """

    def __init__(
        self,
        cfg,
        *,
        pipeline: Any = None,
        prompts: Optional[Sequence[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.prompts = (
            list(prompts) if prompts is not None
            else list(cfg.get("perception.prompts", []) or [])
        )
        if pipeline is None:
            from roboground.perception import build_pipeline  # noqa: PLC0415

            pipeline = build_pipeline(cfg, prompts=self.prompts)
        self.pipeline = pipeline

        self.min_score = float(cfg.get("perception.detector_kwargs.box_threshold", 0.30))
        self.min_points = int(cfg.get("data.auto_label_min_points", 20))
        self.check_scale = bool(cfg.get("data.auto_label_check_scale", True))
        self.min_depth = float(cfg.get("geometry.min_depth", 0.1))
        self.max_depth = float(cfg.get("geometry.max_depth", 8.0))
        self.ghost_percentile = cfg.get("geometry.depth_trunc_percentile", None)

    # ---------------- 单帧标注 ----------------
    def label_frame(self, frame: RGBDFrame, *, prompts: Optional[Sequence[str]] = None) -> List[LabelItem]:
        """对一帧做自动标注（含四道闸门）。"""
        detections: List[Detection2D] = self.pipeline.run(frame, prompts=prompts)
        items: List[LabelItem] = []

        for det in detections:
            item = LabelItem(
                frame_id=frame.frame_id, label=det.label, score=float(det.score),
                bbox=np.asarray(det.bbox, dtype=np.float64).copy(), num_points=0,
            )

            # 闸门 1：置信度
            if item.score < self.min_score:
                item.accepted, item.reject_reason = False, "low_score"
                items.append(item)
                continue

            obs: Observation = backproject_detection(
                det, frame, min_depth=self.min_depth, max_depth=self.max_depth,
                ghost_percentile=self.ghost_percentile,
            )
            item.num_points = obs.num_points
            item.points_world = obs.points_world

            # 闸门 2：几何有效性（点太少 → 深度无效）
            if obs.num_points < self.min_points:
                item.accepted, item.reject_reason = False, "too_few_points"
                items.append(item)
                continue

            lo = obs.points_world.min(axis=0)
            hi = obs.points_world.max(axis=0)
            size = np.clip(hi - lo, 1e-4, None)
            center = (lo + hi) / 2.0
            item.box3d = np.concatenate([center, size, [0.0]]).astype(np.float32)

            # 闸门 3：尺度合理性
            if self.check_scale:
                lo_r, hi_r = PLAUSIBLE_SIZE_RANGES.get(
                    str(det.label).lower(), PLAUSIBLE_SIZE_RANGES["default"]
                )
                diag = float(np.linalg.norm(size))
                if not (lo_r * 0.2 <= diag <= hi_r * 3.0):
                    item.accepted, item.reject_reason = False, f"implausible_size({diag:.2f}m)"
                    items.append(item)
                    continue

            item.extra["size"] = size.tolist()
            items.append(item)

        return items

    # ---------------- 数据集级标注 ----------------
    def label_frames(
        self,
        frames: Iterable[RGBDFrame],
        *,
        out_dir: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """批量标注并（可选）落盘为 COCO / 3D 框 / 待复核三份文件。"""
        frames = list(frames)
        t0 = time.perf_counter()
        all_items: List[LabelItem] = []
        for i, frame in enumerate(frames):
            all_items.extend(self.label_frame(frame, prompts=prompts))
            if (i + 1) % 20 == 0:
                logger.debug(f"  已标注 {i + 1}/{len(frames)} 帧")
        elapsed = time.perf_counter() - t0

        accepted = [it for it in all_items if it.accepted]
        rejected = [it for it in all_items if not it.accepted]

        # 类别分布
        dist: Dict[str, int] = {}
        for it in accepted:
            dist[it.label] = dist.get(it.label, 0) + 1

        # 拒绝原因分布（用于诊断数据/模型问题）
        reasons: Dict[str, int] = {}
        for it in rejected:
            key = str(it.reject_reason).split("(")[0]
            reasons[key] = reasons.get(key, 0) + 1

        report: Dict[str, Any] = {
            "num_frames": len(frames),
            "num_candidates": len(all_items),
            "num_accepted": len(accepted),
            "num_rejected": len(rejected),
            "accept_rate": round(len(accepted) / max(len(all_items), 1), 4),
            "class_distribution": dict(sorted(dist.items(), key=lambda kv: -kv[1])),
            "reject_reasons": reasons,
            "elapsed_s": round(elapsed, 3),
            "ms_per_frame": round(elapsed * 1000.0 / max(len(frames), 1), 3),
        }

        if out_dir:
            out = ensure_dir(out_dir)
            save_json([it.to_dict() for it in accepted], out / "boxes3d.json")
            save_json([it.to_dict() for it in rejected], out / "review.json")
            categories = sorted(dist.keys())
            cat_ids = {name: i + 1 for i, name in enumerate(categories)}
            coco = {
                "info": {"description": f"RoboGround auto labels ({len(frames)} frames)"},
                "categories": [{"id": cid, "name": name} for name, cid in cat_ids.items()],
                "images": [
                    {"id": i, "file_name": f"{frames[i].frame_id}.png",
                     "width": frames[i].width, "height": frames[i].height}
                    for i in range(len(frames))
                ],
                "annotations": [],
            }
            frame_index = {f.frame_id: i for i, f in enumerate(frames)}
            for aid, it in enumerate(accepted, start=1):
                ann = it.to_coco_annotation(frame_index.get(it.frame_id, 0), aid)
                ann["category_id"] = cat_ids.get(it.label, 1)
                coco["annotations"].append(ann)
            save_json(coco, out / "coco.json")
            report["output_dir"] = str(out)
            logger.ok(
                f"自动标注完成：{len(accepted)} 条可用 / {len(rejected)} 条待复核 "
                f"（采纳率 {report['accept_rate']:.1%}），已写入 {out}"
            )

        return report

    # ---------------- 一致性检查（主动学习信号）----------------
    @staticmethod
    def cross_frame_consistency(
        items: Sequence[LabelItem],
        *,
        radius: float = 0.6,
        max_deviation: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """跨帧一致性检查：同类物体的 3D 中心应聚成簇。

        偏差过大的标注返回为"可疑项"，进入人工复核队列 ——
        这就是**主动学习**里"用模型不确定性选样本"的一个廉价替代方案。
        """
        by_label: Dict[str, List[LabelItem]] = {}
        for it in items:
            if it.accepted and it.box3d is not None:
                by_label.setdefault(it.label, []).append(it)

        suspects: List[Dict[str, Any]] = []
        for label, group in by_label.items():
            centers = np.stack([it.box3d[:3] for it in group], axis=0)
            if centers.shape[0] < 3:
                continue
            median = np.median(centers, axis=0)
            dists = np.linalg.norm(centers - median, axis=1)
            for it, d in zip(group, dists):
                if d > max_deviation:
                    suspects.append({
                        "frame_id": it.frame_id, "label": label,
                        "center": [round(float(v), 3) for v in it.box3d[:3]],
                        "deviation_m": round(float(d), 3),
                        "reason": "cross_frame_inconsistent",
                    })
        return suspects
