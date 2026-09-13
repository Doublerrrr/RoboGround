"""离线检测器：不依赖任何模型权重，但能产生**真实可用**的检测。

两种工作模式
------------
1. **GT 投影模式（默认，推荐）**
   如果 `frame.meta` 里带了 `boxes_3d` 和 `labels`（例如 SUN RGB-D 的标注），
   就把 3D 有向包围盒的 8 个角点投影到图像，取 2D 外接矩形作为检测框。

   这样做的价值：**整条 3D 语义地图流水线可以在没有任何模型下载的情况下，
   用真实数据跑通并定量评估**（GT 框 → 2D 检测 → 反投影 → 3D 地图 →
   与 GT 对比）。它同时是回归测试的黄金标准 —— 因为输入输出都可控。

2. **网格模式（兜底）**
   完全没有元数据时，生成确定性的网格框。**只用于形状/接口测试**，
   不产生有意义的语义。

为什么 `filter_by_prompts` 默认开启？
因为真实开放词汇检测器是"给什么 prompt 就检什么"。stub 也模拟这个行为
（用双语别名表做词法过滤），这样"换 prompt 得到不同结果"的链路才真实。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from roboground.perception.base import Detector
from roboground.perception.registry import register_detector
from roboground.types import Detection2D, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("perception.stub")


# ==========================================================================
# 3D 有向包围盒工具
# ==========================================================================
def rotz(angle: float) -> np.ndarray:
    """绕 z 轴旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def obb_corners(center: Sequence[float], size: Sequence[float], heading: float = 0.0) -> np.ndarray:
    """3D 有向包围盒 → 8 个角点（世界系）。

    Parameters
    ----------
    center : (3,)
    size : (3,)
        全边长 (dx, dy, dz)。
    heading : float
        绕 z 轴的偏航角（弧度）。

    Returns
    -------
    np.ndarray, shape (8, 3)
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    s = np.asarray(size, dtype=np.float64).reshape(3)
    half = np.clip(s, 1e-6, None) / 2.0
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1],
    ], dtype=np.float64)
    local = signs * half
    return local @ rotz(heading).T + c


@register_detector("stub", "offline", "gt")
class StubDetector(Detector):
    """离线检测器（GT 投影 / 网格兜底）。

    Parameters
    ----------
    prompts : list[str]
        默认 prompt 列表。
    filter_by_prompts : bool
        是否只保留与 prompt 词法匹配的目标（模拟开放词汇检测行为）。
    score : float
        输出置信度（GT 模式默认 1.0；可加噪声模拟真实检测器）。
    max_detections : int
        最多输出多少个检测。
    jitter_score : float
        给置信度加的高斯噪声标准差（0 = 不加噪）。
    min_visible_corners : int
        至少多少个角点在视野内才输出该检测（过滤"只露出一条边"的框）。
    """

    name = "stub"
    supports_open_vocabulary = False

    def __init__(
        self,
        prompts: Optional[Sequence[str]] = None,
        *,
        filter_by_prompts: bool = True,
        score: float = 1.0,
        max_detections: int = 64,
        jitter_score: float = 0.0,
        min_visible_corners: int = 2,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prompts = list(prompts or [])
        self.filter_by_prompts = bool(filter_by_prompts)
        self.score = float(score)
        self.max_detections = int(max_detections)
        self.jitter_score = float(jitter_score)
        self.min_visible_corners = int(min_visible_corners)
        self._rng = np.random.default_rng(0)

    # ---------------- 主入口 ----------------
    def detect(self, frame: RGBDFrame, prompts: Sequence[str]) -> List[Detection2D]:
        boxes = frame.meta.get("boxes_3d")
        labels = frame.meta.get("labels")

        if boxes is not None and labels is not None and len(labels) > 0:
            return self._detect_from_gt(frame, np.asarray(boxes), list(labels), prompts)

        return self._detect_grid(frame, prompts)

    # ---------------- GT 投影模式 ----------------
    def _detect_from_gt(
        self,
        frame: RGBDFrame,
        boxes: np.ndarray,
        labels: List[str],
        prompts: Sequence[str],
    ) -> List[Detection2D]:
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        if boxes.shape[1] < 6:
            logger.warn(f"boxes_3d 第二维应 ≥6（center+size），收到 {boxes.shape}")
            return []

        use_prompts = list(prompts) if prompts else self.prompts
        keep_idx = self._filter_indices(labels, use_prompts)

        detections: List[Detection2D] = []
        h, w = frame.shape

        for i in keep_idx:
            if len(detections) >= self.max_detections:
                break
            center = boxes[i, :3]
            size = boxes[i, 3:6]
            heading = float(boxes[i, 6]) if boxes.shape[1] > 6 else 0.0

            corners = obb_corners(center, size, heading)
            uv = self._project_corners(corners, frame)
            valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
            if int(valid.sum()) < self.min_visible_corners:
                continue

            vu = uv[valid]
            x1, y1 = float(vu[:, 0].min()), float(vu[:, 1].min())
            x2, y2 = float(vu[:, 0].max()), float(vu[:, 1].max())
            # 裁剪到图像范围
            x1 = max(0.0, min(x1, w - 1.0))
            x2 = max(0.0, min(x2, w - 1.0))
            y1 = max(0.0, min(y1, h - 1.0))
            y2 = max(0.0, min(y2, h - 1.0))
            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue

            score = self.score
            if self.jitter_score > 0:
                score = float(np.clip(score + self._rng.normal(0.0, self.jitter_score), 0.01, 1.0))

            detections.append(Detection2D(
                label=str(labels[i]),
                score=score,
                bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
                prompt=self._matched_prompt(str(labels[i]), use_prompts),
            ))

        return detections

    def _project_corners(self, corners: np.ndarray, frame: RGBDFrame) -> np.ndarray:
        """角点投影到像素；相机后方返回 NaN。

        注意：这里用 `world → camera` 外参（`CameraPose` 的约定），
        深度为负说明在相机后方，必须剔除，否则会得到镜像的错误框。
        """
        cam = frame.pose.world_to_cam(corners)
        z = cam[:, 2]
        uv = np.full((corners.shape[0], 2), np.nan, dtype=np.float64)
        front = z > 1e-6
        safe_z = np.where(front, z, 1.0)
        uv[:, 0] = cam[:, 0] * frame.intrinsics.fx / safe_z + frame.intrinsics.cx
        uv[:, 1] = cam[:, 1] * frame.intrinsics.fy / safe_z + frame.intrinsics.cy
        uv[~front] = np.nan
        return uv

    # ---------------- prompt 过滤 ----------------
    def _filter_indices(self, labels: List[str], prompts: Sequence[str]) -> List[int]:
        idx = list(range(len(labels)))
        if not self.filter_by_prompts or not prompts:
            return idx

        from roboground.mapping.query import LexicalMatcher  # 延迟导入

        matcher = LexicalMatcher()
        kept: List[int] = []
        for prompt in prompts:
            scores = matcher.score(prompt, labels=labels)
            for i in idx:
                if scores[i] > 0.5 and i not in kept:
                    kept.append(i)
        return kept

    def _matched_prompt(self, label: str, prompts: Sequence[str]) -> Optional[str]:
        if not self.filter_by_prompts or not prompts:
            return None
        from roboground.mapping.query import LexicalMatcher  # noqa: PLC0415

        matcher = LexicalMatcher()
        best, best_score = None, 0.5
        for prompt in prompts:
            s = float(matcher.score(prompt, labels=[label])[0])
            if s > best_score:
                best, best_score = prompt, s
        return best

    # ---------------- 网格兜底模式 ----------------
    def _detect_grid(self, frame: RGBDFrame, prompts: Sequence[str]) -> List[Detection2D]:
        """确定性网格框（仅用于接口/形状测试，无语义含义）。"""
        h, w = frame.shape
        use_prompts = list(prompts) if prompts else (self.prompts or ["object"])
        rows, cols = 3, 3
        bh, bw = h / rows, w / cols
        margin = 0.08

        detections: List[Detection2D] = []
        k = 0
        for r in range(rows):
            for c in range(cols):
                if len(detections) >= self.max_detections:
                    break
                x1 = c * bw + bw * margin
                y1 = r * bh + bh * margin
                x2 = (c + 1) * bw - bw * margin
                y2 = (r + 1) * bh - bh * margin
                detections.append(Detection2D(
                    label=str(use_prompts[k % len(use_prompts)]),
                    score=self.score,
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
                    prompt=str(use_prompts[k % len(use_prompts)]),
                ))
                k += 1
        logger.debug("StubDetector 走网格兜底模式（frame.meta 里没有 boxes_3d）")
        return detections

    def __repr__(self) -> str:
        return (
            f"StubDetector(filter_by_prompts={self.filter_by_prompts}, "
            f"max_detections={self.max_detections})"
        )
