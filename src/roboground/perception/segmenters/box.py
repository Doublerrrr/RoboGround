"""几何分割器：把检测框直接当掩码。

为什么需要它？
--------------
1. **降级路径**：没装 SAM 时，整条流水线仍能工作（bbox 反投影出来的
   3D 点会带上框内的背景点，精度下降但可用）；
2. **对照组**：量化"SAM 的精细掩码相比 bbox 带来多少精度提升"——
   这正是我简历里"如果重做会用 SAM mask 而不是 bbox"那句话的实验支撑。

实现要点：掩码必须与图像同尺寸（不是只有框那么大），否则下游
`mask & valid_depth` 的布尔运算会出错。
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

from roboground.perception.base import Segmenter
from roboground.perception.registry import register_segmenter
from roboground.types import Detection2D, RGBDFrame


@register_segmenter("box", "bbox")
class BoxSegmenter(Segmenter):
    """把 bbox 填充成稠密布尔掩码。

    Parameters
    ----------
    shrink : float
        框内缩比例 ∈ [0,1)。略微内缩可以避开框边缘的背景像素，
        对反投影精度有实际帮助（默认 0.0，保持与真实框一致）。
    ellipse : bool
        是否用椭圆掩码而不是矩形（更接近物体的真实形状）。
    """

    name = "box"

    def __init__(self, *, shrink: float = 0.0, ellipse: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.shrink = float(np.clip(shrink, 0.0, 0.9))
        self.ellipse = bool(ellipse)

    def segment(
        self, frame: RGBDFrame, detections: Sequence[Detection2D]
    ) -> List[Detection2D]:
        height, width = frame.shape
        out: List[Detection2D] = []
        for det in detections:
            if det.mask is not None and np.asarray(det.mask).shape == (height, width):
                out.append(det)
                continue
            mask = self._bbox_to_mask(det.bbox, height, width)
            det.mask = mask
            out.append(det)
        return out

    def _bbox_to_mask(self, bbox: np.ndarray, height: int, width: int) -> np.ndarray:
        x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64).reshape(4)

        if self.shrink > 0:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            half_w = (x2 - x1) / 2.0 * (1.0 - self.shrink)
            half_h = (y2 - y1) / 2.0 * (1.0 - self.shrink)
            x1, x2 = cx - half_w, cx + half_w
            y1, y2 = cy - half_h, cy + half_h

        # 裁剪到图像
        xi1 = int(np.clip(np.floor(x1), 0, width))
        xi2 = int(np.clip(np.ceil(x2), 0, width))
        yi1 = int(np.clip(np.floor(y1), 0, height))
        yi2 = int(np.clip(np.ceil(y2), 0, height))

        mask = np.zeros((height, width), dtype=bool)
        if xi2 <= xi1 or yi2 <= yi1:
            return mask

        if not self.ellipse:
            mask[yi1:yi2, xi1:xi2] = True
            return mask

        # 椭圆掩码
        yy, xx = np.mgrid[yi1:yi2, xi1:xi2]
        cx = (xi1 + xi2 - 1) / 2.0
        cy = (yi1 + yi2 - 1) / 2.0
        rx = max((xi2 - xi1) / 2.0, 1e-6)
        ry = max((yi2 - yi1) / 2.0, 1e-6)
        inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        sub = np.zeros((yi2 - yi1, xi2 - xi1), dtype=bool)
        sub[inside] = True
        mask[yi1:yi2, xi1:xi2] = sub
        return mask
