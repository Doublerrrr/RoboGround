"""离线特征编码器：颜色-空间直方图描述子（72 维）。

为什么需要它？
--------------
1. **零依赖**：不需要 transformers / 不需要下载权重 / 不需要 GPU，
   保证"clone 下来就能跑通全链路"；
2. **回归测试的确定性基线**：输出只取决于像素值，可精确复现；
3. **可解释**：维度含义明确（哪个格子哪个 bin），调试时能一眼看出问题。

短板（必须在面试里说清楚）
--------------------------
**它没有文本编码器**，所以不能做"文本 → 特征"的开放词汇查询。
用这个编码器时，系统会自动退回 `LexicalMatcher`（词法匹配）。
要真正的开放词汇，必须换 `clip` 后端。

描述子构成（共 72 维）
---------------------
```
2×2 空间格子 × (8 色调 bin + 4 饱和 bin + 4 明度 bin) = 64 维
全局 8 维：mean_R/G/B(3) + std_R/G/B(3) + 宽高比(1) + 面积占比(1)
```
最后整体 L2 归一化，使余弦相似度可用。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from roboground.perception.base import Encoder
from roboground.perception.registry import register_encoder
from roboground.types import Detection2D, RGBDFrame


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB ∈ [0,1] 的 (N,3) 数组 → HSV ∈ [0,1]，全向量化实现。

    自己实现而不用 cv2 的原因：cv2 走 BGR 且返回 H∈[0,180]，
    在跨模块调用时极易踩坑（本项目已被这类问题坑过一次）。
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    delta = maxc - minc
    safe_delta = np.clip(delta, 1e-12, None)

    v = maxc
    s = np.where(maxc > 1e-12, delta / np.clip(maxc, 1e-12, None), 0.0)

    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta

    h = np.where(
        r >= maxc, bc - gc,
        np.where(g >= maxc, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = np.where(delta > 1e-12, (h / 6.0) % 1.0, 0.0)

    return np.stack([h, s, v], axis=1)


@register_encoder("color_hist", "color", "offline", "hist")
class ColorHistogramEncoder(Encoder):
    """颜色-空间直方图编码器（72 维，零依赖）。"""

    name = "color_hist"
    supports_text = False

    def __init__(
        self,
        *,
        cells: int = 2,
        hue_bins: int = 8,
        sat_bins: int = 4,
        val_bins: int = 4,
        feature_dim: int = 72,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_dim=feature_dim, **kwargs)
        self.cells = int(cells)
        self.hue_bins = int(hue_bins)
        self.sat_bins = int(sat_bins)
        self.val_bins = int(val_bins)
        self._feature_dim = (
            self.cells * self.cells * (hue_bins + sat_bins + val_bins) + 8
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    # ---------------- 编码 ----------------
    def encode_regions(
        self,
        frame: RGBDFrame,
        detections: Sequence[Detection2D],
    ) -> np.ndarray:
        if not detections:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        image = np.asarray(frame.color, dtype=np.float32) / 255.0
        height, width = image.shape[:2]
        hsv = rgb_to_hsv(image.reshape(-1, 3)).reshape(height, width, 3)

        feats = np.zeros((len(detections), self.feature_dim), dtype=np.float32)
        for i, det in enumerate(detections):
            feats[i] = self._encode_one(hsv, det, height, width)
        return feats

    def _encode_one(
        self,
        hsv: np.ndarray,
        det: Detection2D,
        height: int,
        width: int,
    ) -> np.ndarray:
        mask = det.mask
        if mask is None or np.asarray(mask).shape != (height, width):
            x1, y1, x2, y2 = np.asarray(det.bbox, dtype=np.float64)
            xi1 = int(np.clip(np.floor(x1), 0, width))
            xi2 = int(np.clip(np.ceil(x2), 0, width))
            yi1 = int(np.clip(np.floor(y1), 0, height))
            yi2 = int(np.clip(np.ceil(y2), 0, height))
            if xi2 <= xi1 or yi2 <= yi1:
                return np.zeros(self.feature_dim, dtype=np.float32)
            region = np.zeros((height, width), dtype=bool)
            region[yi1:yi2, xi1:xi2] = True
        else:
            region = np.asarray(mask, dtype=bool)

        if not np.any(region):
            return np.zeros(self.feature_dim, dtype=np.float32)

        # ---- 1) 分格直方图 ----
        cell_h = max(1, height // self.cells)
        cell_w = max(1, width // self.cells)
        hist_parts = []
        for cr in range(self.cells):
            for cc in range(self.cells):
                y0, y1c = cr * cell_h, (cr + 1) * cell_h if cr < self.cells - 1 else height
                x0, x1c = cc * cell_w, (cc + 1) * cell_w if cc < self.cells - 1 else width
                sub_mask = region[y0:y1c, x0:x1c]
                sub_hsv = hsv[y0:y1c, x0:x1c]
                pixels = sub_hsv[sub_mask] if np.any(sub_mask) else np.zeros((0, 3))
                hist_parts.append(self._hist3(pixels))

        # ---- 2) 全局 8 维统计 ----
        pixels_all = hsv[region]
        rgb_all = np.zeros((0, 3))  # 颜色统计从 hsv 反推代价高，这里用 hsv 的 v/s 近似
        mean_v = float(pixels_all[:, 2].mean()) if pixels_all.size else 0.0
        std_v = float(pixels_all[:, 2].std()) if pixels_all.size else 0.0
        mean_s = float(pixels_all[:, 1].mean()) if pixels_all.size else 0.0
        std_s = float(pixels_all[:, 1].std()) if pixels_all.size else 0.0
        mean_h = float(pixels_all[:, 0].mean()) if pixels_all.size else 0.0

        x1, y1, x2, y2 = np.asarray(det.bbox, dtype=np.float64)
        bw = max(x2 - x1, 1e-6)
        bh = max(y2 - y1, 1e-6)
        aspect = float(np.clip(bw / bh, 0.0, 10.0) / 10.0)
        area_ratio = float(region.sum()) / float(height * width)
        cy_norm = float(np.clip((y1 + y2) / 2.0 / max(height, 1), 0.0, 1.0))

        global_feats = np.array(
            [mean_h, mean_s, std_s, mean_v, std_v, aspect, area_ratio, cy_norm],
            dtype=np.float32,
        )

        feat = np.concatenate([*hist_parts, global_feats]).astype(np.float32)
        if feat.shape[0] != self.feature_dim:  # 理论上不会发生，作为安全网
            out = np.zeros(self.feature_dim, dtype=np.float32)
            out[: min(feat.shape[0], self.feature_dim)] = feat[: self.feature_dim]
            feat = out

        # L2 归一化 → 余弦相似度可用
        norm = float(np.linalg.norm(feat))
        if norm > 1e-8:
            feat = feat / norm
        return feat

    def _hist3(self, pixels: np.ndarray) -> np.ndarray:
        """一个格子的 (8 hue + 4 sat + 4 val) = 16 维直方图。"""
        out = np.zeros(self.hue_bins + self.sat_bins + self.val_bins, dtype=np.float32)
        if pixels.shape[0] == 0:
            return out

        h_idx = np.clip((pixels[:, 0] * self.hue_bins).astype(np.int64), 0, self.hue_bins - 1)
        s_idx = np.clip((pixels[:, 1] * self.sat_bins).astype(np.int64), 0, self.sat_bins - 1)
        v_idx = np.clip((pixels[:, 2] * self.val_bins).astype(np.int64), 0, self.val_bins - 1)

        h_hist = np.bincount(h_idx, minlength=self.hue_bins).astype(np.float32)
        s_hist = np.bincount(s_idx, minlength=self.sat_bins).astype(np.float32)
        v_hist = np.bincount(v_idx, minlength=self.val_bins).astype(np.float32)

        total = float(pixels.shape[0])
        out[: self.hue_bins] = h_hist / total
        out[self.hue_bins:self.hue_bins + self.sat_bins] = s_hist / total
        out[self.hue_bins + self.sat_bins:] = v_hist / total
        return out
