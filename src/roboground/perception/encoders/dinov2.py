"""DINOv2 特征编码器：patch 级密集语义特征 + 掩码池化。

为什么用它（而不是只用 CLIP）
----------------------------
- **空间精度高**：DINOv2 的特征是 patch 级的（还记得 `patch 级 token 对齐`
  那个机制吗），所以"同一物体内部的 patch 特征接近、跨物体边界差异大"，
  这让它在**区域池化**后得到的实例特征非常干净；
- **不用文本**：它没有文本编码器，所以本项目里它主要用于
  ①给 3D 特征场提供高质量语义；②作为"如果有物体级检索需求"的特征源。
  做文本查询时系统会自动退回词法匹配（或改用 clip 后端）。

依赖：`pip install -e ".[perception]"`
权重默认 `facebook/dinov2-small`（~85MB，384 维）。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from roboground.perception.base import Encoder
from roboground.perception.registry import register_encoder
from roboground.types import Detection2D, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("perception.dinov2")


@register_encoder("dinov2", "dino")
class DINOv2Encoder(Encoder):
    """DINOv2 patch 特征 + 掩码加权池化（懒加载）。"""

    name = "dinov2"
    supports_text = False

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        *,
        pool: str = "mask_mean",
        max_regions: int = 64,
        device: Optional[str] = None,
        feature_dim: int = 384,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_dim=feature_dim, **kwargs)
        self.model_id = model_id
        self.pool = str(pool)
        self.max_regions = int(max_regions)
        self._device = device
        self._processor = None
        self._model = None
        self._dim = int(feature_dim)

    @property
    def feature_dim(self) -> int:
        return self._dim

    # ---------------- 懒加载 ----------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DINOv2Encoder 需要 transformers：pip install -e \".[perception]\""
            ) from exc

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"加载 DINOv2：{self.model_id}（device={self._device}）")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        self._model = self._model.to(self._device).eval()

        # 从 config 读真实维度，避免 config 里写错导致维度不匹配
        hidden = getattr(self._model.config, "hidden_size", None)
        if hidden:
            self._dim = int(hidden)

    def warmup(self) -> None:
        self._ensure_loaded()

    # ---------------- 推理 ----------------
    def encode_regions(
        self,
        frame: RGBDFrame,
        detections: Sequence[Detection2D],
    ) -> np.ndarray:
        if not detections:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        self._ensure_loaded()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        image = Image.fromarray(np.asarray(frame.color, dtype=np.uint8))
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        hidden = outputs.last_hidden_state            # (1, 1+N, D)，第 0 个是 CLS
        tokens = hidden[:, 1:, :]                     # 去掉 CLS
        num_patches = int(tokens.shape[1])
        grid = int(round(num_patches ** 0.5))
        if grid * grid != num_patches:
            logger.warn(
                f"patch 数 {num_patches} 不是完全平方数，无法还原成方形网格；"
                "退回全局均值池化"
            )
            grid = 0

        if grid > 0:
            tokens = tokens.reshape(1, grid, grid, -1)
        else:
            pooled_global = tokens.mean(dim=1)[0]

        height, width = frame.shape
        out = np.zeros((len(detections), self.feature_dim), dtype=np.float32)

        for i, det in enumerate(detections[: self.max_regions]):
            mask = det.mask
            if mask is None or np.asarray(mask).shape != (height, width):
                m = np.zeros((height, width), dtype=np.uint8)
                x1, y1, x2, y2 = np.asarray(det.bbox, dtype=np.float64)
                xi1 = int(np.clip(np.floor(x1), 0, width))
                xi2 = int(np.clip(np.ceil(x2), 0, width))
                yi1 = int(np.clip(np.floor(y1), 0, height))
                yi2 = int(np.clip(np.ceil(y2), 0, height))
                if xi2 > xi1 and yi2 > yi1:
                    m[yi1:yi2, xi1:xi2] = 1
            else:
                m = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255

            if m.sum() == 0 or grid == 0:
                feat = pooled_global if grid == 0 else tokens.reshape(1, -1, self.feature_dim).mean(dim=1)[0]
            else:
                # 掩码缩放到 patch 网格，作为软权重（BILINEAR 给出覆盖率）
                mask_img = Image.fromarray(m).resize((grid, grid), Image.BILINEAR)
                weights = torch.from_numpy(
                    np.asarray(mask_img, dtype=np.float32) / 255.0
                ).to(tokens.device)                                    # (grid, grid)
                w_sum = float(weights.sum().item())
                if w_sum < 1e-6:
                    feat = tokens.reshape(1, -1, self.feature_dim).mean(dim=1)[0]
                else:
                    weighted = (tokens[0] * weights[:, :, None]).sum(dim=(0, 1)) / w_sum
                    feat = weighted

            vec = feat.detach().cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
            out[i] = vec

        return out
