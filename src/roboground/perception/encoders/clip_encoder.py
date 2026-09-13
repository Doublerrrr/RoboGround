"""CLIP 编码器：**唯一支持真正开放词汇查询的后端**（图像 ↔ 文本同空间）。

关键差异（面试要讲清）
--------------------
| 后端 | 图像特征 | 文本特征 | 能否"任意文本查 3D" |
|---|---|---|---|
| color_hist | ✅ 颜色直方图 | ❌ | ❌ 只能词法匹配 |
| dinov2 | ✅ 密集语义 | ❌ | ❌ 只能词法匹配 |
| **clip** | ✅ 图文对齐空间 | ✅ 同空间 | ✅ **真·开放词汇** |

因为 CLIP 的图像特征和文本特征被对比学习对齐到**同一个 512/768 维空间**，
所以"用文本 query 去查 3D 特征场"在数学上才成立（余弦相似度有意义）。
这是整个 RoboGround "开放词汇"名号的根基。

依赖：`pip install -e ".[perception]"`；权重默认 `openai/clip-vit-base-patch32`（~600MB）。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from roboground.perception.base import Encoder
from roboground.perception.encoders._region import region_to_square_crop
from roboground.perception.registry import register_encoder
from roboground.types import Detection2D, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("perception.clip")


@register_encoder("clip")
class CLIPEncoder(Encoder):
    """CLIP 图像/文本双塔编码器（懒加载）。

    Parameters
    ----------
    pool : str
        `"crop"` —— 按 bbox 裁剪后再编码（推荐，符合 CLIP 的训练分布）；
        `"mask"` —— 用掩码把非目标像素涂成均值色再编码（更干净但对
        细长物体不友好）；
        `"full"` —— 直接编码整图（只在没有 bbox 时用）。
    crop_margin : float
        裁剪时向外扩的比例，避免把物体边缘切掉。
    """

    name = "clip"
    supports_text = True

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        *,
        pool: str = "crop",
        crop_margin: float = 0.05,
        square: bool = True,
        max_regions: int = 64,
        device: Optional[str] = None,
        feature_dim: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_dim=feature_dim, **kwargs)
        self.model_id = model_id
        self.pool = str(pool)
        self.crop_margin = float(crop_margin)
        self.square = bool(square)
        self.max_regions = int(max_regions)
        self._device = device
        self._processor = None
        self._model = None
        self._dim = int(feature_dim)
        self._cache: dict = {}

    @property
    def feature_dim(self) -> int:
        return self._dim

    # ---------------- 懒加载 ----------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CLIPEncoder 需要 transformers：pip install -e \".[perception]\""
            ) from exc

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"加载 CLIP：{self.model_id}（device={self._device}）")
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id)
        self._model = self._model.to(self._device).eval()

        proj = getattr(self._model.config, "projection_dim", None)
        if proj:
            self._dim = int(proj)

    def warmup(self) -> None:
        self._ensure_loaded()

    # ---------------- 图像侧 ----------------
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

        height, width = frame.shape
        image = Image.fromarray(np.asarray(frame.color, dtype=np.uint8))

        crops: List[Image.Image] = []
        valid_idx: List[int] = []
        for i, det in enumerate(detections[: self.max_regions]):
            crop = self._make_crop(image, det, height, width)
            if crop is None:
                continue
            crops.append(crop)
            valid_idx.append(i)

        out = np.zeros((len(detections), self.feature_dim), dtype=np.float32)
        if not crops:
            return out

        inputs = self._processor(images=crops, return_tensors="pt")
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)

        vecs = feats.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-8, None)

        for row, i in enumerate(valid_idx):
            out[i] = vecs[row]
        return out

    def _make_crop(self, image: np.ndarray, det: Detection2D, height: int, width: int):
        """按 bbox（+context）裁剪，并**扩成正方形**。

        为什么要扩成正方形：CLIP 的预处理会把输入 resize 成 224×224。
        非方形的裁剪（比如 3:1 的细长书架）会被**拉伸变形**，
        而变形后的"书架"和"桌子"在 CLIP 眼里可能差不多 ——
        这是实测中区域特征区分度差的原因之一。
        """
        from PIL import Image  # noqa: PLC0415

        if self.pool == "full":
            return Image.fromarray(np.asarray(image, dtype=np.uint8))

        return region_to_square_crop(
            np.asarray(image, dtype=np.uint8),
            det.bbox,
            mask=det.mask,
            context=self.crop_margin,
            square=self.square,
            mask_pool=(self.pool == "mask"),
        )

    # ---------------- 文本侧 ----------------
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """编码文本 → (N, feature_dim)，与图像特征同空间。

        这是"开放词汇查询"的另一半：文本特征和建图时的图像特征做余弦相似度。
        """
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)

        vecs = feats.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """编码整张图（"以图搜物"场景）。"""
        self._ensure_loaded()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        vec = feats.detach().cpu().numpy().astype(np.float32)[0]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else vec
