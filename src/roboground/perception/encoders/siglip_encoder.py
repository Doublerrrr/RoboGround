"""SigLIP 编码器：比 CLIP 更适合"区域级开放词汇"，且**自带校准**。

为什么在 CLIP 之外再加 SigLIP（这是实测后的结论，不是追新）
--------------------------------------------------------
在真实 SUN RGB-D 场景上做了池化策略消融（`scripts/11_ablate_clip_pooling.py`），
用 CLIP ViT-B/32 得到的结果是：

| 池化 | self_sim | other_sim | 分离度 | Top-1 |
|---|---|---|---|---|
| crop | 0.246 | 0.252 | **-0.006** | 0.34 |
| mask | 0.236 | 0.250 | **-0.015** | 0.23 |
| full | 0.224 | 0.233 | **-0.009** | 0.31 |

**分离度为负**意味着"和别的类别反而更像"，Top-1 只有 0.34（5~6 类，随机约 0.18）——
即 CLIP ViT-B/32 的区域特征**基本无法区分这些家具类别**。

SigLIP 的两个优势正好对症：
1. **patch16 + sigmoid 损失**：SigLIP 用 sigmoid 而不是 softmax 做图文对比，
   每个图文对得到一个**独立的匹配概率**，而不是"相对谁更像"。
   这天然适合"拒识"（地图里没有的物体就该得到低概率），
   不需要我们再去搞空文本校准。
2. 分类/检索性能显著强于同量级 CLIP（SigLIP 论文的核心结论）。

实现要点
--------
校准公式（来自 SigLIP 论文与 HF 实现）::

    prob = sigmoid(logit_scale * cos_sim + logit_bias)

`logit_scale` / `logit_bias` 是模型自带的**可学习参数**，所以这里是**模型原生校准**，
比我手写的空文本 softmax 更可靠。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from roboground.perception.base import Encoder
from roboground.perception.encoders._region import build_region_views, region_to_square_crop
from roboground.perception.registry import register_encoder
from roboground.types import Detection2D, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("perception.siglip")


@register_encoder("siglip", "siglip-base")
class SigLIPEncoder(Encoder):
    """SigLIP 图像/文本双塔编码器（懒加载，自带 sigmoid 校准）。"""

    name = "siglip"
    supports_text = True

    #: SigLIP 官方推荐的零样本模板（比 CLIP 的模板简单）
    DEFAULT_TEMPLATES = (
        "a photo of a {}.",
        "a photo of the {}.",
        "a cropped photo of a {}.",
    )

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        *,
        pool: str = "crop",
        crop_margin: float = 0.15,
        square: bool = True,
        tta: str = "none",
        batch_size: int = 32,
        max_regions: int = 64,
        device: Optional[str] = None,
        feature_dim: int = 768,
        templates: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_dim=feature_dim, **kwargs)
        self.model_id = model_id
        self.pool = str(pool)
        self.crop_margin = float(crop_margin)
        self.square = bool(square)
        #: 多视图 TTA：none | flip | multicrop（见 `_region.build_region_views`）
        self.tta = str(tta)
        self.batch_size = int(batch_size)
        self.max_regions = int(max_regions)
        self.templates = tuple(templates) if templates else self.DEFAULT_TEMPLATES
        self._device = device
        self._processor = None
        self._model = None
        self._dim = int(feature_dim)
        self._logit_scale: Optional[float] = None
        self._logit_bias: Optional[float] = None

    @property
    def feature_dim(self) -> int:
        return self._dim

    # ---------------- 能力声明 ----------------
    @property
    def is_calibrated(self) -> bool:
        """SigLIP 是模型原生校准的（`pair_scores` 返回概率而非余弦）。"""
        return True

    @property
    def suggested_pair_threshold(self) -> float:
        """**hybrid 模式**的建议阈值 = **0.02**（不是 0.5！）。

        实测（`scripts/11_ablate_clip_pooling.py`，真实 SUN RGB-D）：
        SigLIP-base 的 sigmoid 概率非常保守 ——
        即使是**正确类别**的图文对，概率也只有 0.01~0.05 量级
        （logit_bias = -12.93 把整体压得很低）。
        所以阈值必须按这个尺度设，否则所有查询都会返回空。

        在 hybrid 模式下它充当**精度过滤器**：词法已负责接受判定，
        所以收紧它不损失命中、却能减少假接受（实测 hybrid 综合分 0.834 > 0.5 阈值下纯词法的 0.828）。
        """
        return 0.02

    @property
    def suggested_standalone_threshold(self) -> float:
        """**纯嵌入模式**的回退阈值 = **5e-04**。

        ⚠️ 现在这只是**回退值**：正常情况下 `QueryEngine` 会用
        **自监督标定**（`_calibrate_standalone_threshold`）就地算出阈值，
        只有在标定不可行时（物体少于 2 个 / 只有一个类别 / 编码器异常）
        才会用到这里。
        实测自监督标定**优于**这个硬编码值（综合分 0.232 → 0.291），
        详见 `scripts/16_eval_openvocab_query.py` 第四节。

        与 hybrid 用的 0.02 差 **40 倍** —— 这不是笔误，是**角色不同**：
        纯嵌入时这个阈值就是唯一的接受判定，定高一点就整体拒空。

        实测阈值曲线（比对的是**多视角融合后的地图物体特征**，而非检测区域特征）：

        | 阈值 | 命名类命中 | 描述类命中 | 拒识率 | 综合分 |
        |---|---|---|---|---|
        | 5e-05 | 41.4% | **20.0%** | 59.7% | 0.214 |
        | **5e-04** | 31.5% | 12.5% | **87.5%** | **0.232** |
        | 0.001 | 27.9% | 10.0% | 91.7% | 0.212 |
        | 0.02（hybrid 值） | 3.6% | **2.5%** | 100.0% | 0.033 |

        用 0.02 跑纯嵌入会把描述类查询几乎全拒空（20.0% → 2.5%，差 8 倍）——
        这是本项目实测发现的真实缺陷。
        """
        return 5e-04

    # ---------------- 懒加载 ----------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModel, AutoProcessor  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'SigLIPEncoder 需要 transformers：pip install -e ".[perception]"'
            ) from exc

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"加载 SigLIP：{self.model_id}（device={self._device}）")
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()
        except Exception as exc:
            # 新版 transformers 把 SigLIP 拆成了 SiglipModel；做一次兼容回退
            from transformers import SiglipModel, SiglipProcessor  # noqa: PLC0415

            logger.debug(f"AutoModel 加载失败（{exc}），改用 SiglipModel")
            self._processor = SiglipProcessor.from_pretrained(self.model_id)
            self._model = SiglipModel.from_pretrained(self.model_id).to(self._device).eval()

        cfg = self._model.config
        # SigLIP 的投影维度（text_config / vision_config 里都可能有）
        for attr in ("projection_size", "hidden_size"):
            val = getattr(cfg, attr, None) or getattr(getattr(cfg, "text_config", None), attr, None)
            if isinstance(val, int):
                self._dim = int(val)
                break

        # 读取模型自带的校准参数
        try:
            self._logit_scale = float(self._model.logit_scale.exp().item())
            self._logit_bias = float(self._model.logit_bias.item())
            logger.info(f"SigLIP 校准参数：logit_scale={self._logit_scale:.2f} "
                        f"logit_bias={self._logit_bias:.2f}")
        except Exception:
            self._logit_scale, self._logit_bias = 100.0, 0.0
            logger.debug("未读到 logit_scale/logit_bias，使用默认值")

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

        color = np.asarray(frame.color, dtype=np.uint8)
        subset = list(detections[: self.max_regions])

        # ---- 构造多视图（TTA）：每个检测可展开成多个裁剪 ----
        all_views: List[Any] = []
        groups: List[int] = []
        for i, det in enumerate(subset):
            if self.pool == "full":
                views = [Image.fromarray(color)]
            else:
                views = build_region_views(
                    color, det.bbox,
                    mask=det.mask,
                    tta=self.tta,
                    context=self.crop_margin,
                    square=self.square,
                    mask_pool=(self.pool == "mask"),
                )
            all_views.extend(views)
            groups.extend([i] * len(views))

        out = np.zeros((len(detections), self.feature_dim), dtype=np.float32)
        if not all_views:
            return out

        # 分批编码：TTA 会把视图数放大 2~8 倍，一次全送容易 OOM
        vecs_list = []
        for start in range(0, len(all_views), self.batch_size):
            chunk = all_views[start:start + self.batch_size]
            batch = self._processor(images=chunk, return_tensors="pt")
            batch = {k: (v.to(self._device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            with torch.no_grad():
                feats = self._model.get_image_features(**batch)
            vecs_list.append(feats.detach().cpu().numpy().astype(np.float32))

        vecs = np.concatenate(vecs_list, axis=0)
        vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)

        # ---- 同一检测的多视图特征平均后再归一化 ----
        groups_arr = np.asarray(groups, dtype=np.int64)
        for i in range(len(subset)):
            sel = vecs[groups_arr == i]
            if sel.shape[0] == 0:
                continue
            mean = sel.mean(axis=0)
            n = float(np.linalg.norm(mean))
            out[i] = mean / n if n > 1e-8 else mean
        return out

    # ---------------- 文本侧 ----------------
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        batch = self._processor(
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        )
        batch = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            feats = self._model.get_text_features(**batch)

        vecs = feats.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)

    # ---------------- 校准打分 ----------------
    def pair_scores(self, image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """**模型原生校准的匹配概率**（不是余弦）。

        `prob = sigmoid(logit_scale * cos + logit_bias)`

        与 CLIP 的余弦相比，这个分数的好处是：
        - 有绝对含义（"这就是这一个类别的概率"），可以直接设阈值；
        - 不需要空文本校准；
        - 天然具备拒识能力（不匹配的类别会掉到 0.1 以下）。
        """
        self._ensure_loaded()
        img = np.asarray(image_features, dtype=np.float32)
        txt = np.asarray(text_features, dtype=np.float32)
        if img.ndim == 1:
            img = img.reshape(1, -1)
        if txt.ndim == 1:
            txt = txt.reshape(1, -1)

        # 归一化后点积 = 余弦
        img = img / np.clip(np.linalg.norm(img, axis=1, keepdims=True), 1e-8, None)
        txt = txt / np.clip(np.linalg.norm(txt, axis=1, keepdims=True), 1e-8, None)

        scale = float(self._logit_scale if self._logit_scale is not None else 100.0)
        bias = float(self._logit_bias if self._logit_bias is not None else 0.0)
        logits = scale * (img @ txt.T) + bias
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        batch = self._processor(images=Image.fromarray(np.asarray(image, dtype=np.uint8)),
                                return_tensors="pt")
        batch = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            feats = self._model.get_image_features(**batch)
        vec = feats.detach().cpu().numpy().astype(np.float32)[0]
        n = float(np.linalg.norm(vec))
        return vec / n if n > 1e-8 else vec
