"""感知层抽象接口。

三个可插拔组件
--------------
```
Detector   图像 + 文本 prompt  →  2D 检测框（开放词汇的关键入口）
Segmenter  图像 + 检测框       →  实例掩码
Encoder    图像 + 区域         →  语义特征向量（升维到 3D 的"语义载体"）
```

设计原则
--------
1. **每个组件都必须有离线降级实现** —— 没装模型/没网也能跑通全链路；
2. **接口尽量窄** —— 只有 `detect` / `segment` / `encode_regions` 三个方法，
   方便替换成任意模型；
3. **明确声明能力** —— `supports_text` / `feature_dim` 让上层能自动决策
   （比如能不能做嵌入查询、体素特征该多宽）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from roboground.types import Detection2D, RGBDFrame


# ==========================================================================
# 检测
# ==========================================================================
class Detector(ABC):
    """开放词汇检测器接口。"""

    name: str = "detector"
    supports_open_vocabulary: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)

    @abstractmethod
    def detect(self, frame: RGBDFrame, prompts: Sequence[str]) -> List[Detection2D]:
        """在图像上检测 `prompts` 指定的目标。

        Parameters
        ----------
        frame
            输入帧（至少需要 `color`）。
        prompts
            自然语言类别描述，如 `["cup", "table"]`。

        Returns
        -------
        List[Detection2D]
            检测结果（未做分割，`mask` 通常为 None）。
        """

    def warmup(self) -> None:
        """可选：预加载模型权重（懒加载后端在首次 detect 时才加载）。"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ==========================================================================
# 分割
# ==========================================================================
class Segmenter(ABC):
    """实例分割器接口。"""

    name: str = "segmenter"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)

    @abstractmethod
    def segment(
        self, frame: RGBDFrame, detections: Sequence[Detection2D]
    ) -> List[Detection2D]:
        """为每个检测框生成实例掩码（原地或返回新对象均可）。

        Returns
        -------
        List[Detection2D]
            与输入等长、且 `mask` 已填充的检测列表。
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ==========================================================================
# 特征编码
# ==========================================================================
class Encoder(ABC):
    """区域特征编码器接口。

    这是"开放词汇"能否成立的关键：只有图像特征和文本特征落在**同一个空间**
    （CLIP 类），才能用任意文本去查询 3D 地图。DINOv2 / 颜色直方图没有文本
    编码器，此时系统会自动退回词法匹配（见 `mapping.query.LexicalMatcher`）。
    """

    name: str = "encoder"
    supports_text: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self._feature_dim: int = int(kwargs.get("feature_dim", 256))

    # ---------------- 能力声明 ----------------
    @property
    def feature_dim(self) -> int:
        """输出特征维度。子类用自己的原生维度覆盖它。"""
        return self._feature_dim

    # ---------------- 核心方法 ----------------
    @abstractmethod
    def encode_regions(
        self,
        frame: RGBDFrame,
        detections: Sequence[Detection2D],
    ) -> np.ndarray:
        """为每个检测区域编码一个特征向量。

        Returns
        -------
        np.ndarray, shape (M, feature_dim), float32
            M = len(detections)。M=0 时返回 (0, feature_dim)。
        """

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """编码文本（仅 `supports_text=True` 的后端支持）。

        Raises
        ------
        NotImplementedError
            当前后端不具备文本编码能力。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 不支持文本编码；"
            "开放词汇查询请使用 clip / siglip 后端，或依赖词法匹配（LexicalMatcher）"
        )

    @property
    def is_calibrated(self) -> bool:
        """该后端的图文分数是否**已经是校准概率**。

        - `False`（默认，如 CLIP）：`pair_scores` 返回余弦相似度，
          需要上层做额外校准（本项目用空文本 softmax）；
        - `True`（如 SigLIP）：`pair_scores` 直接返回匹配概率，
          因为模型是用 sigmoid 损失训的，`logit_scale`/`logit_bias` 就是原生校准。
        """
        return False

    @property
    def suggested_pair_threshold(self) -> float:
        """该后端"算命中"的建议阈值 —— **必须按编码器区分**。

        这是实测踩出来的：不同后端的分数尺度差了两个数量级
        （见 `scripts/11_ablate_clip_pooling.py`）：

        | 后端 | 分数类型 | 典型"命中"分数 | 建议阈值 |
        |---|---|---|---|
        | `color_hist` / `dinov2` | 无文本能力 | — | — |
        | `clip` | 余弦 | 0.25 ~ 0.30 | 用空文本校准后的概率，见 `EmbeddingMatcher` |
        | `siglip` | sigmoid 概率 | **0.01 ~ 0.05** | **0.02** |

        如果对 SigLIP 用 0.5 的阈值，**所有查询都会返回空** ——
        因为它的 sigmoid 概率天然保守（实测命中样本也只有 0.037）。
        """
        return 0.5

    @property
    def suggested_standalone_threshold(self) -> float:
        """**只有嵌入这一路可用时**的建议阈值（默认与 `suggested_pair_threshold` 相同）。

        为什么需要和上一个属性分开？因为**同一个阈值的角色会随流水线变化**：

        - **hybrid（词法 + 嵌入）**：词法已经负责"接不接受"，
          嵌入阈值退化为**精度过滤器** —— 收紧它不损失命中，却能减少假接受；
        - **纯嵌入**：嵌入阈值就是**接受阈值**，收紧它直接砍掉召回。

        实测（`scripts/16_eval_openvocab_query.py`，SigLIP + 融合后的地图物体特征）：
        两种角色的最优阈值差 **40 倍**（hybrid 0.02 vs 纯嵌入 0.0005）。
        用同一角色的值去跑另一种角色，纯嵌入模式下描述类查询命中率会从
        12.5% 掉到 **2.5%**。

        ⚠️ 注意这个值是在**地图物体特征**（多视角融合后）上标定的，
        与检测区域特征的尺度差约 20 倍 —— 标定基准必须和实际比对的特征一致。
        """
        return self.suggested_pair_threshold

    def pair_scores(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
    ) -> np.ndarray:
        """图文对的匹配分数矩阵 (M, N)。

        默认实现是**余弦相似度**（作用于 L2 归一化后的特征）。
        具备原生校准能力的后端（SigLIP）应覆写它，直接返回概率。
        """
        img = np.asarray(image_features, dtype=np.float32)
        txt = np.asarray(text_features, dtype=np.float32)
        if img.ndim == 1:
            img = img.reshape(1, -1)
        if txt.ndim == 1:
            txt = txt.reshape(1, -1)
        img = img / np.clip(np.linalg.norm(img, axis=1, keepdims=True), 1e-8, None)
        txt = txt / np.clip(np.linalg.norm(txt, axis=1, keepdims=True), 1e-8, None)
        return (img @ txt.T).astype(np.float32)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """编码整张图（用于把参考图作为 query 的"以图搜物"场景）。"""
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 encode_image"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"dim={self.feature_dim}, text={self.supports_text})"
        )


# ==========================================================================
# 组合管线
# ==========================================================================
class PerceptionPipeline:
    """把 检测 → 分割 → 编码 三段串成一条流水线。

    Examples
    --------
    >>> pipe = PerceptionPipeline(detector, segmenter, encoder, prompts=["cup"])
    >>> detections = pipe(frame)
    >>> detections[0].feature.shape     # doctest: +SKIP
    (72,)
    """

    def __init__(
        self,
        detector: Detector,
        segmenter: Optional[Segmenter] = None,
        encoder: Optional[Encoder] = None,
        *,
        prompts: Optional[Sequence[str]] = None,
        with_features: bool = True,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.encoder = encoder
        self.prompts = list(prompts or [])
        self.with_features = bool(with_features)
        self.stats: Dict[str, float] = {}

    def __call__(self, frame: RGBDFrame, prompts: Optional[Sequence[str]] = None) -> List[Detection2D]:
        return self.run(frame, prompts=prompts)

    def run(
        self,
        frame: RGBDFrame,
        prompts: Optional[Sequence[str]] = None,
    ) -> List[Detection2D]:
        import time  # noqa: PLC0415

        use_prompts = list(prompts) if prompts is not None else self.prompts

        t0 = time.perf_counter()
        detections = self.detector.detect(frame, use_prompts)
        t1 = time.perf_counter()

        if self.segmenter is not None and detections:
            detections = self.segmenter.segment(frame, detections)
        t2 = time.perf_counter()

        if self.with_features and self.encoder is not None and detections:
            features = self.encoder.encode_regions(frame, detections)
            features = np.asarray(features, dtype=np.float32)
            for det, feat in zip(detections, features):
                det.feature = feat
        t3 = time.perf_counter()

        self.stats = {
            "num_detections": float(len(detections)),
            "detect_s": t1 - t0,
            "segment_s": t2 - t1,
            "encode_s": t3 - t2,
            "total_s": t3 - t0,
        }
        return detections

    def profile(self) -> Dict[str, float]:
        """最近一次 run 的耗时拆解（秒）。"""
        return dict(self.stats)

    def __repr__(self) -> str:
        return (
            f"PerceptionPipeline(detector={self.detector.name!r}, "
            f"segmenter={self.segmenter.name if self.segmenter else None!r}, "
            f"encoder={self.encoder.name if self.encoder else None!r}, "
            f"prompts={len(self.prompts)})"
        )
