"""开放词汇检测：Grounding DINO（文本 prompt → 检测框）。

这是 Stage 1 的**主力后端** —— 服务机器人要"用户问什么就找什么"，
就必须用文本驱动的开放词汇检测，而不是固定 label 的封闭集检测器。

依赖
----
```bash
pip install -e ".[perception]"     # transformers + timm + safetensors
```
权重默认 `IDEA-Research/grounding-dino-tiny`（~700MB，8GB 显存足够）。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from roboground.perception.base import Detector
from roboground.perception.registry import register_detector
from roboground.types import Detection2D, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("perception.grounding_dino")


def format_prompts_for_grounding(prompts: Sequence[str]) -> str:
    """把 prompt 列表拼成 Grounding DINO 要求的格式。

    官方要求：小写、以 `. ` 分隔、以 `.` 结尾。
    例：`["cup", "table"]` → `"cup. table."`
    """
    cleaned = [str(p).strip().lower().rstrip(".") for p in prompts if str(p).strip()]
    if not cleaned:
        return ""
    return ". ".join(cleaned) + "."


@register_detector("grounding_dino", "grounding-dino", "gdino")
class GroundingDINODetector(Detector):
    """基于 transformers 的 Grounding DINO 开放词汇检测器（懒加载）。

    Parameters
    ----------
    canonicalize_labels : bool
        **实测必需**（默认 True）。Grounding DINO 返回的是**原始文本片段**，
        直接拿来当标签会有一堆问题：
        - `"trash"` 而 prompt 里写的是 `"trash can"`；
        - 甚至出现跨 prompt 的合并片段，例如 `"table bookshelf"`；
        - 大小写/单复数不一致。

        这会让下游的地图标签不统一、词法匹配失效。所以这里把检出标签
        **映射回最匹配的 prompt**（我们本来就是按 prompt 检的，
        映射回去语义上最合理），并丢弃与任何 prompt 都不匹配的"幻觉类别"。

    min_label_match : float
        标签映射到 prompt 的最低相似度；低于它视为幻觉类别丢弃。
    """

    name = "grounding_dino"
    supports_open_vocabulary = True

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        *,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        max_detections: int = 64,
        device: Optional[str] = None,
        prompts: Optional[Sequence[str]] = None,
        canonicalize_labels: bool = True,
        min_label_match: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.max_detections = int(max_detections)
        self.prompts = list(prompts or [])
        self.canonicalize_labels = bool(canonicalize_labels)
        self.min_label_match = float(min_label_match)
        self._device = device
        self._processor = None
        self._model = None
        self._matcher = None
        self.stats: dict = {}

    # ---------------- 懒加载 ----------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GroundingDINODetector 需要 transformers 与 torch："
                'pip install -e ".[perception]"'
            ) from exc

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"加载 Grounding DINO：{self.model_id}（device={self._device}）")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self._model = self._model.to(self._device).eval()

    def warmup(self) -> None:
        """预加载权重**并跑一次空推理**（去掉首次调用的 CUDA 初始化开销）。

        不预热的话，第一次 `detect` 会包含 CUDA context 建立 + cuDNN autotune，
        实测 480×640 图上需要 **1.4 秒**，而预热后只要 **100~200 毫秒** ——
        相差近 10 倍。做延迟基准时必须先 warmup，否则数字没有意义。
        """
        self._ensure_loaded()
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            from PIL import Image  # noqa: PLC0415

            inputs = self._processor(
                images=Image.fromarray(dummy), text="object.", return_tensors="pt"
            )
            inputs = {k: (v.to(self._device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            with torch.no_grad():
                self._model(**inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:  # 预热失败不该阻断流程
            logger.debug(f"Grounding DINO 预热失败（可忽略）：{exc}")

    # ---------------- 标签规范化 ----------------
    def _get_matcher(self):
        if self._matcher is None:
            from roboground.mapping.query import LexicalMatcher  # noqa: PLC0415

            self._matcher = LexicalMatcher()
        return self._matcher

    def canonicalize(self, raw_label: str, prompts: Sequence[str]) -> Optional[str]:
        """把模型吐出的原始文本片段映射回最匹配的 prompt。

        Returns
        -------
        str or None
            匹配到就返回 prompt 原文；与所有 prompt 都不匹配时返回 None
            （视为幻觉类别，调用方应丢弃）。
        """
        label = str(raw_label or "").strip().lower()
        if not label:
            return None
        if not prompts:
            return label

        # 1) 精确命中（最常见）
        for p in prompts:
            if label == str(p).strip().lower():
                return str(p)

        # 2) 词法匹配（处理 "trash" vs "trash can"、单复数、中英别名）
        matcher = self._get_matcher()
        scores = matcher.score(label, labels=[str(p) for p in prompts])
        best = int(np.argmax(scores))
        best_score = float(scores[best])
        if best_score >= self.min_label_match:
            return str(prompts[best])
        return None

    # ---------------- 推理 ----------------
    def detect(self, frame: RGBDFrame, prompts: Sequence[str]) -> List[Detection2D]:
        self._ensure_loaded()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        use_prompts = list(prompts) if prompts else self.prompts
        text = format_prompts_for_grounding(use_prompts)
        if not text:
            return []

        image = Image.fromarray(np.asarray(frame.color, dtype=np.uint8))

        inputs = self._processor(images=image, text=text, return_tensors="pt")
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = results.get("boxes")
        scores = results.get("scores")
        # ⚠️ 前向兼容：transformers ≥4.51 起 `labels` 会返回**整数 id**，
        # 字符串名要取 `text_labels`。优先用后者，没有才退回前者。
        labels = results.get("text_labels", results.get("labels"))

        detections: List[Detection2D] = []
        if boxes is None or len(boxes) == 0:
            self.stats = {"raw_detections": 0, "dropped_unknown_label": 0}
            return detections

        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        dropped = 0
        for i in range(min(len(boxes_np), self.max_detections * 2)):
            raw_label = str(labels[i]) if labels is not None else "object"

            if self.canonicalize_labels:
                label = self.canonicalize(raw_label, use_prompts)
                if label is None:
                    dropped += 1
                    logger.debug(f"丢弃幻觉类别：{raw_label!r}（不匹配任何 prompt）")
                    continue
            else:
                label = raw_label.strip().lower()

            detections.append(Detection2D(
                label=label,
                score=float(scores_np[i]),
                bbox=boxes_np[i].astype(np.float64),
                prompt=label,
            ))
            if len(detections) >= self.max_detections:
                break

        self.stats = {
            "raw_detections": int(len(boxes_np)),
            "dropped_unknown_label": int(dropped),
            "kept": len(detections),
        }
        if dropped:
            logger.debug(f"标签规范化丢弃 {dropped} 个不匹配 prompt 的检测")
        return detections
