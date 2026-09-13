"""封闭集检测：YOLO（可选后端，用于对比"封闭集 vs 开放词汇"）。

存在意义
--------
1. 简历里的对比实验需要一个**封闭集基线**（我在绿联用 YOLO，
   RoboGround 用它来量化"开放词汇带来了什么、代价是什么"）；
2. 端侧部署时 YOLO 更轻，是"快慢分级"里快那一档的候选。

依赖：`ultralytics`（不在默认依赖里，可选装）。权重会自动下载。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from roboground.perception.base import Detector
from roboground.perception.registry import register_detector
from roboground.types import Detection2D, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("perception.yolo")


@register_detector("yolo", "yolov8", "yolo11")
class YOLODetector(Detector):
    """YOLO 封闭集检测器（懒加载，依赖 ultralytics）。

    ⚠️ 它是**封闭集**的：只能检测 COCO 的 80 类。当 prompt 里的类别不在
    这些类别中时，无法检出 —— 这正是它与 `GroundingDINODetector` 的本质区别。
    `name_map` 允许把 COCO 类名映射到项目统一词表（例如 'cup' 保持，
    'dining table' → 'table'）。
    """

    name = "yolo"
    supports_open_vocabulary = False

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        *,
        conf: float = 0.30,
        iou: float = 0.50,
        max_detections: int = 64,
        prompts: Optional[Sequence[str]] = None,
        name_map: Optional[dict] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.weights = weights
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.prompts = list(prompts or [])
        self.name_map = dict(name_map or {})
        self._device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "YOLODetector 需要 ultralytics：pip install ultralytics"
            ) from exc
        logger.info(f"加载 YOLO 权重：{self.weights}")
        self._model = YOLO(self.weights)

    def warmup(self) -> None:
        self._ensure_loaded()

    def detect(self, frame: RGBDFrame, prompts: Sequence[str]) -> List[Detection2D]:
        self._ensure_loaded()
        use_prompts = list(prompts) if prompts else self.prompts

        results = self._model.predict(
            source=np.asarray(frame.color, dtype=np.uint8),
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self._device,
        )
        if not results:
            return []

        res = results[0]
        names = getattr(res, "names", {}) or {}
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int)

        # 封闭集 + prompt 过滤：只保留被 prompt 覆盖的类别
        from roboground.mapping.query import LexicalMatcher  # noqa: PLC0415

        matcher = LexicalMatcher()
        detections: List[Detection2D] = []
        for i in range(len(xyxy)):
            raw = str(names.get(int(clss[i]), str(clss[i])))
            label = self.name_map.get(raw, raw)
            if use_prompts:
                scores = matcher.score(use_prompts[0] if len(use_prompts) == 1 else label,
                                       labels=[label])
                # 用所有 prompt 一起判断
                best = max(
                    float(matcher.score(p, labels=[label])[0]) for p in use_prompts
                )
                if best <= 0.5:
                    continue
            detections.append(Detection2D(
                label=label,
                score=float(confs[i]),
                bbox=xyxy[i].astype(np.float64),
            ))
            if len(detections) >= self.max_detections:
                break
        return detections
