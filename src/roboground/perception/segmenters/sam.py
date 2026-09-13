"""SAM 分割器：检测框当 prompt → 精细掩码。

对应简历里那句"如果重做，我会用 SAM 的 mask 而不是 bbox 来反投影"——
本模块就是那句话的实现。它通常能把门窗/桌面这类**细薄或有内部结构**的
物体边界切准，从而显著提升反投影到 3D 的点云质量（少带背景点）。

依赖
----
```bash
pip install -e ".[perception]"
```
权重默认 `facebook/sam-vit-base`（~375MB）。8GB 显存下建议：
- 每帧最多送 `max_boxes` 个框（SAM 的 prompt encoder 是逐个框算的）；
- 图像可以先缩放到 `max_side` 再分割，最后把掩码放大回原尺寸。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from roboground.perception.base import Segmenter
from roboground.perception.registry import register_segmenter
from roboground.types import Detection2D, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("perception.sam")


@register_segmenter("sam", "sam2", "segment_anything")
class SAMSegmenter(Segmenter):
    """基于 transformers 的 SAM 分割器（懒加载）。"""

    name = "sam"

    def __init__(
        self,
        model_id: str = "facebook/sam-vit-base",
        *,
        multimask_output: bool = False,
        max_boxes: int = 32,
        max_side: int = 1024,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id
        self.multimask_output = bool(multimask_output)
        self.max_boxes = int(max_boxes)
        self.max_side = int(max_side)
        self._device = device
        self._processor = None
        self._model = None

    # ---------------- 懒加载 ----------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import SamModel, SamProcessor  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SAMSegmenter 需要 transformers 与 torch："
                'pip install -e ".[perception]"'
            ) from exc

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"加载 SAM：{self.model_id}（device={self._device}）")
        self._processor = SamProcessor.from_pretrained(self.model_id)
        self._model = SamModel.from_pretrained(self.model_id)
        self._model = self._model.to(self._device).eval()

    def warmup(self) -> None:
        self._ensure_loaded()

    # ---------------- 推理 ----------------
    def segment(
        self, frame: RGBDFrame, detections: Sequence[Detection2D]
    ) -> List[Detection2D]:
        if not detections:
            return list(detections)
        self._ensure_loaded()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        height, width = frame.shape
        image = Image.fromarray(np.asarray(frame.color, dtype=np.uint8))

        # 控制输入分辨率：SAM 的 image encoder 是固定 1024，但过大原图会拖慢预处理
        scale = 1.0
        if max(height, width) > self.max_side:
            scale = self.max_side / float(max(height, width))
            new_size = (int(round(width * scale)), int(round(height * scale)))
            image = image.resize(new_size, Image.BILINEAR)

        subset = list(detections[: self.max_boxes])
        scaled_boxes = [
            (np.asarray(d.bbox, dtype=np.float64) * scale).tolist() for d in subset
        ]

        inputs = self._processor(
            images=image,
            input_boxes=[scaled_boxes],
            return_tensors="pt",
        )
        inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=self.multimask_output)

        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]                                  # (num_boxes, num_masks, H', W')
        iou_scores = outputs.iou_scores.detach().cpu().numpy()[0]   # (num_boxes, num_masks)

        out = list(detections)
        for i, det in enumerate(subset):
            if i >= masks.shape[0]:
                break
            m = masks[i]
            if m.ndim == 3:
                best = int(np.argmax(iou_scores[i])) if iou_scores.shape[1] > 1 else 0
                m = m[best]
            mask_small = np.asarray(m, dtype=bool)

            # 放大回原图尺寸
            if mask_small.shape != (height, width):
                mask_img = Image.fromarray((mask_small * 255).astype(np.uint8))
                mask_small = np.asarray(
                    mask_img.resize((width, height), Image.NEAREST)
                ) > 127

            # 极端情况下 SAM 可能返回空掩码 → 保留 bbox 掩码作为兜底
            if mask_small.sum() == 0:
                continue
            det.mask = mask_small

        return out
