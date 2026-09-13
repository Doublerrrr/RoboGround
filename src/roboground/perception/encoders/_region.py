"""区域裁剪的共享工具（CLIP / SigLIP / 未来任何图文编码器都用）。

为什么需要单独抽出来
-------------------
"把检测区域变成一个可以喂给图文模型的图像"这件事有两个**必须做对**的细节，
而且它们在实测中直接决定了特征质量：

1. **保持长宽比（方形填充）**
   图文模型（CLIP / SigLIP）的预处理会把输入 resize 成正方形（224×224）。
   如果直接把一个 3:1 的细长裁剪丢进去，目标会被**拉伸变形** ——
   椅子和书架在变形后可能长得差不多。
   解法：先把裁剪区域扩成正方形（优先在图像内扩，多拿一点上下文；
   超出边界时用图像均值色补齐）。

2. **留出上下文（context margin）**
   视觉语言模型是在**整图**上训练的，"紧贴目标边缘"的裁剪会丢掉
   上下文线索（椅子旁边的桌子腿、显示器的支架）。
   实测留 10~25% 的上下文比紧贴裁剪更稳。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def expand_bbox(
    bbox: Sequence[float],
    *,
    margin: float = 0.0,
) -> Tuple[float, float, float, float]:
    """按比例向四周外扩 bbox（margin 是相对当前边长比例）。"""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    mx, my = bw * float(margin), bh * float(margin)
    return x1 - mx, y1 - my, x2 + mx, y2 + my


def square_bbox(
    bbox: Sequence[float],
    *,
    image_size: Optional[Tuple[int, int]] = None,
    allow_expand: bool = True,
) -> Tuple[float, float, float, float]:
    """把 bbox 扩成正方形（以中心为锚），可选限制在图像范围内。

    Parameters
    ----------
    image_size
        `(width, height)`。给出且 `allow_expand=True` 时，尽量在图像内扩；
        扩不下就把中心挪回图像内（这样至少不引入人工像素）。
    allow_expand
        是否允许把框扩大到图像边界之外（由调用方决定怎么补）。
    """
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1, 1.0)
    half = side / 2.0

    nx1, ny1 = cx - half, cy - half
    nx2, ny2 = cx + half, cy + half

    if image_size is not None and not allow_expand:
        w, h = image_size
        # 把中心挪回图像内，保证正方形完整落在图像里
        side = min(side, w, h)
        half = side / 2.0
        cx = float(np.clip(cx, half, max(w - half, half)))
        cy = float(np.clip(cy, half, max(h - half, half)))
        nx1, ny1, nx2, ny2 = cx - half, cy - half, cx + half, cy + half

    return nx1, ny1, nx2, ny2


def build_region_views(
    color: np.ndarray,
    bbox: Sequence[float],
    *,
    mask: Optional[np.ndarray] = None,
    tta: str = "none",
    context: float = 0.15,
    square: bool = True,
    mask_pool: bool = False,
):
    """为一个检测区域生成**一组视图**（多视图 TTA 用）。

    Parameters
    ----------
    tta : {"none", "flip", "multicrop"}
        - `none`：单个方形裁剪
        - `flip`：方形裁剪 + 水平翻转（2 视图）
        - `multicrop`：多尺度 + 多上下文 + 掩码池化，各带翻转（最多 8 视图）

    Returns
    -------
    list[PIL.Image]
        该区域的视图列表；特征取平均。

    为什么多视图有效：单一裁剪的"尺度/上下文"是一次抽签 ——
    同一个书架裁紧一点像柜子、裁松一点像桌子。
    多视图平均把这次抽签的方差压下去，相当于对裁剪策略做了集成，
    而且**不需要训练**。
    """
    if tta == "none":
        return [region_to_square_crop(color, bbox, mask=mask, context=context,
                                      square=square, mask_pool=mask_pool)]

    if tta == "flip":
        base = region_to_square_crop(color, bbox, mask=mask, context=context,
                                     square=square, mask_pool=mask_pool)
        return [base, base.transpose(0)]        # PIL: 0 == FLIP_LEFT_RIGHT

    views = []
    specs = [
        dict(context=0.0, mask_pool=False),                # 紧裁剪
        dict(context=context, mask_pool=False),            # 带上下文
        dict(context=context, mask_pool=True),             # 掩码剔除背景
        dict(context=context * 2.0, mask_pool=False),      # 更大上下文
    ]
    for spec in specs:
        img = region_to_square_crop(color, bbox, mask=mask, square=square, **spec)
        views.append(img)
        views.append(img.transpose(0))
    return views


def region_to_square_crop(
    color: np.ndarray,
    bbox: Sequence[float],
    *,
    mask: Optional[np.ndarray] = None,
    context: float = 0.15,
    square: bool = True,
    mask_pool: bool = False,
    out_size: Optional[int] = None,
):
    """把检测区域裁成一个**方形、带上下文**的小图。

    Parameters
    ----------
    color : (H,W,3) uint8
    bbox : (4,) `(x1,y1,x2,y2)` 像素
    mask : (H,W) bool, optional
        `mask_pool=True` 时用它把背景像素涂成区域均值色。
    context
        额外上下文比例（相对 bbox 边长）。
    square
        是否扩成正方形（**强烈建议 True**，见模块 docstring）。
    mask_pool
        是否用掩码剔除背景（涂成均值色）。
    out_size
        输出边长（None 表示不 resize）。

    Returns
    -------
    PIL.Image
    """
    from PIL import Image  # noqa: PLC0415

    height, width = color.shape[:2]
    x1, y1, x2, y2 = expand_bbox(bbox, margin=context)

    if square:
        x1, y1, x2, y2 = square_bbox((x1, y1, x2, y2), image_size=(width, height),
                                     allow_expand=True)

    # ---- 在图像内的部分直接裁 ----
    ix1 = int(np.clip(np.floor(x1), 0, width))
    ix2 = int(np.clip(np.ceil(x2), 0, width))
    iy1 = int(np.clip(np.floor(y1), 0, height))
    iy2 = int(np.clip(np.ceil(y2), 0, height))

    side = max(int(round(max(x2 - x1, y2 - y1))), 1)

    if ix2 <= ix1 or iy2 <= iy1:
        # 退化情形：给一个中性色方块，避免调用方拿到 None 后到处判空
        return Image.new("RGB", (side, side), (128, 128, 128))

    patch = np.asarray(color[iy1:iy2, ix1:ix2], dtype=np.uint8)

    # ---- mask_pool：把掩码外的像素涂成区域均值色 ----
    if mask_pool and mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == (height, width):
            sub = m[iy1:iy2, ix1:ix2]
            if sub.any():
                arr = patch.astype(np.float32).copy()
                mean_color = arr[sub].mean(axis=0)
                arr[~sub] = mean_color
                patch = arr.astype(np.uint8)

    img = Image.fromarray(patch)

    # ---- 超出图像边界 → 用图像整体均值色补齐成正方形 ----
    if square and (img.width != img.height):
        bg = tuple(int(v) for v in np.asarray(color, dtype=np.float32).reshape(-1, 3).mean(axis=0))
        canvas = Image.new("RGB", (side, side), bg)
        canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        img = canvas

    if out_size is not None and img.size != (int(out_size), int(out_size)):
        img = img.resize((int(out_size), int(out_size)), Image.BICUBIC)

    return img
