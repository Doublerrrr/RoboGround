"""镜头检测的**真实数据金种子**：把"合成数据上的 F1"拿到真视频上验一遍。

为什么必须做这件事
==================
`data/video/shot.py` 的结论（自适应判据 F1 0.833、固定阈值 0.250）全部来自
**受控合成视频** —— 那里镜头边界是程序生成的，直方图差异非常干净。
真实视频完全不是这样：

- **渐变转场**（淡入淡出、叠化）没有突变，直方图差异是缓坡；
- **同场景内的剧烈运动**（快速摇镜、爆炸、闪光）直方图差异比真切还大 ——
  这是**假阳性**的主要来源；
- **同场景内的切镜**（正反打对话）差异很小 —— 这是**假阴性**的主要来源。

所以"合成上 F1 0.833"**不能直接外推**。必须在真实数据上重新测一遍，
而且真实数据**没有现成的边界标注** —— 得自己建金种子。

金种子怎么建（本模块的核验协议）
================================
不做"逐帧看完全片"（不现实），而是**两路夹逼**：

1. **漏检检查**：把全片等间隔抽 48 帧拼成一张总览条。
   一眼扫过去，如果两个相邻缩略图之间发生了场景变化，就是一次**疑似漏检**。
2. **误检检查**：对每个**检测到的**边界 b，渲染 `[b-half, b+half]` 的密集条
   （13 帧逐帧）。人工判断"这里到底有没有切"。

两路合起来覆盖了 P 和 R 两侧，且人工成本与"检测到的边界数"成正比
（通常每片 0~3 个），而不是与帧数成正比。

⚠️ 核验结果的**规模必须诚实标注**。本模块把 `n_videos_verified` 写进指标里，
不允许只报 F1 不报样本量 —— 否则就是把 12 条视频的结论伪装成普适结论。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.data.corpus.schema import VideoRecord
from roboground.utils.logging import get_logger

logger = get_logger("data.corpus.goldset")


# =============================================================================
# 配置
# =============================================================================
@dataclass
class GoldSetConfig:
    """金种子构建/核验配置。"""

    out_dir: Path = Path("runs/goldset")
    #: 总览条抽多少帧（漏检检查用）
    overview_frames: int = 48
    #: 边界条每侧多少帧（误检检查用）
    boundary_half: int = 6
    #: 缩略图宽度（像素）
    thumb_width: int = 200
    #: 每行放几张
    cols: int = 8
    #: 判为"命中"的时间容差（帧）
    tolerance: int = 2


# =============================================================================
# 图像拼装
# =============================================================================
def _thumb(frame: np.ndarray, width: int) -> np.ndarray:
    import cv2  # noqa: PLC0415

    h, w = frame.shape[:2]
    scale = width / max(w, 1)
    return cv2.resize(frame, (width, max(int(h * scale), 1)),
                      interpolation=cv2.INTER_AREA)


def _label(img: np.ndarray, text: str, *, height: int = 18) -> np.ndarray:
    """在缩略图底部烧上文字（用 PIL，避免 cv2 的中文/字体问题）。"""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    canvas = Image.new("RGB", (pil.width, pil.height + height), (255, 255, 255))
    canvas.paste(pil, (0, 0))
    ImageDraw.Draw(canvas).text((3, pil.height + 3), text, fill=(0, 0, 0))
    return np.asarray(canvas)


def make_strip(
    frames: Sequence[np.ndarray],
    indices: Sequence[int],
    *,
    cfg: Optional[GoldSetConfig] = None,
    title: str = "",
    mark: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """把若干帧按网格拼成一张带帧号标注的联络图（contact sheet）。

    Parameters
    ----------
    mark
        需要**高亮**的帧下标集合（例如检测到的边界）。高亮方式是给该格
        加一圈边框，方便人工直接定位"算法说这里切了"。
    """
    cfg = cfg or GoldSetConfig()
    import cv2  # noqa: PLC0415

    cells = []
    for i in indices:
        if i < 0 or i >= len(frames):
            continue
        t = _label(_thumb(frames[i], cfg.thumb_width), f"#{i}")
        if mark and i in set(mark):
            t = cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                   value=(220, 40, 40))
        cells.append(t)
    if not cells:
        raise ValueError("没有可渲染的帧")

    h = max(c.shape[0] for c in cells)
    w = max(c.shape[1] for c in cells)
    rows = int(np.ceil(len(cells) / cfg.cols))
    pad = 4
    sheet = np.full((rows * (h + pad) + 26, cfg.cols * (w + pad), 3), 255, np.uint8)
    for k, c in enumerate(cells):
        r, cc = divmod(k, cfg.cols)
        y, x = r * (h + pad), cc * (w + pad)
        sheet[y:y + c.shape[0], x:x + c.shape[1]] = c
    if title:
        sheet = _label(sheet, title, height=20)
    return sheet


def _save(img: np.ndarray, path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


# =============================================================================
# 渲染核验材料
# =============================================================================
def render_verification_material(
    rec: VideoRecord,
    boundaries: Sequence[int],
    *,
    cfg: Optional[GoldSetConfig] = None,
) -> Dict[str, Any]:
    """为一条视频渲染"总览条 + 逐边界密集条"，返回落盘路径。"""
    cfg = cfg or GoldSetConfig()
    from roboground.data.video.pipeline import open_video_source

    src = open_video_source(rec.path)
    try:
        frames = list(src)
    finally:
        src.close()
    n = len(frames)
    if n == 0:
        return {"video_id": rec.video_id, "error": "no_frames"}

    tag = rec.video_id if rec.video_id.startswith(rec.source) \
        else f"{rec.source}_{rec.video_id}"
    out: Dict[str, Any] = {"video_id": rec.video_id, "n_frames": n,
                           "detected": list(boundaries), "sheets": []}

    # ---- 总览条（漏检检查）----
    step = max(n // cfg.overview_frames, 1)
    idx = list(range(0, n, step))[: cfg.overview_frames]
    sheet = make_strip(frames, idx, cfg=cfg,
                       title=f"{tag}  overview  n={n} step={step}  "
                             f"detected={list(boundaries)}",
                       mark=boundaries)
    out["sheets"].append(str(_save(sheet, cfg.out_dir / f"{tag}__overview.png")))

    # ---- 逐边界密集条（误检检查）----
    for b in boundaries:
        half = cfg.boundary_half
        idx = list(range(max(0, b - half), min(n, b + half + 1)))
        if not idx:
            continue
        sheet = make_strip(frames, idx, cfg=cfg,
                           title=f"{tag}  boundary@{b}  window=[{idx[0]},{idx[-1]}]",
                           mark=[b])
        out["sheets"].append(
            str(_save(sheet, cfg.out_dir / f"{tag}__b{b:05d}.png")))
    return out


# =============================================================================
# 金种子读写
# =============================================================================
def save_goldset(gold: Dict[str, List[int]], path: Path) -> Path:
    """`{video_id: [真实边界帧下标, ...]}` → JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_goldset(path: Path) -> Dict[str, List[int]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# =============================================================================
# 评测
# =============================================================================
def match_boundaries(
    pred: Sequence[int], gold: Sequence[int], tolerance: int = 2,
) -> Tuple[int, int, int]:
    """一对一匹配（贪心最近优先）。返回 `(tp, fp, fn)`。

    ⚠️ **必须一对一**，不能"只要附近有真值就算命中"。
    否则一个真值边界旁边挤了 5 个假预测，会被算成 5 个 TP ——
    这正是当年 DETR 用匈牙利匹配替掉 NMS 的同一类问题：
    **多对一会让指标虚高**。
    """
    pred = sorted(int(p) for p in pred)
    gold = sorted(int(g) for g in gold)
    used = [False] * len(gold)
    tp = 0
    for p in pred:
        best, best_d = -1, tolerance + 1
        for j, g in enumerate(gold):
            if used[j]:
                continue
            d = abs(p - g)
            if d <= tolerance and d < best_d:
                best, best_d = j, d
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, len(pred) - tp, len(gold) - tp


def evaluate_goldset(
    predictions: Dict[str, Sequence[int]],
    gold: Dict[str, Sequence[int]],
    *,
    tolerance: int = 2,
) -> Dict[str, Any]:
    """在人工核验的金种子上算 P/R/F1。**样本量写进结果**。"""
    tp = fp = fn = 0
    per_video: List[Dict[str, Any]] = []
    for vid, g in gold.items():
        # 跳过元数据键（`_note` / `_meta` 之类）。标注文件带注释是常态，
        # 不跳过的话 `int("说明文字")` 会直接抛异常。
        if str(vid).startswith("_"):
            continue
        p = predictions.get(vid, [])
        a, b, c = match_boundaries(p, g, tolerance)
        tp, fp, fn = tp + a, fp + b, fn + c
        per_video.append({"video_id": vid, "n_pred": len(p), "n_gold": len(g),
                          "tp": a, "fp": b, "fn": c})
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        # 计数用**实际评测的视频数**，不是 `len(gold)` —— 后者会把
        # `_note` 之类的元数据键也算进去（实测多算 1 条）。
        "n_videos_verified": len(per_video),
        "tolerance": tolerance,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "per_video": per_video,
        "note": f"金种子仅覆盖 {len(per_video)} 条真实视频（人工核验），"
                f"结论方向可信，绝对值不应外推为全数据集指标。",
    }


__all__ = [
    "GoldSetConfig", "make_strip", "render_verification_material",
    "save_goldset", "load_goldset", "match_boundaries", "evaluate_goldset",
]
