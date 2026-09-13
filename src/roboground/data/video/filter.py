"""去重与质量过滤：把「抽出来的帧」变成「能进训练的数据」。

为什么去重要放在抽帧**之后**而不是之前
====================================
抽帧只保证"选得分散"，不保证"选得不重复"：
- 镜头检测漏掉一个边界 → 两个镜头被当成一个，抽帧仍会抽到重复画面；
- 短片里同一物体在多个镜头反复出现 → 语义重复但像素不同，镜头检测管不了。

所以工业管线的顺序是 **抽帧 → 去重 → 质量过滤**，且去重要分两层：

| 层 | 手段 | 抓什么 | 为什么不能省 |
|---|---|---|---|
| **感知哈希** | dHash / pHash | 近重复**像素**（同一画面重复、压缩伪影） | 快（~0.05 ms/帧），能过海量数据 |
| **嵌入聚类** | 归一化嵌入 + 贪心去重 | **语义**重复（同物体不同角度、不同光照） | 哈希抓不到"看起来不同其实是同一个东西" |

只做哈希会留下大量"语义重复"，训练时等于给同一概念加权；
只做嵌入则算力吃不消（每帧都要过模型）。**两级级联**才是工程解：
哈希先砍掉 80% 明显重复，剩下的才进嵌入。

质量过滤：四道闸门（与项目 `data/auto_label.py` 的四道闸门同一套思路）
===================================================================
1. **模糊**（拉普拉斯方差）—— 糊帧喂给 VLM 会得到幻觉 caption；
2. **曝光**（过曝/欠曝像素占比）—— 全黑全白帧没有信息；
3. **分辨率** —— 低于最小边长直接丢（VLM 会 resize，但太小就没细节）；
4. **信息量**（帧内标准差 + 帧间变化）—— 纯色/静止帧没有可描述内容。

每条都返回**分数而不是布尔**，阈值单独给 —— 这样阈值调参有依据，
也能在消融里看到"放宽某一闸门"的代价。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("data.video.filter")


# =============================================================================
# 感知哈希
# =============================================================================
def dhash(frame: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """差值哈希（dHash）：转灰度 → resize → 比较水平相邻像素。

    比 aHash（与均值比）更稳：aHash 在亮度整体偏移时会翻转大量 bit，
    而 dHash 看的是**梯度方向**，对曝光变化不敏感 ——
    这正是我们要的（同一画面不同曝光不该被当成两帧）。
    """
    import cv2  # noqa: PLC0415

    img = np.asarray(frame)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).reshape(-1)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


@dataclass
class DedupConfig:
    """去重配置。"""

    hash_size: int = 8
    #: 哈希汉明距离阈值（64 bit 里允许差几位）。≤6 实践经验值：
    #: 太小（2）抓不到压缩伪影，太大（>10）会把不同帧误判为重复。
    hash_threshold: int = 6
    #: 第二级：嵌入余弦相似度阈值（越高越严）
    embedding_threshold: float = 0.97
    #: 是否启用嵌入级（没有嵌入就自动跳过）
    use_embedding: bool = True


def dedup_frames(
    frames: Sequence[np.ndarray],
    indices: Sequence[int],
    *,
    embeddings: Optional[np.ndarray] = None,
    cfg: Optional[DedupConfig] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """对**已抽中的帧**做两级去重，返回保留下来的帧下标。

    Returns
    -------
    (kept_indices, stats)
        `stats` 记录两级各砍掉多少、哈希阈值是否过于激进等信息。
    """
    cfg = cfg or DedupConfig()
    kept: List[int] = []
    hashes: List[np.ndarray] = []
    dropped_hash: List[int] = []
    dropped_emb: List[int] = []

    E = None
    if cfg.use_embedding and embeddings is not None:
        E = np.asarray(embeddings, dtype=np.float64)
        n = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.clip(n, 1e-12, None)

    for i in indices:
        if i < 0 or i >= len(frames):
            continue
        h = dhash(frames[i], cfg.hash_size)
        # ---- 第一级：感知哈希 ----
        if any(hamming(h, hk) <= cfg.hash_threshold for hk in hashes):
            dropped_hash.append(i)
            continue
        # ---- 第二级：嵌入 ----
        if E is not None and i < E.shape[0] and kept:
            sims = E[kept] @ E[i]
            if float(sims.max()) >= cfg.embedding_threshold:
                dropped_emb.append(i)
                continue
        kept.append(i)
        hashes.append(h)

    stats = {
        "n_in": len(indices),
        "n_kept": len(kept),
        "dropped_by_hash": len(dropped_hash),
        "dropped_by_embedding": len(dropped_emb),
        "keep_rate": len(kept) / max(len(indices), 1),
        "hash_drop_rate": len(dropped_hash) / max(len(indices), 1),
    }
    logger.debug(
        f"去重：{stats['n_in']} → {stats['n_kept']} 帧"
        f"（哈希砍 {stats['dropped_by_hash']}、嵌入砍 {stats['dropped_by_embedding']}）"
    )
    return kept, stats


# =============================================================================
# 质量过滤
# =============================================================================
@dataclass
class QualityConfig:
    """质量闸门配置。每条闸门都是"分数 < 阈值 → 丢"。"""

    #: 拉普拉斯方差下限（越小越糊）。经验值：清晰自然图 >100，明显糊 <30
    min_sharpness: float = 30.0
    #: 过曝/欠曝像素占比上限
    max_overexposed: float = 0.35
    max_underexposed: float = 0.35
    #: 最小边长（像素）
    min_side: int = 128
    #: 帧内标准差下限（纯色帧没有可描述内容）
    min_contrast: float = 8.0
    #: 是否启用各闸门（消融时逐条关掉，看代价）
    enable: Dict[str, bool] = field(default_factory=lambda: {
        "sharpness": True, "exposure": True, "resolution": True, "contrast": True,
    })


def quality_scores(frame: np.ndarray) -> Dict[str, float]:
    """算一帧的全部质量分数（**只算分，不判死活**）。"""
    import cv2  # noqa: PLC0415

    img = np.asarray(frame)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    g = gray.astype(np.float32)
    sharpness = float(cv2.Laplacian(g, cv2.CV_32F).var())
    total = g.size
    return {
        "sharpness": sharpness,
        "overexposed": float((g >= 250).sum()) / total,
        "underexposed": float((g <= 5).sum()) / total,
        "min_side": float(min(gray.shape[:2])),
        "contrast": float(g.std()),
    }


def passes_quality(scores: Dict[str, float], cfg: QualityConfig) -> Tuple[bool, str]:
    """按配置判定；返回 (是否通过, 未通过的原因)。"""
    en = cfg.enable
    if en.get("resolution", True) and scores["min_side"] < cfg.min_side:
        return False, "resolution"
    if en.get("sharpness", True) and scores["sharpness"] < cfg.min_sharpness:
        return False, "blur"
    if en.get("contrast", True) and scores["contrast"] < cfg.min_contrast:
        return False, "low_contrast"
    if en.get("exposure", True):
        if scores["overexposed"] > cfg.max_overexposed:
            return False, "overexposed"
        if scores["underexposed"] > cfg.max_underexposed:
            return False, "underexposed"
    return True, ""


def filter_by_quality(
    frames: Sequence[np.ndarray],
    indices: Sequence[int],
    *,
    cfg: Optional[QualityConfig] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """四道质量闸门；返回保留下来的帧下标与逐闸门淘汰统计。

    **统计比结果更重要**：能看出是哪道闸门在大量淘汰数据。
    如果 80% 的帧都死在"模糊"，那要么阈值太严，要么视频源本身有问题 ——
    两种情况的处理方式完全不同。
    """
    cfg = cfg or QualityConfig()
    kept: List[int] = []
    reasons: Dict[str, int] = {}
    all_scores: List[Dict[str, float]] = []

    for i in indices:
        if i < 0 or i >= len(frames):
            continue
        sc = quality_scores(frames[i])
        all_scores.append(sc)
        ok, why = passes_quality(sc, cfg)
        if ok:
            kept.append(i)
        else:
            reasons[why] = reasons.get(why, 0) + 1

    stats: Dict[str, Any] = {
        "n_in": len(indices),
        "n_kept": len(kept),
        "keep_rate": len(kept) / max(len(indices), 1),
        "drop_reasons": reasons,
    }
    if all_scores:
        for key in ("sharpness", "overexposed", "underexposed", "contrast"):
            vals = [s[key] for s in all_scores]
            stats[f"median_{key}"] = float(np.median(vals))
    logger.debug(f"质量过滤：{stats['n_in']} → {stats['n_kept']} 帧，"
                 f"淘汰原因 {reasons or '无'}")
    return kept, stats
