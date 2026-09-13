"""字幕工程：把"抓来的字幕"变成"能进训练的字幕"。

为什么字幕要单独做一层，而不是"顺手 filter 一下"
================================================
多模态数据的质量瓶颈**几乎从来不在图像侧，而在文本侧**。
图像最差也就是模糊/重复，肉眼可判；而文本的问题更隐蔽：

- **模板化污染** —— MSR-VTT 里大量字幕是同一个句式
  （"a man is talking about ..."），它们让模型学到的是句式而不是视觉；
- **重复与近重复** —— 同一视频的 20 条字幕里往往有 3~5 条语义重复
  （"a band performing" / "a band plays a song"），等于给同一概念加权；
- **退化文本** —— 空串、单字、纯符号、URL、"..."；
- **与画面无关** —— 这是最难发现的：字幕语法完全正常，
  但它描述的物体**不在这一帧里**（抽帧抽到了别处）。

前四类可以用**规则 + 去重**解决，第五类必须**跨模态打分**（图文对齐）才能发现。
本模块把前四类做成"必跑"，第五类做成"可插拔"（需要编码器，离线时明确跳过
而不是假装做了）。

三级去重的设计依据
==================
| 级别 | 手段 | 复杂度 | 抓什么 |
|---|---|---|---|
| L1 | 归一化后**精确匹配** | O(n) | 完全重复（镜像数据拼接、重复采集） |
| L2 | **词集签名**（排序去重后 join） | O(n) | 词序不同但词集相同（"a dog runs" / "runs a dog"） |
| L3 | **MinHash + LSH 分带** | ≈O(n) | 近重复（改几个词、加个形容词） |

为什么 L3 用 MinHash+LSH 而不是两两算 Jaccard：
60k 条字幕两两比是 **18 亿次**比较，纯 Python 要跑几十分钟；
MinHash 把每条字幕压成 64 维签名，LSH 分带后只在**桶内**比较，
实测降到万级比较，且召回由带数控制（带越多召回越高、代价是更多假阳性）。

⚠️ 一个必须做的检查：**跨视频重复**。
MSR-VTT 的字幕里有相当比例的通用句式在**多个视频**上重复出现
（不是同一视频的 20 条之间）。这类"跨视频高频字幕"是最有害的模板污染，
因为它们在训练里被反复加权。本模块把它单独统计出来（`cross_video_dups`），
而不是和"同视频内重复"混在一起。
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.data.corpus.schema import CaptionRecord
from roboground.utils.logging import get_logger

logger = get_logger("data.corpus.caption")


# =============================================================================
# 清洗
# =============================================================================
@dataclass
class CaptionCleanConfig:
    """字幕清洗配置。阈值全部可调，便于做"放宽某条闸门"的代价消融。"""

    min_words: int = 3
    max_words: int = 60
    #: 非 ASCII 字符占比上限（MSR-VTT 是英文语料；放宽可支持中文数据集）
    max_non_ascii: float = 0.15
    #: 最少字母字符数（挡住 "..." / "1 2 3" / "♪♪♪"）
    min_alpha_chars: int = 6
    #: 重复词占比上限（挡住 "very very very very"）
    max_repeat_word_ratio: float = 0.5
    #: 连续重复标点折叠阈值
    max_punct_run: int = 3
    #: 是否丢弃含 URL 的文本
    drop_urls: bool = True
    #: 各闸门开关（消融用）
    enable: Dict[str, bool] = field(default_factory=lambda: {
        "length": True, "script": True, "alpha": True,
        "repeat": True, "url": True,
    })


_URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PUNCT_RUN_RE = re.compile(r"([!?.,;:])\1{%d,}" % 2)
_WORD_RE = re.compile(r"[A-Za-z']+")


def normalize_text(text: str) -> str:
    """归一化：解 HTML 实体 → 去标签 → Unicode NFKC → 折叠空白与标点。

    为什么要 NFKC：数据集里混着全角/半角、带连字符的变体
    （`café` vs `cafe\u0301`），不归一化的话"看起来一样"的两条文本
    会在精确去重里被判为不同，于是模板污染逃过 L1。
    """
    t = html.unescape(str(text))
    t = _HTML_TAG_RE.sub(" ", t)
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u200b", " ").replace("\ufeff", " ")
    t = _PUNCT_RUN_RE.sub(r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_caption(text: str, cfg: Optional[CaptionCleanConfig] = None) -> Tuple[str, str]:
    """清洗单条字幕。返回 `(清洗后文本, 丢弃原因)`。

    丢弃原因用**字符串而不是布尔** —— 因为"哪一类问题杀掉了最多数据"
    本身就是这份数据集的体检结论（数据卡里要写）。
    空字符串表示保留。
    """
    cfg = cfg or CaptionCleanConfig()
    t = normalize_text(text)
    en = cfg.enable

    if en.get("url", True) and cfg.drop_urls and _URL_RE.search(t):
        return t, "url"

    words = t.split()
    if en.get("length", True):
        if len(words) < cfg.min_words:
            return t, "too_short"
        if len(words) > cfg.max_words:
            return t, "too_long"

    if en.get("script", True):
        if not t:
            return t, "empty"
        non_ascii = sum(1 for ch in t if ord(ch) > 127) / len(t)
        if non_ascii > cfg.max_non_ascii:
            return t, "non_ascii"

    if en.get("alpha", True):
        if len(_WORD_RE.findall(t)) == 0 or sum(len(w) for w in _WORD_RE.findall(t)) < cfg.min_alpha_chars:
            return t, "no_alpha"

    if en.get("repeat", True):
        low = [w.lower() for w in _WORD_RE.findall(t)]
        if low:
            ratio = 1.0 - len(set(low)) / len(low)
            if ratio > cfg.max_repeat_word_ratio:
                return t, "degenerate_repeat"

    return t, ""


def clean_captions(
    records: Sequence[CaptionRecord],
    cfg: Optional[CaptionCleanConfig] = None,
) -> Dict[str, Any]:
    """就地清洗一批字幕，回填 `cleaned` / `tokens` / `drop_reason`，返回统计。

    **就地**是刻意的：数据工程里"同一条记录逐步补全"比"每步返回新列表"
    更贴近实际，也避免了 60k 条记录被复制十几遍。
    """
    cfg = cfg or CaptionCleanConfig()
    reasons: Counter[str] = Counter()
    kept = 0
    for c in records:
        cleaned, why = clean_caption(c.text, cfg)
        c.cleaned = cleaned
        c.tokens = len(cleaned.split())
        c.drop_reason = why
        if why:
            reasons[why] += 1
        else:
            kept += 1
    stats = {
        "n_in": len(records),
        "n_kept": kept,
        "keep_rate": kept / max(len(records), 1),
        "drop_reasons": dict(reasons.most_common()),
    }
    logger.debug(f"字幕清洗：{stats['n_in']} → {kept}，原因 {stats['drop_reasons'] or '无'}")
    return stats


# =============================================================================
# MinHash + LSH 近重复检测
# =============================================================================
_MERSENNE = (1 << 61) - 1


def _token_ids(texts: Sequence[str]) -> Tuple[List[List[int]], Dict[str, int]]:
    """把文本转成 token id 序列（建一次全局词表）。"""
    vocab: Dict[str, int] = {}
    out: List[List[int]] = []
    for t in texts:
        ids = []
        for w in _WORD_RE.findall(t.lower()):
            if w not in vocab:
                vocab[w] = len(vocab) + 1
            ids.append(vocab[w])
        out.append(ids)
    return out, vocab


def _stable_hash(token: str) -> int:
    """跨进程稳定的字符串哈希。

    ⚠️ 不能用内置 `hash()` —— Python 对 str 的哈希**默认带随机盐**
    （PYTHONHASHSEED），同一个词在不同进程里哈希值不同，
    于是 MinHash 签名不可复现、断点续跑会得到不同结果。
    """
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
                          "big")


def minhash_signature(token_ids: Sequence[int], n_perm: int = 64,
                      seed: int = 0) -> np.ndarray:
    """MinHash 签名：`n_perm` 个随机置换下的最小 token id。

    用 universal hashing `h(x) = (a*x + b) mod p` 模拟置换，
    系数由 `seed` 决定 —— 同 seed 必然得到同签名（可复现）。
    """
    sig = np.full(n_perm, np.iinfo(np.int64).max, dtype=np.int64)
    if not token_ids:
        return sig
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE, size=n_perm, dtype=np.int64)
    b = rng.integers(0, _MERSENNE, size=n_perm, dtype=np.int64)
    ids = np.asarray(token_ids, dtype=np.int64)
    # (n_perm, n_tokens) 的置换结果，逐行取最小
    hv = (a[:, None] * ids[None, :] + b[:, None]) % _MERSENNE
    return hv.min(axis=1)


def lsh_bands(sig: np.ndarray, band_size: int = 8) -> List[Tuple[int, int]]:
    """把签名切成带，返回每带的 `(带序号, 桶哈希)`。"""
    out = []
    n = sig.shape[0]
    for s in range(0, n - band_size + 1, band_size):
        chunk = sig[s:s + band_size].tobytes()
        out.append((s // band_size, _stable_hash(chunk.decode("latin-1"))))
    return out


@dataclass
class DedupConfig:
    """字幕去重配置。"""

    #: 是否做 L3 近重复（MinHash+LSH）
    use_minhash: bool = True
    n_perm: int = 64
    band_size: int = 8
    #: 判为近重复的 Jaccard 阈值
    jaccard_threshold: float = 0.80
    #: MinHash 的哈希族种子。**必须对所有文档取同一个值** ——
    #: MinHash 只有在"所有文档用同一组置换"时签名才可比。
    #: ⚠️ 曾经的 bug：实现里用 `seed=i`（文档下标）逐条生成签名，
    #: 于是每条文档落在不同的哈希族里，签名根本不可比，
    #: LSH 永远分不到同一个桶 → **L3 近重复去重静默失效**（命中恒为 0），
    #: 而统计里只看得到"近重复 0 条"，看起来像"数据很干净"。
    #: 回归锁：`tests/test_corpus.py::test_dedup_near_duplicate_detected`
    minhash_seed: int = 0
    #: 跨视频高频字幕的判定：同一文本出现在 ≥ 该数量的不同视频上
    cross_video_min_videos: int = 3


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def dedup_captions(
    records: Sequence[CaptionRecord],
    *,
    cfg: Optional[DedupConfig] = None,
    scope: str = "global",
) -> Dict[str, Any]:
    """三级去重。被判定为重复的记录**不删除**，而是打上 `drop_reason`。

    Parameters
    ----------
    scope
        `"global"`：跨视频去重（默认，适合数据集级清洗）；
        `"per_video"`：只在同一视频内去重（保留"同一句话在不同视频出现"的信息）。

    Returns
    -------
    stats
        各级砍掉多少、跨视频高频模板 Top-N。
    """
    cfg = cfg or DedupConfig()
    cands = [c for c in records if c.kept]

    # ---------- L1 精确（归一化后小写） ----------
    seen: Dict[str, str] = {}          # norm -> 首次出现的 video_id
    dropped_l1 = 0
    for c in cands:
        key = c.cleaned.lower()
        owner = seen.get(key)
        if owner is None:
            seen[key] = c.video_id
        elif scope == "global" or owner == c.video_id:
            c.drop_reason = "dup_exact"
            dropped_l1 += 1

    # ---------- L2 词集签名 ----------
    seen_sig: Dict[str, str] = {}
    dropped_l2 = 0
    for c in cands:
        if not c.kept:
            continue
        sig = " ".join(sorted(set(_WORD_RE.findall(c.cleaned.lower()))))
        owner = seen_sig.get(sig)
        if owner is None:
            seen_sig[sig] = c.video_id
        elif scope == "global" or owner == c.video_id:
            c.drop_reason = "dup_wordset"
            dropped_l2 += 1

    # ---------- L3 MinHash + LSH ----------
    dropped_l3 = 0
    if cfg.use_minhash:
        live = [c for c in cands if c.kept]
        if live:
            ids, _ = _token_ids([c.cleaned for c in live])
            # ⚠️ seed 必须是**同一个哈希族**（cfg.minhash_seed），不能按文档下标变化，
            # 否则签名不可比、LSH 永远分不到同一桶（见 DedupConfig 的说明）。
            sigs = (np.stack([minhash_signature(t, cfg.n_perm,
                                                seed=cfg.minhash_seed)
                              for t in ids], axis=0)
                    if ids else np.zeros((0, cfg.n_perm), dtype=np.int64))
            buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
            for i in range(len(live)):
                for b in lsh_bands(sigs[i], cfg.band_size):
                    buckets[b].append(i)
            # 桶内两两比 Jaccard，超阈值就"并到同一代表"，只保留代表。
            # 用 representative 映射而不是简单地"丢后出现的那条" ——
            # 后者在传递关系上会出错：A≈B、B≈C 但 A≉C 时，
            # 简单的两两丢法可能把 B 丢掉、C 留下，于是 A 和 C 都活着。
            representative: Dict[int, int] = {}

            def _rep(k: int) -> int:
                while representative.get(k, k) != k:
                    k = representative[k]
                return k

            for _, members in buckets.items():
                if len(members) < 2:
                    continue
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        i, j = members[x], members[y]
                        if not live[i].kept or not live[j].kept:
                            continue
                        # scope 必须贯穿到 L3：`per_video` 下"同一句话出现在
                        # 不同视频"不算重复 —— 那是**跨视频模板**，由专门的一级
                        # 统计（见下方 cross_video 段），两者混在一起会误判。
                        if scope == "per_video" and live[i].video_id != live[j].video_id:
                            continue
                        ri, rj = _rep(i), _rep(j)
                        if ri == rj:
                            continue
                        si = set(_WORD_RE.findall(live[i].cleaned.lower()))
                        sj = set(_WORD_RE.findall(live[j].cleaned.lower()))
                        if _jaccard(si, sj) >= cfg.jaccard_threshold:
                            # 保留下标小的那条作为代表（确定性）
                            keep, victim = (ri, rj) if ri < rj else (rj, ri)
                            live[victim].drop_reason = "dup_near"
                            representative[victim] = keep
                            dropped_l3 += 1

    # ---------- 跨视频高频模板 ----------
    per_video: Dict[str, set] = defaultdict(set)
    for c in records:
        if c.kept:
            per_video[c.cleaned.lower()].add(c.video_id)
    cross = {k: len(v) for k, v in per_video.items()
             if len(v) >= cfg.cross_video_min_videos}
    dropped_cross = 0
    for c in records:
        if c.kept and c.cleaned.lower() in cross:
            c.drop_reason = "cross_video_template"
            dropped_cross += 1

    stats = {
        "n_in": len(records),
        "n_kept": sum(1 for c in records if c.kept),
        "dropped_exact": dropped_l1,
        "dropped_wordset": dropped_l2,
        "dropped_near": dropped_l3,
        "dropped_cross_video": dropped_cross,
        "n_unique_texts": len(per_video),
        "n_cross_video_templates": len(cross),
        "top_cross_video": sorted(cross.items(), key=lambda kv: -kv[1])[:10],
    }
    logger.info(
        f"字幕去重：{stats['n_in']} → {stats['n_kept']}"
        f"（精确 {dropped_l1} / 词集 {dropped_l2} / 近重复 {dropped_l3} / "
        f"跨视频模板 {dropped_cross}）"
    )
    return stats


# =============================================================================
# 统计
# =============================================================================
def caption_stats(records: Sequence[CaptionRecord]) -> Dict[str, Any]:
    """字幕语料的分布统计（数据卡的核心内容）。

    除了常规的长度/词表统计，额外给两个**能反映数据健康度**的指标：

    - **type-token ratio (TTR)**：词表大小 / 总词数。
      多模态字幕天然重复度高，TTR 很低（远低于自然语料）——
      所以不要拿"TTR 低"当异常，要拿**它随数据量增长的曲线**判断：
      健康的数据集 TTR 应随规模缓慢下降（Heaps 定律），
      如果几乎不下降，说明新数据在重复同样的模板。
    - **每视频字幕数分布**：方差过大意味着有的视频被过度描述、
      有的严重欠描述，配比时要做上限截断。
    """
    kept = [c for c in records if c.kept]
    toks = [c.tokens for c in kept]
    words: Counter[str] = Counter()
    for c in kept:
        words.update(_WORD_RE.findall(c.cleaned.lower()))
    per_video: Counter[str] = Counter(c.video_id for c in kept)

    def _pct(vals: Sequence[float], q: float) -> float:
        return float(np.percentile(vals, q)) if vals else 0.0

    total_words = sum(toks)
    return {
        "n_total": len(records),
        "n_kept": len(kept),
        "keep_rate": len(kept) / max(len(records), 1),
        "total_tokens": total_words,
        "mean_tokens": (total_words / len(kept)) if kept else 0.0,
        "p10_tokens": _pct(toks, 10),
        "median_tokens": _pct(toks, 50),
        "p90_tokens": _pct(toks, 90),
        "max_tokens": max(toks) if toks else 0,
        "vocab_size": len(words),
        "type_token_ratio": (len(words) / total_words) if total_words else 0.0,
        "top_words": words.most_common(15),
        "n_videos_with_captions": len(per_video),
        "captions_per_video_mean": (len(kept) / len(per_video)) if per_video else 0.0,
        "captions_per_video_max": max(per_video.values()) if per_video else 0,
        "captions_per_video_min": min(per_video.values()) if per_video else 0,
    }


# =============================================================================
# 跨模态对齐打分（可插拔）
# =============================================================================
#: 打分回调：`(frames, texts) -> 每条文本的相似度`。
#: 之所以做成回调而不是直接依赖某个模型，是为了让离线环境能**明确跳过**
#: 而不是伪造一个假的分数。
AlignFn = Callable[[Sequence[Any], Sequence[str]], np.ndarray]


def score_alignment(
    pairs: Sequence[Tuple[Any, str]],
    align_fn: Optional[AlignFn],
    *,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """对 `(帧, 字幕)` 对做图文对齐打分，返回分布统计。

    ⚠️ **没有编码器时必须显式说明"没做"**，而不是返回空 dict 让人误以为通过了。
    对齐打分是多模态数据清洗里**唯一能抓"字幕与画面无关"**的手段，
    跳过它意味着这一类噪声完全没被处理 —— 这必须写进数据卡。
    """
    if align_fn is None:
        return {
            "performed": False,
            "reason": "未提供图文编码器（align_fn=None），跨模态对齐未执行；"
                      "『字幕与画面无关』这一类噪声**未被覆盖**",
            "n_pairs": len(pairs),
        }
    if not pairs:
        return {"performed": False, "reason": "没有可打分的样本", "n_pairs": 0}
    frames = [p[0] for p in pairs]
    texts = [p[1] for p in pairs]
    sims = np.asarray(align_fn(frames, texts), dtype=np.float64).reshape(-1)
    return {
        "performed": True,
        "n_pairs": len(pairs),
        "mean": float(sims.mean()),
        "median": float(np.median(sims)),
        "p10": float(np.percentile(sims, 10)),
        "p90": float(np.percentile(sims, 90)),
        "std": float(sims.std()),
        "below_threshold": int((sims < threshold).sum()),
        "threshold": threshold,
    }


__all__ = [
    "CaptionCleanConfig", "DedupConfig",
    "normalize_text", "clean_caption", "clean_captions",
    "minhash_signature", "lsh_bands", "dedup_captions",
    "caption_stats", "score_alignment", "AlignFn",
]
