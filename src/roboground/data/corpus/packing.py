"""数据配比与分片打包：决定"训练时到底看到什么"。

为什么"配比"是数据工程里最容易被忽略、又最影响结果的一步
========================================================
自然分布的数据集**不等于**好的训练集。三个典型问题：

1. **长尾失衡** —— MSR-VTT 里 `sports`(324) 和 `howto`(47) 差了 7 倍。
   直接按自然分布训，小类永远欠拟合。
2. **模板刷屏** —— 某类视频的字幕句式高度雷同（新闻播报），
   按样本数配比会让它主导梯度。
3. **长度偏置** —— 长视频贡献的帧数远多于短视频，
   如果按"帧"配比，等于让长视频主导。

所以配比要做三件事：**重新加权（temperature）→ 设上限（cap）→ 算 token 预算**。

温度采样的依据
==============
按 `p_i ∝ n_i^α` 重采样：

- `α = 1.0`：完全按自然分布（不重加权）；
- `α = 0.5`：**常用的折中**，拉平长尾但不把小类抬到与大类同权
  （同 NLP 里对语言/领域做 temperature sampling 的做法）；
- `α = 0.0`：完全均匀，每个 key 等量 —— 会让极小的类被过度重复，
  反而过拟合，所以一般不用。

⚠️ **温度采样之后必须再看一次实际分布**。因为采样是**有放回**的
（小类被重复采），只看目标分布会漏掉"某个类被重复了多少次"这件事 ——
重复次数才是过拟合的真正来源，所以统计里必须同时给出
`target_ratio` 和 `reuse_factor`（被重复采样的倍数）。

分片打包
========
产出 JSONL 分片 + `manifest.json`。选 JSONL 而不是 tar/WebDataset 的理由：
- **可读、可 diff、可断点** —— 数据工程里"能一眼看出这条数据长什么样"的价值
  高于几个百分点的读取效率；
- 真要转 WebDataset 只是 `tar` 一下的事，但反过来（从 tar 里查一条）很痛苦。
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from roboground.data.corpus.schema import VideoRecord
from roboground.utils.logging import get_logger

logger = get_logger("data.corpus.packing")


# =============================================================================
# 训练样本
# =============================================================================
@dataclass
class TrainingSample:
    """一条可进训练的多模态样本：**(视频, 字幕) 对**。

    为什么样本单位是"对"而不是"视频"：
    多模态训练实际消费的是 `(视觉输入, 文本)` 的配对。一条视频有 20 条字幕，
    就是 20 条样本（共用同一段视觉）。配比和 token 预算都必须在**对**这一级做，
    否则"每视频 20 条字幕"和"每视频 1 条字幕"的数据集会被错误地等权对待。
    """

    video_id: str
    source: str
    caption: str
    tokens: int
    duration: float
    frames_kept: int
    start: float = 0.0
    end: float = 0.0
    group: str = ""          # 配比分组键
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id, "source": self.source,
            "caption": self.caption, "tokens": self.tokens,
            "duration": round(self.duration, 3), "frames_kept": self.frames_kept,
            "start": round(self.start, 3), "end": round(self.end, 3),
            "group": self.group, **self.meta,
        }


def duration_bucket(seconds: float) -> str:
    """时长分桶（配比的一个常用维度：短视频和长视频的分布要均衡）。"""
    if seconds < 15:
        return "<15s"
    if seconds < 30:
        return "15-30s"
    if seconds < 60:
        return "30-60s"
    if seconds < 120:
        return "60-120s"
    return ">120s"


def iter_samples(records: Sequence[VideoRecord], *,
                 group_key: str = "category") -> Iterator[TrainingSample]:
    """把视频记录展开成训练样本流（只取成功视频 + 存活字幕）。"""
    for r in records:
        if not r.ok:
            continue
        for c in r.kept_captions:
            yield TrainingSample(
                video_id=r.video_id, source=r.source,
                caption=c.cleaned or c.text, tokens=c.tokens,
                duration=r.duration, frames_kept=r.n_frames_kept,
                start=c.start, end=c.end,
                group=_group_of(r, group_key),
                meta={"category": r.meta.get("category")},
            )


def _group_of(rec: VideoRecord, group_key: str) -> str:
    if group_key == "category":
        return str(rec.meta.get("category", "?"))
    if group_key == "duration":
        return duration_bucket(rec.duration)
    if group_key == "source":
        return rec.source
    return str(rec.meta.get(group_key, "?"))


# =============================================================================
# 配比
# =============================================================================
@dataclass
class MixtureConfig:
    """配比配置。"""

    #: 分组维度：category / duration / source
    group_key: str = "category"
    #: 温度采样指数 α。1.0=自然分布；0.5=拉平长尾（推荐）；0=完全均匀
    temperature: float = 0.5
    #: 单个分组在**目标分布**里的占比上限（防止某类刷屏）
    max_ratio_per_key: float = 0.25
    #: 单个分组被重复采样的倍数上限（防过拟合小类）
    max_reuse_factor: float = 3.0
    #: 采样总条数；None = 用全部可用样本
    total_budget: Optional[int] = None
    #: token 预算；超过则按比例截断（None = 不限）
    token_budget: Optional[int] = None
    seed: int = 0


def plan_mixture(
    samples: Sequence[TrainingSample],
    cfg: Optional[MixtureConfig] = None,
) -> Tuple[List[TrainingSample], Dict[str, Any]]:
    """按温度采样重新配比，返回 `(选中的样本, 统计)`。

    统计里同时给 **自然分布**、**目标分布**、**实际采样分布** 和
    **重复倍数** 四张表 —— 只给其中一张都会误导：
    - 只看自然分布 → 不知道长尾有多严重；
    - 只看目标分布 → 不知道小类被重复了多少次（过拟合风险）；
    - 只看实际分布 → 不知道它和自然分布差多少（有没有真的起作用）。
    """
    cfg = cfg or MixtureConfig()
    if not samples:
        return [], {"n_available": 0, "n_selected": 0}

    rng = np.random.default_rng(cfg.seed)
    by_group: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_group[s.group].append(i)

    counts = {g: len(v) for g, v in by_group.items()}
    total = len(samples)
    natural = {g: c / total for g, c in counts.items()}

    # ---------- 温度重加权 ----------
    weights = {g: (c / total) ** cfg.temperature for g, c in counts.items()}
    wsum = sum(weights.values())
    target = {g: w / wsum for g, w in weights.items()}

    # ---------- 占比上限 ----------
    capped = _cap_ratios(target, cfg.max_ratio_per_key)

    # ---------- 预算 ----------
    budget = cfg.total_budget or total
    # 每个 key 的目标条数（先按占比算，再受 max_reuse_factor 约束）
    target_n: Dict[str, int] = {}
    for g, r in capped.items():
        want = int(round(r * budget))
        cap = int(math.floor(counts[g] * cfg.max_reuse_factor))
        target_n[g] = max(1, min(want, max(cap, 1)))

    # 预算对齐（上面四舍五入后可能超出/不足）
    target_n = _rescale_to_budget(target_n, budget, counts, cfg.max_reuse_factor)

    # ---------- 采样 ----------
    selected: List[TrainingSample] = []
    for g, n in target_n.items():
        pool = by_group[g]
        if n <= len(pool):
            pick = rng.choice(len(pool), size=n, replace=False)
        else:
            pick = rng.integers(0, len(pool), size=n)   # 有放回：小类被重复
        selected.extend(samples[pool[j]] for j in np.asarray(pick).reshape(-1))

    # ---------- token 预算截断 ----------
    token_note = ""
    if cfg.token_budget is not None:
        selected.sort(key=lambda s: rng.random())
        acc, kept = 0, []
        for s in selected:
            if acc + s.tokens > cfg.token_budget:
                continue
            acc += s.tokens
            kept.append(s)
        token_note = f"token 预算 {cfg.token_budget:,} 截断：{len(selected)} → {len(kept)}"
        selected = kept

    actual = Counter(s.group for s in selected)
    n_sel = max(len(selected), 1)
    stats = {
        "group_key": cfg.group_key,
        "temperature": cfg.temperature,
        "n_available": total,
        "n_selected": len(selected),
        "n_groups": len(counts),
        "natural_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "natural_ratio": {g: round(v, 4) for g, v in sorted(natural.items(), key=lambda kv: -kv[1])},
        "target_ratio": {g: round(v, 4) for g, v in sorted(capped.items(), key=lambda kv: -kv[1])},
        "actual_ratio": {g: round(v / n_sel, 4) for g, v in sorted(actual.items(), key=lambda kv: -kv[1])},
        "actual_counts": dict(sorted(actual.items(), key=lambda kv: -kv[1])),
        "reuse_factor": {g: round(actual[g] / max(counts.get(g, 1), 1), 3)
                         for g in sorted(actual, key=lambda g: -actual[g])},
        "token_budget": cfg.token_budget,
        "token_note": token_note,
        "total_tokens": sum(s.tokens for s in selected),
    }
    logger.info(f"配比：可用 {total} → 选中 {len(selected)} 条，"
                f"分组 {len(counts)} 个，α={cfg.temperature}")
    return selected, stats


def _cap_ratios(target: Dict[str, float], cap: float) -> Dict[str, float]:
    """把超过 `cap` 的占比削掉，削掉的部分**按比例**分给其余组。

    为什么要"按比例分给其余"而不是直接归一化：直接归一化会把削掉的质量
    无差别地摊回去，可能让原本第二大的组再次超限。按比例分 + 迭代直到收敛，
    才是稳定的做法。
    """
    r = dict(target)
    for _ in range(20):
        over = {g: v for g, v in r.items() if v > cap + 1e-12}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for g in over:
            r[g] = cap
        rest = {g: v for g, v in r.items() if v < cap - 1e-12}
        rest_sum = sum(rest.values())
        if rest_sum <= 1e-12:
            break
        for g in rest:
            r[g] += excess * (rest[g] / rest_sum)
    s = sum(r.values())
    return {g: v / s for g, v in r.items()} if s > 0 else r


def _rescale_to_budget(
    target_n: Dict[str, int], budget: int,
    counts: Dict[str, int], max_reuse: float,
) -> Dict[str, int]:
    """把各组条数缩放到总预算（同时不突破重复倍数上限）。"""
    out = dict(target_n)
    for _ in range(50):
        total = sum(out.values())
        if total == budget or total == 0:
            break
        scale = budget / total
        out = {g: max(1, int(round(n * scale))) for g, n in out.items()}
        for g in out:
            out[g] = min(out[g], max(int(math.floor(counts[g] * max_reuse)), 1))
        if sum(out.values()) == total:
            break
    return out


# =============================================================================
# token 预算核算
# =============================================================================
def token_budget_report(samples: Sequence[TrainingSample]) -> Dict[str, Any]:
    """token 预算核算：训练成本直接由 token 数决定，所以必须先算清楚。"""
    if not samples:
        return {"n_samples": 0, "total_tokens": 0}
    toks = np.array([s.tokens for s in samples], dtype=np.float64)
    frames = np.array([s.frames_kept for s in samples], dtype=np.float64)
    return {
        "n_samples": len(samples),
        "total_tokens": int(toks.sum()),
        "mean_tokens": float(toks.mean()),
        "median_tokens": float(np.median(toks)),
        "p95_tokens": float(np.percentile(toks, 95)),
        #: 每样本平均视觉 token 的**代理量**（保留帧数）。
        #: 真实的视觉 token 数取决于 VLM 的 patch 化策略，
        #: 这里给的是与它单调相关的量，用于比较不同配比方案的相对成本。
        "mean_frames_kept": float(frames.mean()),
        "text_token_share_of_1k": float(toks.mean() / 1000.0),
    }


# =============================================================================
# 分片打包
# =============================================================================
@dataclass
class PackConfig:
    """打包配置。"""

    shard_size: int = 2000
    out_dir: Path = Path("data/packed")
    #: 是否附带一条可读的预览行（每片第一条，方便人工抽查）
    write_preview: bool = True


def pack_shards(
    samples: Sequence[TrainingSample],
    cfg: Optional[PackConfig] = None,
) -> Dict[str, Any]:
    """把样本打成 JSONL 分片 + manifest。"""
    cfg = cfg or PackConfig()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shards: List[Dict[str, Any]] = []
    for k in range(0, len(samples), cfg.shard_size):
        chunk = samples[k:k + cfg.shard_size]
        idx = k // cfg.shard_size
        p = out / f"shard-{idx:05d}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for s in chunk:
                fh.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        shards.append({
            "path": p.name, "n_samples": len(chunk),
            "n_tokens": sum(s.tokens for s in chunk),
            "bytes": p.stat().st_size,
        })

    manifest = {
        "n_samples": len(samples),
        "n_shards": len(shards),
        "shard_size": cfg.shard_size,
        "total_tokens": sum(s.tokens for s in samples),
        "shards": shards,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.write_preview and samples:
        preview = out / "preview.txt"
        with preview.open("w", encoding="utf-8") as fh:
            for s in samples[:20]:
                fh.write(f"[{s.group}] {s.video_id} | {s.tokens:3d}tok | {s.caption}\n")

    logger.info(f"打包完成：{len(samples)} 条 → {len(shards)} 片，输出 {out}")
    return manifest


__all__ = [
    "TrainingSample", "MixtureConfig", "PackConfig",
    "duration_bucket", "iter_samples", "plan_mixture",
    "token_budget_report", "pack_shards",
]
