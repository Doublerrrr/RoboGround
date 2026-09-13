"""规模跑批：把单视频管线跑到**数据集规模**，并给出可引用的吞吐与淘汰率。

单视频跑通 ≠ 数据管线可用
=========================
`data/video/process_video` 在一条视频上跑通，只能证明**逻辑正确**；
它证明不了下面任何一件事，而这些恰恰是数据工程真正的难点：

1. **会不会在某类视频上崩** —— 变分辨率、变帧率、10 秒短片 vs 2 分钟长片、
   竖屏、黑白、损坏文件。真实数据集里一定有这些，而且比例不低。
2. **淘汰率是否失控** —— 单视频"保留 12/24 帧"看起来合理，
   但如果在 3000 条视频上平均只保留 3 帧，这条管线就是**在丢数据**。
   必须按数据集统计**逐阶段漏斗**。
3. **吞吐够不够** —— 决定"下一次重跑要多久"。数据管线是要反复跑的，
   吞吐就是迭代速度。
4. **失败有没有被吞掉** —— 解码失败如果不记录，指标会被"成功的那部分"
   悄悄美化。所以本模块把**失败原因分类统计**作为一等公民。

本模块的三条设计原则
====================
- **失败必须留痕**：`VideoRecord.error` 非空即失败，且**不参与**任何均值统计
  （否则"平均保留帧数"会被 0 拉低，看起来像质量闸门太严）。
- **断点续跑**：2990 条视频要跑几十分钟，中途中断是常态。
  每处理完一条就 append 一行 JSONL，重启时按 `video_id` 跳过已完成项。
- **两遍结构**：视频侧是流式的（逐条处理，内存恒定），
  字幕侧是**全局**的（跨视频去重必须看到全量）。所以字幕清洗/去重
  放在所有视频处理完之后，而不是每条视频内部。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roboground.data.corpus import caption as cap_mod
from roboground.data.corpus.schema import CaptionRecord, VideoRecord
from roboground.data.corpus.sources import CorpusSource
from roboground.utils.logging import get_logger

logger = get_logger("data.corpus.runner")


# =============================================================================
# 配置
# =============================================================================
@dataclass
class CorpusRunConfig:
    """数据集规模跑批配置。"""

    #: 每条视频抽多少帧（预算，不是结果 —— 去重/质量会再砍）
    target_frames: int = 24
    #: 抽帧策略：uniform / thirds / keyframe / adaptive
    sampling_strategy: str = "adaptive"
    #: 特征计算时的降采样边长（大视频上算 HSV 直方图很贵）
    feature_max_side: int = 160
    #: 视频级去重/质量配置（沿用 data/video 的默认）
    dedup_hash_threshold: int = 6
    dedup_embedding_threshold: float = 0.97
    min_sharpness: float = 30.0
    min_contrast: float = 8.0
    #: 只处理前 N 条（冒烟）；None = 全量
    max_videos: Optional[int] = None
    #: 并发解码线程数。cv2 解码会释放 GIL，所以线程能拿到真实加速；
    #: 但显存/内存有限，8 线程是 8GB 机器上的稳妥上限。
    workers: int = 4
    #: 断点续跑文件（JSONL，一行一条视频）
    checkpoint: Optional[Path] = None
    #: 每处理多少条写一次 checkpoint
    checkpoint_every: int = 20
    #: 跳过没有视频文件的记录（只有元数据时设 True）
    skip_missing_video: bool = True
    seed: int = 0


# =============================================================================
# 结果
# =============================================================================
@dataclass
class CorpusRunResult:
    """跑批产物：记录 + 全套漏斗/吞吐/失败统计。"""

    records: List[VideoRecord]
    stage_seconds: Dict[str, float]
    funnel: Dict[str, Any]
    throughput: Dict[str, Any]
    failures: Dict[str, int]
    caption_clean: Dict[str, Any]
    caption_dedup: Dict[str, Any]
    caption_stats: Dict[str, Any]
    n_skipped_resume: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "funnel": self.funnel,
            "throughput": self.throughput,
            "failures": self.failures,
            "stage_seconds": self.stage_seconds,
            "caption_clean": self.caption_clean,
            "caption_dedup": self.caption_dedup,
            "caption_stats": self.caption_stats,
            "n_skipped_resume": self.n_skipped_resume,
            "videos": [r.to_dict() for r in self.records],
        }

    def summary_text(self) -> str:
        f, t = self.funnel, self.throughput
        lines = [
            f"  视频        : {f['n_videos']} 条（成功 {f['n_ok']} / 失败 {f['n_failed']}）",
            f"  总时长      : {f['total_duration_s'] / 60:.1f} 分钟",
            f"  解码帧数    : {f['n_frames_decoded']:,}",
            f"  抽帧        : {f['n_picked']:,} → 去重后 {f['n_after_dedup']:,}"
            f" → 质量后 {f['n_after_quality']:,}（保留率 {f['overall_keep_rate']:.1%}）",
            f"  镜头        : 共 {f['n_shots']:,}（均值 {f['shots_per_video_mean']:.1f}/视频）",
            f"  字幕        : {self.caption_clean['n_in']:,} → 清洗 {self.caption_clean['n_kept']:,}"
            f" → 去重 {self.caption_dedup['n_kept']:,}",
            f"  吞吐        : {t['videos_per_min']:.1f} 视频/分钟"
            f"、{t['frames_per_sec']:.0f} 帧/秒、{t['mb_per_sec']:.2f} MB/s",
            f"  阶段耗时    : { {k: round(v, 1) for k, v in list(self.stage_seconds.items())[:6]} }",
        ]
        if self.failures:
            lines.append(f"  失败归因    : {self.failures}")
        return "\n".join(lines)


# =============================================================================
# 单条视频
# =============================================================================
def _process_one(rec: VideoRecord, cfg: CorpusRunConfig) -> VideoRecord:
    """处理单条视频：跑 `data/video` 的完整管线，把统计回填到 `rec.stats`。

    所有异常都被**分类**捕获（而不是笼统 `except Exception: pass`）——
    因为"打不开文件"和"解到一半失败"是完全不同的问题：
    前者是数据下载不完整，后者是编解码器/文件损坏。
    """
    from roboground.data.video.filter import DedupConfig, QualityConfig
    from roboground.data.video.pipeline import PipelineConfig, process_video
    from roboground.data.video.sampling import SamplingConfig

    if rec.path is None:
        rec.error = "missing_file"
        return rec
    if not Path(rec.path).exists():
        rec.error = "missing_file"
        return rec

    pcfg = PipelineConfig(
        sampling=SamplingConfig(strategy=cfg.sampling_strategy,
                                target_frames=cfg.target_frames),
        dedup=DedupConfig(hash_threshold=cfg.dedup_hash_threshold,
                          embedding_threshold=cfg.dedup_embedding_threshold),
        quality=QualityConfig(min_sharpness=cfg.min_sharpness,
                              min_contrast=cfg.min_contrast),
        feature_max_side=cfg.feature_max_side,
    )
    try:
        res = process_video(rec.path, cfg=pcfg)
    except IOError as exc:
        rec.error = f"open_failed:{type(exc).__name__}"
        return rec
    except ValueError as exc:
        rec.error = f"empty_video:{type(exc).__name__}"
        return rec
    except Exception as exc:  # noqa: BLE001 - 兜底但**记录类型**，不吞
        rec.error = f"decode_failed:{type(exc).__name__}"
        return rec

    rec.stats = {
        "n_frames": res.stats["n_frames"],
        "n_shots": res.stats["n_shots"],
        "n_hard_cuts": res.stats["n_hard_cuts"],
        "n_gradual": res.stats["n_gradual"],
        "n_picked": res.stats["n_picked"],
        "n_after_dedup": res.stats["n_after_dedup"],
        "n_after_quality": res.stats["n_after_quality"],
        "overall_keep_rate": res.stats["overall_keep_rate"],
        "coverage": res.stats.get("coverage", {}),
        "quality_drop_reasons": res.stats["quality"].get("drop_reasons", {}),
        "total_seconds": res.stats["total_seconds"],
        "bytes_rgb": res.stats.get("bytes_rgb", 0),
    }
    return rec


# =============================================================================
# 跑批主循环
# =============================================================================
def run_corpus(
    source: CorpusSource,
    cfg: Optional[CorpusRunConfig] = None,
    *,
    progress_every: int = 200,
) -> CorpusRunResult:
    """在整份语料上跑批。

    结构是**两遍**的：
    1. 视频侧流式处理（可并发），逐条落 checkpoint；
    2. 字幕侧全局处理（清洗 → 去重 → 统计）——
       跨视频去重必须看到全量，所以只能放在最后。
    """
    cfg = cfg or CorpusRunConfig()
    t_start = time.perf_counter()

    done_ids = _load_checkpoint(cfg.checkpoint) if cfg.checkpoint else set()
    records: List[VideoRecord] = []
    n_skipped = 0

    todo: List[VideoRecord] = []
    for rec in source:
        if rec.video_id in done_ids:
            n_skipped += 1
            continue
        if cfg.skip_missing_video and rec.path is None:
            rec.error = "missing_file"
        todo.append(rec)
        if cfg.max_videos is not None and len(todo) >= cfg.max_videos:
            break

    logger.info(f"待处理 {len(todo)} 条（断点跳过 {n_skipped} 条），"
                f"并发 {cfg.workers}")

    t_video0 = time.perf_counter()
    if cfg.workers and cfg.workers > 1:
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            futs = {ex.submit(_process_one, r, cfg): r for r in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                records.append(fut.result())
                if progress_every and k % progress_every == 0:
                    logger.info(f"  进度 {k}/{len(todo)}  "
                                f"({(time.perf_counter() - t_video0) / k:.2f} s/条)")
                if cfg.checkpoint and k % cfg.checkpoint_every == 0:
                    _append_checkpoint(cfg.checkpoint, records[-cfg.checkpoint_every:])
    else:
        for k, r in enumerate(todo, 1):
            records.append(_process_one(r, cfg))
            if progress_every and k % progress_every == 0:
                logger.info(f"  进度 {k}/{len(todo)}  "
                            f"({(time.perf_counter() - t_video0) / k:.2f} s/条)")
            if cfg.checkpoint and k % cfg.checkpoint_every == 0:
                _append_checkpoint(cfg.checkpoint, records[-cfg.checkpoint_every:])
    t_video = time.perf_counter() - t_video0

    if cfg.checkpoint and records:
        tail = records[-(len(records) % cfg.checkpoint_every or cfg.checkpoint_every):]
        _append_checkpoint(cfg.checkpoint, tail)

    # ---------------- 字幕侧（全局） ----------------
    t_cap0 = time.perf_counter()
    all_caps: List[CaptionRecord] = [c for r in records for c in r.captions]
    clean_stats = cap_mod.clean_captions(all_caps)
    dedup_stats = cap_mod.dedup_captions(all_caps)
    stats = cap_mod.caption_stats(all_caps)
    t_cap = time.perf_counter() - t_cap0

    # ---------------- 汇总 ----------------
    funnel = _build_funnel(records)
    failures = _failure_histogram(records)
    total = time.perf_counter() - t_start
    throughput = {
        "wall_seconds": round(total, 2),
        "video_seconds": round(t_video, 2),
        "caption_seconds": round(t_cap, 2),
        "videos_per_min": funnel["n_ok"] / (t_video / 60) if t_video > 0 else 0.0,
        "frames_per_sec": funnel["n_frames_decoded"] / t_video if t_video > 0 else 0.0,
        "mb_per_sec": (funnel["bytes_rgb"] / 1e6) / t_video if t_video > 0 else 0.0,
        "seconds_per_video": t_video / max(len(records), 1),
        "workers": cfg.workers,
    }
    stage_seconds = {
        "video_pipeline": round(t_video, 2),
        "caption_clean_dedup": round(t_cap, 2),
        "total": round(total, 2),
    }

    logger.info(f"跑批完成：{funnel['n_ok']}/{funnel['n_videos']} 成功，"
                f"{throughput['videos_per_min']:.1f} 视频/分钟")
    return CorpusRunResult(
        records=records, stage_seconds=stage_seconds, funnel=funnel,
        throughput=throughput, failures=failures,
        caption_clean=clean_stats, caption_dedup=dedup_stats,
        caption_stats=stats, n_skipped_resume=n_skipped,
    )


# =============================================================================
# 汇总辅助
# =============================================================================
def _build_funnel(records: Sequence[VideoRecord]) -> Dict[str, Any]:
    """逐阶段漏斗：解码 → 抽帧 → 去重 → 质量。

    ⚠️ 均值只对**成功的**视频算。把失败视频当 0 帧计入，
    会让"平均保留帧数"被拉低，看起来像质量闸门太严 ——
    这会把人的注意力引到错误的方向上（项目在 `data/video` 里踩过同类坑）。
    """
    ok = [r for r in records if r.ok]
    bad = [r for r in records if not r.ok]
    n_frames = sum(int(r.stats.get("n_frames", 0)) for r in ok)
    n_picked = sum(int(r.stats.get("n_picked", 0)) for r in ok)
    n_dedup = sum(int(r.stats.get("n_after_dedup", 0)) for r in ok)
    n_qual = sum(int(r.stats.get("n_after_quality", 0)) for r in ok)
    n_shots = sum(int(r.stats.get("n_shots", 0)) for r in ok)
    bytes_rgb = sum(int(r.stats.get("bytes_rgb", 0)) for r in ok)
    durs = [r.duration for r in ok]

    drop_reasons: Dict[str, int] = {}
    for r in ok:
        for k, v in (r.stats.get("quality_drop_reasons") or {}).items():
            drop_reasons[k] = drop_reasons.get(k, 0) + int(v)

    return {
        "n_videos": len(records),
        "n_ok": len(ok),
        "n_failed": len(bad),
        "total_duration_s": float(sum(durs)),
        "mean_duration_s": float(np.mean(durs)) if durs else 0.0,
        "median_duration_s": float(np.median(durs)) if durs else 0.0,
        "n_frames_decoded": n_frames,
        "n_picked": n_picked,
        "n_after_dedup": n_dedup,
        "n_after_quality": n_qual,
        "n_shots": n_shots,
        "shots_per_video_mean": n_shots / len(ok) if ok else 0.0,
        "frames_kept_per_video_mean": n_qual / len(ok) if ok else 0.0,
        "pick_rate": n_picked / n_frames if n_frames else 0.0,
        "dedup_keep_rate": n_dedup / n_picked if n_picked else 0.0,
        "quality_keep_rate": n_qual / n_dedup if n_dedup else 0.0,
        "overall_keep_rate": n_qual / n_frames if n_frames else 0.0,
        "quality_drop_reasons": dict(sorted(drop_reasons.items(), key=lambda kv: -kv[1])),
        "bytes_rgb": bytes_rgb,
    }


def _failure_histogram(records: Sequence[VideoRecord]) -> Dict[str, int]:
    """失败原因直方图。**这是数据集的体检报告**：
    如果 30% 是 `missing_file`，问题在下载；如果集中在 `decode_failed`，
    问题在编解码器或文件损坏 —— 两者的修法完全不同。
    """
    hist: Dict[str, int] = {}
    for r in records:
        if r.error:
            hist[r.error] = hist.get(r.error, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


# =============================================================================
# 断点续跑
# =============================================================================
def _load_checkpoint(path: Optional[Path]) -> set:
    if path is None or not Path(path).exists():
        return set()
    ids = set()
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["video_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info(f"断点：已完成 {len(ids)} 条")
    return ids


def _append_checkpoint(path: Path, records: Iterable[VideoRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(
                {"video_id": r.video_id, "error": r.error, "stats": r.stats},
                ensure_ascii=False) + "\n")


__all__ = ["CorpusRunConfig", "CorpusRunResult", "run_corpus"]
