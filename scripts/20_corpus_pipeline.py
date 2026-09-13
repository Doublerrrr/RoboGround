#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""20 · 大规模多模态语料工程：真实数据上的数据管线全流程。

为什么需要这个脚本（岗位驱动的判断）
==================================
面经里面试官的原话是"更想找做过**大规模多模态数据处理**经验的"，
并且**反复追问"视频怎么切帧"**。而项目原有的 `17_eval_video_pipeline.py`
跑的是**受控合成视频**（72 帧、边界程序生成）—— 那能证明**逻辑对**，
证明不了**在真实数据上、在数据规模上**成立。

本脚本补的正是这一段：

1. **真实语料**：MSR-VTT 测试集（2990 条真实 YouTube 短片 / 59,800 条人工字幕）；
2. **真实规模**：逐条跑完整管线，给出漏斗、吞吐、失败归因；
3. **真实核验**：镜头检测在**人工核验的金种子**上重测 P/R/F1
   （合成上的结论不能直接外推，见 `data/corpus/goldset.py`）；
4. **数据工程闭环**：字幕清洗 → 三级去重 → 温度配比 → token 预算 → 分片打包 → 数据卡。

用法::

    # 冒烟（合成数据，几秒，不需要下载）
    python scripts/20_corpus_pipeline.py --source synthetic --limit 8

    # 真实数据（需要先下载 MSR-VTT 测试集）
    python scripts/20_corpus_pipeline.py --source msrvtt --limit 200
    python scripts/20_corpus_pipeline.py --source msrvtt --full

    # 只出金种子核验材料（不跑全量）
    python scripts/20_corpus_pipeline.py --source msrvtt --goldset-only --goldset-videos 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.corpus import (                                     # noqa: E402
    CorpusRunConfig, GoldSetConfig, MSRVTTConfig, MixtureConfig, PackConfig,
    SyntheticCorpusConfig, build_datacard, build_source, evaluate_goldset,
    iter_samples, load_goldset, pack_shards, plan_mixture,
    render_verification_material, token_budget_report,
)
from roboground.utils.io import ensure_dir                               # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity           # noqa: E402

log = get_logger("corpus_pipeline")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNO = ROOT / "data/raw/msrvtt/test_videodatainfo.json.zip"
DEFAULT_VIDEO_DIR = ROOT / "data/raw/msrvtt/test_videos"


# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="大规模多模态语料工程")
    p.add_argument("--source", default="msrvtt",
                   choices=["msrvtt", "activitynet", "synthetic"])
    p.add_argument("--anno", default=str(DEFAULT_ANNO), help="标注文件")
    p.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    p.add_argument("--limit", type=int, default=0,
                   help="只处理前 N 条（0 = 全量）")
    p.add_argument("--full", action="store_true", help="全量（等同 --limit 0）")
    p.add_argument("--target-frames", type=int, default=24)
    p.add_argument("--strategy", default="adaptive")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--group-key", default="category")
    p.add_argument("--token-budget", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--out", default=str(ROOT / "runs/corpus_report.json"))
    p.add_argument("--datacard", default=str(ROOT / "docs/数据卡_MSRVTT.md"))
    p.add_argument("--pack-dir", default=str(ROOT / "data/packed/msrvtt"))
    p.add_argument("--checkpoint", default=str(ROOT / "runs/corpus_checkpoint.jsonl"))
    # 金种子
    p.add_argument("--goldset-only", action="store_true",
                   help="只渲染镜头检测核验材料，不跑全量管线")
    p.add_argument("--goldset-videos", type=int, default=12)
    p.add_argument("--goldset-dir", default=str(ROOT / "runs/goldset"))
    p.add_argument("--goldset-json", default=str(ROOT / "runs/goldset_msrvtt.json"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def make_source(args) -> Any:
    if args.source == "msrvtt":
        anno = Path(args.anno)
        if not anno.exists():
            raise SystemExit(
                f"找不到标注文件：{anno}\n"
                f"请先下载：\n"
                f"  curl -L -o {anno} \\\n"
                f"    https://hf-mirror.com/datasets/AlexZigma/msr-vtt/resolve/main/data/test_videodatainfo.json.zip\n"
                f"并解压视频：\n"
                f"  unzip {ROOT / 'data/raw/msrvtt/test_videos.zip'} -d {args.video_dir}")
        vdir = Path(args.video_dir)
        return build_source("msrvtt", cfg=MSRVTTConfig(
            anno=anno, video_dir=vdir if vdir.exists() else None,
            limit=args.limit or None))
    if args.source == "synthetic":
        return build_source("synthetic", cfg=SyntheticCorpusConfig(
            n_videos=args.limit or 8))
    return build_source(args.source, anno=Path(args.anno),
                        video_dir=Path(args.video_dir),
                        limit=args.limit or None)


# =============================================================================
def run_goldset(args, source) -> Dict[str, Any]:
    """渲染镜头检测核验材料；若已有金种子标注则直接出指标。"""
    from roboground.data.video import PipelineConfig, SamplingConfig, process_video

    cfg = GoldSetConfig(out_dir=Path(args.goldset_dir))
    recs = [r for r in source if r.path is not None and Path(r.path).exists()]
    recs = recs[: args.goldset_videos]
    if not recs:
        log.warn("没有可用的视频文件，跳过金种子核验")
        return {}

    preds: Dict[str, List[int]] = {}
    for r in recs:
        try:
            res = process_video(r.path, cfg=PipelineConfig(
                sampling=SamplingConfig(strategy=args.strategy,
                                        target_frames=args.target_frames)))
        except Exception as exc:  # noqa: BLE001
            log.warn(f"{r.video_id} 处理失败：{exc}")
            continue
        bounds = [s.start for s in res.shots if s.index > 0]
        preds[r.video_id] = bounds
        info = render_verification_material(r, bounds, cfg=cfg)
        log.info(f"  {r.video_id}: {info.get('n_frames')} 帧，"
                 f"检出 {len(bounds)} 个边界 → {len(info.get('sheets', []))} 张核验图")

    out: Dict[str, Any] = {"n_videos": len(preds), "predictions": preds}
    # ⚠️ `goldset_predictions.json` **必须每次都写**。
    # 第一版把它写在 `else`（金种子文件不存在）分支里 —— 于是金种子一建好，
    # 这个产物就**永久冻结在首跑状态**，却仍然看起来像"当前的预测结果"。
    # 实测后果：`exempt_edge_shots` 改了默认之后，核验图已经重新渲染成 4 个边界，
    # 而 predictions 里还是旧的 3 个 —— 两个产物**互相矛盾**。
    # 这是典型的静默失效：不报错、不崩溃，只是产物悄悄过期。
    preds_path = Path(args.out).with_name("goldset_predictions.json")
    preds_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    gold_path = Path(args.goldset_json)
    if gold_path.exists():
        gold = load_goldset(gold_path)
        out["goldset"] = evaluate_goldset(preds, gold, tolerance=cfg.tolerance)
        log.info(f"金种子评测：P={out['goldset']['precision']:.3f} "
                 f"R={out['goldset']['recall']:.3f} F1={out['goldset']['f1']:.3f} "
                 f"（{out['goldset']['n_videos_verified']} 条视频）")
    else:
        out["goldset"] = {
            "status": "pending_manual_verification",
            "hint": f"核验图已渲染到 {cfg.out_dir}，"
                    f"人工核对后把 {{video_id: [真实边界帧号]}} 写入 {gold_path} 即可出指标",
            "predictions_path": str(preds_path),
        }
    return out


# =============================================================================
def main() -> int:
    args = build_argparser().parse_args()
    set_verbosity(2 if args.verbose else 1)
    t0 = time.perf_counter()

    log.info("=" * 68)
    log.info("大规模多模态语料工程")
    log.info("=" * 68)

    source = make_source(args)
    log.info(f"数据源：{source.describe()}")

    # ---------------- 金种子模式 ----------------
    if args.goldset_only:
        gs = run_goldset(args, source)
        out_path = Path(args.out).with_name("goldset_report.json")
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(gs, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        log.info(f"金种子报告 → {out_path}")
        return 0

    # ---------------- 主跑批 ----------------
    run_cfg = CorpusRunConfig(
        target_frames=args.target_frames,
        sampling_strategy=args.strategy,
        max_videos=args.limit or None,
        workers=args.workers,
        checkpoint=Path(args.checkpoint),
    )
    result = run_corpus_safe(source, run_cfg)

    log.info("-" * 68)
    log.info("跑批结果：")
    for line in result.summary_text().splitlines():
        log.info(line)

    # ---------------- 配比 + 打包 ----------------
    samples = list(iter_samples(result.records, group_key=args.group_key))
    log.info(f"训练样本（视频×字幕对）：{len(samples):,} 条")

    mix_cfg = MixtureConfig(
        group_key=args.group_key, temperature=args.temperature,
        token_budget=args.token_budget or None,
    )
    selected, mix_stats = plan_mixture(samples, mix_cfg)
    tok = token_budget_report(selected)
    log.info(f"配比后：{len(selected):,} 条，{tok['total_tokens']:,} tokens")

    manifest = pack_shards(selected, PackConfig(
        shard_size=args.shard_size, out_dir=Path(args.pack_dir)))
    log.info(f"打包：{manifest['n_shards']} 片 → {args.pack_dir}")

    # ---------------- 金种子（附带） ----------------
    goldset = {}
    try:
        goldset = run_goldset(args, source)
    except Exception as exc:  # noqa: BLE001
        log.warn(f"金种子核验跳过：{exc}")

    # ---------------- 数据卡 ----------------
    run_snapshot = {
        "source": args.source, "limit": args.limit or "full",
        "target_frames": args.target_frames, "strategy": args.strategy,
        "workers": args.workers, "temperature": args.temperature,
        "group_key": args.group_key, "token_budget": args.token_budget or None,
    }
    card = build_datacard(result, source_desc=source.describe(),
                          run_cfg=run_snapshot, mixture=mix_stats)
    dc_path = Path(args.datacard)
    ensure_dir(dc_path.parent)
    dc_path.write_text(card, encoding="utf-8")
    log.info(f"数据卡 → {dc_path}")

    # ---------------- 汇总落盘 ----------------
    payload = {
        "run_config": run_snapshot,
        "source": source.describe(),
        "run": {k: v for k, v in result.to_dict().items() if k != "videos"},
        "mixture": mix_stats,
        "token_budget": tok,
        "packing": manifest,
        "goldset": goldset,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log.info("=" * 68)
    log.info(f"完成，用时 {payload['wall_seconds']}s → {out_path}")
    return 0


def run_corpus_safe(source, run_cfg):
    from roboground.data.corpus import run_corpus
    return run_corpus(source, run_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
