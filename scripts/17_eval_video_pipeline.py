#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""17 · 视频数据管线评测：镜头检测量化 + 四种抽帧策略对比。

为什么这个脚本是"面试弹药"
========================
岗位面经里面试官反复追问"视频怎么切帧"，说明他们的痛点是
**抽帧策略直接决定下游效果**。所以这里不只实现，而是给出**可量化的对比**：

1. **镜头检测有真值**：用合成视频（镜头边界已知）算 P/R/F1。
   绝大多数人只能"肉眼看还行"，我们要能报出精确率/召回率。
2. **抽帧策略在固定预算下对比**：四种策略都抽同样多帧，
   比**信息效率**（镜头覆盖 / 预算均衡 / 冗余度 / 变化捕获）——
   这样"选哪个策略"是有依据的，不是凭感觉。
3. **吞吐量**：frame/s 与 GB/h，用来把"大规模"落地成可外推的工程数字。

用法::

    python scripts/17_eval_video_pipeline.py                 # 全部
    python scripts/17_eval_video_pipeline.py --only sampling # 只看抽帧对比
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.video import (                                    # noqa: E402
    STRATEGIES, PipelineConfig, SamplingConfig, ShotDetectionConfig,
    detect_shots, evaluate_shot_detection, make_synthetic_shot_video,
    process_video, sample_frames, coverage_metrics,
)
from roboground.utils.io import ensure_dir                             # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity         # noqa: E402

log = get_logger("eval_video")


# =============================================================================
def build_video(args) -> Dict[str, Any]:
    """构造镜头边界已知的合成视频。"""
    kind = "结构性异构（长短镜头混合）" if args.heterogeneous else "均匀镜头"
    log.info(f"构造合成视频：{args.shots} 个镜头，{kind} ...")
    t0 = time.perf_counter()
    frames, truth, labels = make_synthetic_shot_video(
        n_shots=args.shots, frames_per_shot=args.frames_per_shot,
        width=args.width, height=args.height, num_objects=args.objects,
        seed=args.seed, gradual_at=tuple(args.gradual_at),
        blend_frames=args.blend_frames, heterogeneous=args.heterogeneous,
    )
    dt = time.perf_counter() - t0
    counts = [labels.count(i) for i in sorted(set(labels))]
    log.info(f"合成视频就绪：{len(frames)} 帧，镜头长度 {counts}，"
             f"真值边界 {truth}（生成耗时 {dt:.1f}s）")
    return {"frames": frames, "truth": truth, "labels": labels}


# =============================================================================
def eval_shot_detection(frames, truth, args) -> Dict[str, Any]:
    """镜头检测量化：不同判据组合 + 容差。"""
    print()
    print("=" * 84)
    print("一、镜头检测量化（受控合成视频，边界已知）")
    print("=" * 84)
    print(f"  视频：{len(frames)} 帧，{len(truth)} 个真值边界 {truth}")
    print()

    variants = [
        ("仅直方图", ShotDetectionConfig(use_embedding=False)),
        ("仅直方图（固定阈值）", ShotDetectionConfig(use_embedding=False,
                                                threshold_mode="fixed")),
        ("直方图 + 渐变判据", ShotDetectionConfig(use_embedding=False,
                                             gradual_window=5,
                                             gradual_ratio=0.45)),
    ]
    rows: List[Dict[str, Any]] = []
    for name, cfg in variants:
        t0 = time.perf_counter()
        shots = detect_shots(frames, cfg=cfg)
        dt = time.perf_counter() - t0
        for tol in (0, 2):
            m = evaluate_shot_detection(shots, truth, tolerance=tol)
            rows.append({"variant": name, "tolerance": tol, "seconds": round(dt, 3), **m})

    print(f"  {'判据':<24}{'容差':>5}{'检出':>6}{'真值':>6}{'TP':>5}{'FP':>5}"
          f"{'FN':>5}{'精确率':>9}{'召回率':>9}{'F1':>8}")
    print("  " + "-" * 80)
    for r in rows:
        print(f"  {r['variant']:<24}{r['tolerance']:>5}{r['n_pred']:>6}{r['n_truth']:>6}"
              f"{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
              f"{r['precision']:>9.1%}{r['recall']:>9.1%}{r['f1']:>8.3f}")

    print()
    print("  读法：")
    print("  - 容差 0 是苛刻度量（边界必须完全对齐）；容差 2 更贴近实际（渐变边界本身有主观性）。")
    print("  - **召回率比精确率重要**：漏切一个镜头 → 两个镜头混在一起 →")
    print("    抽帧会在一段里抽到两段内容，caption 直接矛盾。多切一刀只是多几个碎片。")
    print("  - 所以阈值宁可偏松（本实现的融合判据就是「任一显著即切」）。")
    return {"rows": rows, "n_frames": len(frames), "truth": list(truth)}


# =============================================================================
def eval_sampling(frames, labels, truth, args) -> Dict[str, Any]:
    """四种抽帧策略在**固定预算**下的信息效率对比。"""
    print()
    print("=" * 84)
    print("二、抽帧策略对比（固定预算，同一段视频）")
    print("=" * 84)
    print(f"  预算：每段视频抽 {args.target_frames} 帧；镜头数 {len(truth)+1}")
    print()

    shots = detect_shots(frames, cfg=ShotDetectionConfig(use_embedding=False))
    diffs = np.array([
        float(np.mean(np.abs(np.asarray(frames[i], dtype=float)
                             - np.asarray(frames[i + 1], dtype=float))))
        for i in range(len(frames) - 1)
    ])
    # ⚠️ `labels` 是**真实**镜头号（0..n_true-1），所以 `n_shots` 必须用**真实镜头数**。
    # 第一版把 `len(shots)`（检测出的镜头数，含误报，实测 11）传了进来，
    # 而 labels 只有 0..5 —— 于是"镜头覆盖"被除以 11，最高只能到 6/11=54.5%，
    # 四种策略全都卡在这个天花板，看起来"没有区分度"，其实是**指标算错了**。
    # 教训：**评测指标的口径必须和数据来源对齐**，这里两者一个是检测输出、一个是真值。
    n_true_shots = len(set(int(x) for x in labels))

    print(f"  {'策略':<12}{'选中':>6}{'镜头覆盖':>10}{'预算均衡':>10}"
          f"{'冗余度↓':>10}{'变化捕获':>10}{'耗时(ms)':>10}")
    print("  " + "-" * 70)
    results: Dict[str, Any] = {}
    for strat in STRATEGIES:
        t0 = time.perf_counter()
        picked = sample_frames(shots, diffs, len(frames),
                               SamplingConfig(strategy=strat,
                                              target_frames=args.target_frames))
        dt = (time.perf_counter() - t0) * 1000
        cov = coverage_metrics(picked, labels, n_true_shots, diffs=diffs)
        results[strat] = {"picked": len(picked), "seconds_ms": round(dt, 2), **cov}
        print(f"  {strat:<12}{len(picked):>6}{cov['shot_coverage']:>10.1%}"
              f"{cov['shot_balance']:>10.3f}{cov['redundancy']:>10.3f}"
              f"{cov['motion_captured']:>10.1%}{dt:>10.2f}")

    print()
    print("  指标含义（都在「同样抽 N 帧」的前提下比）：")
    print("  - **镜头覆盖**：有多少镜头至少被抽到一帧。漏掉镜头 = 整段场景没进训练数据。")
    print("  - **预算均衡**：1-基尼系数，衡量配额的分散程度。")
    print("  - **冗余度↓**：抽中帧之间的平均相似度（越低越好）。")
    print("  - **变化捕获**：抽中帧覆盖了全片多少帧间变化量。")
    print()
    best_cov = max(results.items(), key=lambda kv: kv[1]["shot_coverage"])
    best_mot = max(results.items(), key=lambda kv: kv[1]["motion_captured"])
    print(f"  → 镜头覆盖最高：**{best_cov[0]}**（{best_cov[1]['shot_coverage']:.1%}）")
    print(f"  → 变化捕获最高：**{best_mot[0]}**（{best_mot[1]['motion_captured']:.1%}）")
    return results


# =============================================================================
def eval_pipeline(frames, args) -> Dict[str, Any]:
    """整条管线的端到端吞吐与各阶段淘汰率。"""
    print()
    print("=" * 84)
    print("三、端到端管线与吞吐量（「大规模」的工程证据）")
    print("=" * 84)
    print()
    print(f"  {'策略':<12}{'原始':>7}{'抽中':>7}{'去重后':>8}{'过质量':>8}"
          f"{'保留率':>9}{'frame/s':>10}{'GB/h':>9}")
    print("  " + "-" * 72)
    results: Dict[str, Any] = {}
    for strat in STRATEGIES:
        cfg = PipelineConfig(sampling=SamplingConfig(
            strategy=strat, target_frames=args.target_frames))
        res = process_video(frames, cfg=cfg)
        s = res.stats
        results[strat] = s
        print(f"  {strat:<12}{s['n_frames']:>7}{s['n_picked']:>7}"
              f"{s['n_after_dedup']:>8}{s['n_after_quality']:>8}"
              f"{s['overall_keep_rate']:>9.1%}{s['throughput_fps']:>10.1f}"
              f"{s['throughput_gb_per_hour']:>9.2f}")

    any_stats = next(iter(results.values()))
    print()
    print(f"  耗时分解（以 {next(iter(results))} 为例）：{any_stats['stage_seconds']}")
    print()
    print("  ★ 怎么用这组数字回答「大规模数据处理」：")
    fps = any_stats["throughput_fps"]
    print(f"     实测 {fps:.1f} frame/s（单进程、{args.width}×{args.height}）。")
    print(f"     → 处理 100 万帧约需 {1e6 / max(fps, 1e-9) / 3600:.1f} 小时单进程；")
    print("     → 管线已按「分块无状态」设计，N 进程线性外推，且内存与视频长度无关。")
    print("     → 真正的瓶颈在**特征计算**（见上方耗时分解），生产上应 GPU batch + 抽稀。")
    return results


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=6, help="合成视频的镜头数")
    ap.add_argument("--frames-per-shot", type=int, default=12)
    ap.add_argument("--target-frames", type=int, default=24, help="抽帧预算")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gradual-at", type=int, nargs="*", default=[3],
                    help="这些镜头编号处用渐变过渡（验证渐变判据）")
    ap.add_argument("--blend-frames", type=int, default=10,
                    help="渐变过渡长度（帧）。太短就不构成渐变盲区，实测 4 帧时逐帧差分照样能抓到")
    ap.add_argument("--heterogeneous", action="store_true",
                    help="用结构性异构视频（长短镜头混合）—— 抽帧策略的区分力依赖它")
    ap.add_argument("--only", choices=["shot", "sampling", "pipeline"], default=None)
    ap.add_argument("--output", default="runs/video_pipeline.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    video = build_video(args)
    report: Dict[str, Any] = {"config": vars(args)}

    if args.only in (None, "shot"):
        report["shot_detection"] = eval_shot_detection(video["frames"],
                                                       video["truth"], args)
    if args.only in (None, "sampling"):
        report["sampling"] = eval_sampling(video["frames"], video["labels"],
                                           video["truth"], args)
    if args.only in (None, "pipeline"):
        report["pipeline"] = eval_pipeline(video["frames"], args)

    print()
    print("=" * 84)
    print("面试可讲的三句话")
    print("=" * 84)
    print("""
  1. **"切帧的语义单位是镜头，不是秒。"**
     不做镜头检测直接均匀抽帧，会出现「静止长镜头刷屏 + 短镜头漏采」两种坏数据。
     正确顺序：镜头检测 → 抽帧 → 去重 → 质量过滤。

  2. **"抽帧不是采样问题，是预算分配问题。"**
     总帧数固定时，应该按 **内容变化率** 而不是时间长度分配。
     我实现了四种策略在**同一预算**下对比：均匀 / 三等分 / 关键帧 / 内容自适应，
     并用镜头覆盖、预算均衡、冗余度、变化捕获四个指标量化 —— 
     **选哪个策略是有数据的，不是凭感觉。**

  3. **"去重必须两级级联。"**
     感知哈希（dHash）砍近重复像素，嵌入聚类砍语义重复。
     只做哈希会留下大量"同物体不同角度"的语义重复，训练时等于给同一概念加权；
     只做嵌入则算力吃不消。哈希先砍掉大部分，剩下的才进模型。
""")

    if args.output:
        ensure_dir(Path(args.output).parent)
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        log.ok(f"结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
