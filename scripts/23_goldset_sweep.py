#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金种子驱动的镜头检测参数选择（跑**真实管线**，不是后筛）。

为什么不能用"后筛"
==================
第一版参数扫描是拿已检出的 `Shot.boundary_score` 重筛一遍阈值 —— 快，但**错**：
`min_shot_len` 是在**判定之后**才起作用的，它把候选边界整段丢掉，
后筛的输入里根本没有那个边界，扫多少遍 `n_signal` 都看不见它。
实测踩过：`n_signal ∈ {4,6,8,10,12,16}` 后筛结果**一模一样**，
结论"阈值调不动"其实是**扫描方法本身有盲区**。

所以这个脚本对每个配置**完整重跑 `process_video`** ——
6 条视频 × 每个配置约 1 秒，代价可以忽略，换来的是**扫描与生产同路径**。

参数网格（对应两个已定位的漏检根因）
====================================
| 参数 | 假设 |
|---|---|
| `exempt_edge_shots` | video7014#2 是首镜头内的真硬切，被 `min_shot_len=3` 丢弃 |
| `robust_n_signal`   | video7014#298 的帧差 0.4388 被阈值 0.4391 挡在门外（差 0.0003） |
| `min_shot_len`      | 见上；同时也影响短镜头（正反打）的召回 |

用法::

    python scripts/23_goldset_sweep.py
    python scripts/23_goldset_sweep.py --gold runs/goldset_msrvtt.json --out runs/goldset_sweep.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.corpus.goldset import evaluate_goldset  # noqa: E402
from roboground.data.video.pipeline import PipelineConfig, process_video  # noqa: E402
from roboground.data.video.shot import ShotDetectionConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = ROOT / "data/raw/msrvtt/test_videos/TestVideo"


def predict(paths: List[Path], cfg: ShotDetectionConfig) -> Dict[str, List[int]]:
    """对每条视频跑完整管线，返回 `{video_id: [边界帧下标, ...]}`。

    ⚠️ 边界定义：`shots[1:]` 的 `start` —— 第 0 个镜头的 `start` 恒为 0，
    那是**视频开头**不是切点，绝不能算成预测边界（算了会让 P 恒为 0）。
    """
    out: Dict[str, List[int]] = {}
    for p in paths:
        res = process_video(p, cfg=PipelineConfig(shot=cfg))
        out[p.stem] = [s.start for s in res.shots[1:]]
    return out


def run_one(paths: List[Path], gold: Dict[str, Any], cfg: ShotDetectionConfig,
            tolerance: int) -> Dict[str, Any]:
    preds = predict(paths, cfg)
    metrics = evaluate_goldset(preds, gold, tolerance=tolerance)
    metrics["preds"] = preds
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="金种子驱动的镜头检测参数扫描")
    ap.add_argument("--gold", default=str(ROOT / "runs/goldset_msrvtt.json"))
    ap.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    ap.add_argument("--out", default=str(ROOT / "runs/goldset_sweep.json"))
    ap.add_argument("--tolerance", type=int, default=2)
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    vids = [k for k in gold if not str(k).startswith("_")]
    vdir = Path(args.video_dir)
    paths = [vdir / f"{v}.mp4" for v in vids]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print("缺视频：", missing)
        return 1

    def fmt(m: Dict[str, Any]) -> str:
        return (f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
                f"TP={m['tp']:2d} FP={m['fp']:2d} FN={m['fn']:2d}")

    results: Dict[str, Any] = {"gold": args.gold, "tolerance": args.tolerance,
                               "n_videos": len(paths), "runs": {}}

    # ---------------------------------------------------------------
    # 0) 基线：现行默认
    # ---------------------------------------------------------------
    print(f"金种子：{len(paths)} 条视频，"
          f"真值边界合计 {sum(len(gold[v]) for v in vids)} 个\n")
    base = run_one(paths, gold, ShotDetectionConfig(), args.tolerance)
    results["runs"]["baseline"] = base
    print(f"{'baseline（现行默认）':<46} {fmt(base)}")

    # ---------------------------------------------------------------
    # 1) 首镜头豁免 开/关（对照）
    # ---------------------------------------------------------------
    print("\n--- ① 首/尾镜头豁免 `min_shot_len` ---")
    for flag in (False, True):
        cfg = ShotDetectionConfig(exempt_edge_shots=flag)
        m = run_one(paths, gold, cfg, args.tolerance)
        key = f"exempt_edge={flag}"
        results["runs"][key] = m
        print(f"{key:<46} {fmt(m)}")

    # ---------------------------------------------------------------
    # 2) min_shot_len × exempt_edge
    # ---------------------------------------------------------------
    print("\n--- ② `min_shot_len` × `exempt_edge_shots` ---")
    for ml, ex in itertools.product((1, 2, 3, 5), (False, True)):
        cfg = ShotDetectionConfig(min_shot_len=ml, exempt_edge_shots=ex)
        m = run_one(paths, gold, cfg, args.tolerance)
        key = f"min_shot_len={ml},exempt_edge={ex}"
        results["runs"][key] = m
        print(f"{key:<46} {fmt(m)}")

    # ---------------------------------------------------------------
    # 3) robust_n_signal（在最优 edge 设置下）
    # ---------------------------------------------------------------
    print("\n--- ③ `robust_n_signal`（`exempt_edge_shots=True`, `min_shot_len=3`）---")
    for ns in (2, 4, 8, 16, 32, 64):
        cfg = ShotDetectionConfig(robust_n_signal=ns, exempt_edge_shots=True)
        m = run_one(paths, gold, cfg, args.tolerance)
        key = f"n_signal={ns}"
        results["runs"][key] = m
        print(f"{key:<46} {fmt(m)}")

    # ---------------------------------------------------------------
    # 4) robust_k / robust_quantile 联动（阈值整体下压）
    # ---------------------------------------------------------------
    print("\n--- ④ `robust_k` × `robust_quantile` ---")
    for k, q in itertools.product((3.0, 4.5, 6.0), (0.95, 0.99)):
        cfg = ShotDetectionConfig(robust_k=k, robust_quantile=q,
                                  exempt_edge_shots=True)
        m = run_one(paths, gold, cfg, args.tolerance)
        key = f"robust_k={k},q={q}"
        results["runs"][key] = m
        print(f"{key:<46} {fmt(m)}")

    # ---------------------------------------------------------------
    # 逐视频明细（基线 vs 首镜头豁免）
    # ---------------------------------------------------------------
    print("\n--- 逐视频（baseline vs exempt_edge）---")
    for r in (results["runs"]["baseline"],
              results["runs"]["exempt_edge=True"]):
        for pv in r["per_video"]:
            print(f"  {pv['video_id']:<14} pred={pv['n_pred']} gold={pv['n_gold']} "
                  f"tp={pv['tp']} fp={pv['fp']} fn={pv['fn']}")
        print("  " + "-" * 46)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
