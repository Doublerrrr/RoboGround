#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金种子评测：产出**报告引用的最终指标**（四阶段 + 现行默认）。

为什么要有这个脚本（而不是每次手敲一段临时脚本）
================================================
报告的每一个数字都必须**可复现**。第一版这些数字是用临时脚本跑出来的，
没进仓库 —— 于是真值一修正，报告里的数字就和产物对不上了，
而**没有任何东西会提醒你**（`runs/goldset_before_after.json` 里存的还是旧真值的
结果：video7014 只标了 3 个真值，于是 F1 显示 1.000）。
把评测固化成脚本，才能保证"改了真值 / 改了默认参数 → 重跑一遍就一致"。

四个阶段
========
============  ==========================================================
阶段            配置
============  ==========================================================
修复前          旧 Otsu（无噪声地板守卫） + 渐变路径无突出度门
只修阈值        新 Otsu（守卫生效，实际回退 robust） + 渐变无门
修阈值+渐变门    robust + 渐变门，**首镜头豁免关闭**（即上一轮的状态）
现行默认        robust + 渐变门 + 首镜头豁免（本轮的改动）
============  ==========================================================

用法::

    python scripts/24_goldset_eval.py
    python scripts/24_goldset_eval.py --gold runs/goldset_msrvtt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _shot_legacy  # noqa: E402

from roboground.data.corpus.goldset import evaluate_goldset  # noqa: E402
from roboground.data.video.pipeline import PipelineConfig, process_video  # noqa: E402
from roboground.data.video.shot import ShotDetectionConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = ROOT / "data/raw/msrvtt/test_videos/TestVideo"


def predict(paths: List[Path], cfg: ShotDetectionConfig) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for p in paths:
        res = process_video(p, cfg=PipelineConfig(shot=cfg))
        # 边界 = `shots[1:]` 的 start（第 0 个镜头的 start 恒为 0，是片头不是切点）
        out[p.stem] = [s.start for s in res.shots[1:]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="金种子评测（四阶段）")
    ap.add_argument("--gold", default=str(ROOT / "runs/goldset_msrvtt.json"))
    ap.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    ap.add_argument("--out", default=str(ROOT / "runs/goldset_eval.json"))
    ap.add_argument("--out-compare",
                    default=str(ROOT / "runs/goldset_before_after.json"))
    ap.add_argument("--tolerance", type=int, default=2)
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    vids = [k for k in gold if not str(k).startswith("_")]
    paths = [Path(args.video_dir) / f"{v}.mp4" for v in vids]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print("缺视频：", missing)
        return 1

    n_gold = sum(len(gold[v]) for v in vids)
    print(f"金种子：{len(paths)} 条视频 / 真值边界 {n_gold} 个 / 容差 ±{args.tolerance} 帧\n")

    def fmt(m: Dict[str, Any]) -> str:
        return (f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
                f"TP={m['tp']:2d} FP={m['fp']:3d} FN={m['fn']:2d}")

    stages: Dict[str, Dict[str, Any]] = {}

    # ---- ① 修复前 ----
    _shot_legacy.apply_legacy_patches()
    m = evaluate_goldset(predict(paths, ShotDetectionConfig(
        threshold_mode="otsu", use_gradual_prominence=False)),
        gold, tolerance=args.tolerance)
    stages["修复前"] = m
    print(f"{'修复前（旧 Otsu + 渐变无门）':<34} {fmt(m)}")
    _shot_legacy.restore()

    # ---- ② 只修阈值 ----
    m = evaluate_goldset(predict(paths, ShotDetectionConfig(
        threshold_mode="otsu", use_gradual_prominence=False)),
        gold, tolerance=args.tolerance)
    stages["只修阈值"] = m
    print(f"{'只修阈值（噪声地板守卫生效）':<34} {fmt(m)}")

    # ---- ③ 修阈值 + 渐变门（首镜头豁免关闭）----
    m = evaluate_goldset(predict(paths, ShotDetectionConfig(
        exempt_edge_shots=False)), gold, tolerance=args.tolerance)
    stages["修阈值+渐变门"] = m
    print(f"{'修阈值 + 渐变突出度门':<34} {fmt(m)}")

    # ---- ④ 现行默认（+ 首镜头豁免）----
    m = evaluate_goldset(predict(paths, ShotDetectionConfig()),
                         gold, tolerance=args.tolerance)
    stages["现行默认"] = m
    print(f"{'现行默认（+ 首镜头豁免）':<34} {fmt(m)}")

    # ---- 逐视频明细（现行默认）----
    print("\n逐视频（现行默认）：")
    for pv in stages["现行默认"]["per_video"]:
        flag = "  ← 有漏检" if pv["fn"] else ("  ← 有误检" if pv["fp"] else "")
        print(f"  {pv['video_id']:<14} pred={pv['n_pred']} gold={pv['n_gold']} "
              f"tp={pv['tp']} fp={pv['fp']} fn={pv['fn']}{flag}")
    fn_vids = [pv["video_id"] for pv in stages["现行默认"]["per_video"] if pv["fn"]]
    if fn_vids:
        print(f"\n⚠️ 仍有漏检的视频：{fn_vids} —— 见报告第四节的取舍分析")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gold": args.gold, "tolerance": args.tolerance,
        "n_videos": len(paths), "n_gold_boundaries": n_gold,
        "metrics": stages["现行默认"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")

    oc = Path(args.out_compare)
    oc.write_text(json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {oc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
