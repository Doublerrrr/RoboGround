#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""镜头检测消融：量化"修复前 vs 修复后"的差异。

为什么单独写一个脚本
--------------------
`threshold_mode="otsu"` **不能**用来复现修复前的行为 —— 修复给
`_otsu_threshold` 加了"阈值必须高于噪声地板"的守卫，于是现在即使显式选
`otsu` 也会在真实视频上回退到 `robust`，两者结果一模一样（实测 36 vs 36）。
要测真正的修复前行为，必须**把旧的实现整个换回去**。

旧实现放在 `scripts/_shot_legacy.py` 里共享（`24_goldset_eval.py` 也要用）——
**同一份旧实现只能有一处**，否则又是"修一边漏一边"。

用法::

    python scripts/22_shot_ablation.py --videos video7010 ... video7019
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _shot_legacy  # noqa: E402

from roboground.data.video import shot as S  # noqa: E402
from roboground.data.video.pipeline import PipelineConfig, process_video  # noqa: E402
from roboground.data.video.shot import ShotDetectionConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def count_shots(paths, cfg: ShotDetectionConfig) -> list:
    counts, preds = [], {}
    for p in paths:
        res = process_video(p, cfg=PipelineConfig(shot=cfg))
        counts.append(len(res.shots))
        preds[p.stem] = [s.start for s in res.shots[1:]]
    return counts, preds


def main() -> int:
    ap = argparse.ArgumentParser(description="镜头检测消融")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--video-dir",
                    default=str(ROOT / "data/raw/msrvtt/test_videos/TestVideo"))
    ap.add_argument("--out", default=str(ROOT / "runs/shot_ablation.json"))
    args = ap.parse_args()

    vdir = Path(args.video_dir)
    paths = [vdir / (v if v.endswith(".mp4") else f"{v}.mp4") for v in args.videos]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("没有可用视频")
        return 1

    results = {}

    # ---- 修复前：旧 Otsu（无守卫） + 渐变无突出度门 ----
    _shot_legacy.apply_legacy_patches()
    old_cfg = ShotDetectionConfig(threshold_mode="otsu",
                                  use_gradual_prominence=False)
    c, p = count_shots(paths, old_cfg)
    results["before"] = {"counts": c, "total": sum(c), "per_video": sum(c) / len(c),
                         "preds": p}
    print(f"修复前（旧 Otsu + 渐变无门）  总镜头={sum(c):5d}  每视频={sum(c)/len(c):6.1f}")

    # ---- 只修阈值：新 Otsu（带守卫，实际回退 robust） + 渐变无门 ----
    _shot_legacy.restore()
    mid_cfg = ShotDetectionConfig(threshold_mode="otsu",
                                  use_gradual_prominence=False)
    c, p = count_shots(paths, mid_cfg)
    results["threshold_only"] = {"counts": c, "total": sum(c),
                                 "per_video": sum(c) / len(c), "preds": p}
    print(f"只修阈值（新守卫生效）        总镜头={sum(c):5d}  每视频={sum(c)/len(c):6.1f}")

    # ---- 现行默认：robust + 渐变突出度门 + 首镜头豁免 ----
    c, p = count_shots(paths, ShotDetectionConfig())
    results["after"] = {"counts": c, "total": sum(c), "per_video": sum(c) / len(c),
                        "preds": p}
    print(f"修复后（robust + 渐变门 + 首镜头豁免）  总镜头={sum(c):5d}  "
          f"每视频={sum(c)/len(c):6.1f}")
    print(f"视频数={len(paths)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
