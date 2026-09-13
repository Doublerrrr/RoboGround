#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""06 · 自动化标注（Stage 5 数据闭环）。

产出三份文件（对应三种用途）：
- `coco.json`   → 2D 检测/分割标注（可直接喂 YOLO/Mask2Former）
- `boxes3d.json`→ 3D 框标注（喂 3D 检测 / VLM 空间推理）
- `review.json` → 被质量闸门拦下、需要人工复核的样本（**不是丢弃**）

四道质量闸门：置信度 / 几何有效性（点数）/ 尺度合理性（物理先验）/ 跨帧一致性。

用法::

    python scripts/06_auto_label.py --source synthetic --frames 6
    python scripts/06_auto_label.py --source sunrgbd --scenes 5
    python scripts/06_auto_label.py --source synthetic --out runs/labels --check-consistency
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config                        # noqa: E402
from roboground.data.auto_label import PLAUSIBLE_SIZE_RANGES, AutoLabeler  # noqa: E402
from roboground.utils.io import save_json                        # noqa: E402
from roboground.utils.logging import get_logger                  # noqa: E402

PROMPTS = ["chair", "table", "desk", "monitor", "door", "window",
           "cup", "box", "bottle", "sofa", "bed", "bookshelf", "trash can"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "sunrgbd"], default="synthetic")
    ap.add_argument("--frames", type=int, default=6, help="合成模式帧数")
    ap.add_argument("--scenes", type=int, default=5, help="真实模式场景数")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--min-score", type=float, default=None, help="置信度闸门")
    ap.add_argument("--min-points", type=int, default=None, help="几何有效性闸门（最小 3D 点数）")
    ap.add_argument("--no-scale-check", action="store_true", help="关闭尺度合理性闸门")
    ap.add_argument("--check-consistency", action="store_true", help="额外做跨帧一致性检查")
    ap.add_argument("--out", default="runs/labels")
    args = ap.parse_args()

    log = get_logger("auto_label")
    cfg = load_config()
    prompts = list(args.prompts) if args.prompts else PROMPTS
    cfg.set("perception.prompts", prompts)
    if args.min_score is not None:
        cfg.set("perception.detector_kwargs.box_threshold", float(args.min_score))

    # ---- 收集数据 ----
    if args.source == "synthetic":
        from roboground.data.synthetic import make_synthetic_sequence

        frames = make_synthetic_sequence(
            seed=args.seed, num_frames=args.frames,
            width=args.width, height=args.height, num_objects=6,
        )
    else:
        from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene

        index_path = str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz")
        try:
            index = load_scene_index(index_path)
        except FileNotFoundError as exc:
            log.error(str(exc))
            log.info("请先运行：python scripts/02_build_sunrgbd_index.py")
            return 2

        wanted = {p.lower() for p in prompts}
        counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))
        frames = []
        for pos in np.argsort(-counts):
            i = int(pos)
            if len(frames) >= args.scenes:
                break
            off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
            labels = {str(x).lower() for x in np.asarray(index["label_flat"][off0:off1]).ravel()}
            if not (labels & wanted):
                continue
            scene = load_sunrgbd_scene(index, i, max_depth=8.0, resize=(args.width, args.height))
            if scene is not None and scene.boxes_3d.shape[0] > 0:
                frames.append(scene.to_frame())

    if not frames:
        log.error("没有收集到任何帧")
        return 2
    log.info(f"待标注帧数：{len(frames)}")

    # ---- 标注 ----
    labeler = AutoLabeler(cfg, prompts=prompts)
    if args.min_points is not None:
        labeler.min_points = int(args.min_points)
    if args.no_scale_check:
        labeler.check_scale = False

    report = labeler.label_frames(frames, out_dir=args.out)

    log.kv("标注报告", {k: v for k, v in report.items() if k != "class_distribution"})
    log.info("类别分布：")
    for label, cnt in report["class_distribution"].items():
        log.info(f"    {label:<18} {cnt}")
    if report["reject_reasons"]:
        log.info("拒绝原因分布：")
        for reason, cnt in report["reject_reasons"].items():
            log.info(f"    {reason:<24} {cnt}")

    # ---- 跨帧一致性（主动学习信号）----
    if args.check_consistency:
        items = []
        for f in frames:
            items.extend(labeler.label_frame(f))
        suspects = AutoLabeler.cross_frame_consistency(items)
        log.info(f"跨帧一致性检查：{len(suspects)} 条可疑标注")
        for s in suspects[:10]:
            log.info(f"    {s['label']:<14} 偏差 {s['deviation_m']}m  ({s['frame_id']})")
        if suspects:
            save_json(suspects, Path(args.out) / "consistency_suspects.json")

    log.kv("采纳率", {"候选": report["num_candidates"], "采纳": report["num_accepted"],
                      "待复核": report["num_rejected"],
                      "采纳率": f"{report['accept_rate']:.1%}",
                      "每帧耗时(ms)": report["ms_per_frame"]})
    log.ok(f"输出目录：{report.get('output_dir', args.out)}")
    log.info("提示：闸门阈值可通过 --min-score / --min-points 调整；"
             "尺度先验表见 roboground/data/auto_label.py 的 PLAUSIBLE_SIZE_RANGES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
