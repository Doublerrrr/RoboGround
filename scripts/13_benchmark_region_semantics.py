#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""13 · 区域语义质量评测（GT 区域，干净口径）与 TTA 消融。

为什么需要这个脚本（与 11 的区别）
--------------------------------
`11_ablate_clip_pooling.py` 用的是**检测器输出的区域 + 检测器给的标签**，
测出来的分数同时包含"检测错"和"编码错"两种误差 —— 无法定位问题。

本脚本用 **SUN RGB-D 的 GT 3D 框投影出 2D 区域 + GT 类别名**：
- 区域是准的（来自人工标注的 3D 框）；
- 标签是准的（来自数据集类别）；
- 深度有效性照旧过滤（没有深度的区域不该参与评测）。

于是测出来的就是**纯粹的"区域级图文表示质量"**，可以直接对比不同编码器与策略。

评估的维度
----------
1. **编码器**：clip / siglip（以及未来任何 supports_text 的后端）
2. **池化**：crop / mask
3. **多视图 TTA**：none / flip / multicrop
4. **文本模板**：`a photo of a {}` / `{}` / `a {}`

用法::

    python scripts/13_benchmark_region_semantics.py --scenes 2
    python scripts/13_benchmark_region_semantics.py --scenes 2 --quick     # 只跑最优组合
    python scripts/13_benchmark_region_semantics.py --scenes 2 --output runs/region_sem.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.io import ensure_dir                      # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity   # noqa: E402

log = get_logger("region_semantics")

#: 常见室内类别的标准 10 类（保持与 SUN RGB-D 官方评测一致，减少噪声类别）
EVAL_CLASSES = [
    "bed", "table", "sofa", "chair", "toilet",
    "desk", "cabinet", "nightstand", "bookshelf", "bathtub",
    "monitor", "box", "door", "window", "trash can",
]

TEMPLATE_VARIANTS = {
    "photo_of_a": "a photo of a {}.",
    "bare": "{}.",
    "a_x": "a {}.",
}


def collect_gt_regions(cfg, args):
    """收集"GT 区域 + GT 标签"样本。"""
    from roboground.data.synthetic import rotz
    from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene
    from roboground.types import Detection2D

    index = load_scene_index(str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz"))
    eval_set = {c.lower() for c in EVAL_CLASSES}
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))

    samples = []
    used = []
    for pos in np.argsort(-counts)[:4000]:
        i = int(pos)
        if len(used) >= args.scenes:
            break
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = [str(x) for x in np.asarray(index["label_flat"][off0:off1]).ravel()]
        if len({l.lower() for l in labels} & eval_set) < 2:
            continue

        scene = load_sunrgbd_scene(index, i, max_depth=8.0, resize=(args.width, args.height))
        if scene is None or scene.boxes_3d.shape[0] == 0:
            continue

        frame = scene.to_frame()
        valid_d = (frame.depth_m > 0.1) & (frame.depth_m < 8.0)
        h, w = frame.shape

        dets = []
        for k in range(scene.boxes_3d.shape[0]):
            label = str(scene.labels[k]).lower()
            if label not in eval_set:
                continue
            cx, cy, cz, dx, dy, dz = scene.boxes_3d[k, :6]
            heading = float(scene.boxes_3d[k, 6]) if scene.boxes_3d.shape[1] > 6 else 0.0

            # 8 个角点 → 投影 → 2D AABB
            half = np.abs([dx, dy, dz]) / 2.0
            signs = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=np.float64)
            corners = (signs * half) @ rotz(heading).T + np.array([cx, cy, cz])
            cam = frame.pose.world_to_cam(corners)
            z = cam[:, 2]
            front = z > 0.3
            if front.sum() < 2:
                continue
            u = cam[front, 0] * frame.intrinsics.fx / z[front] + frame.intrinsics.cx
            v = cam[front, 1] * frame.intrinsics.fy / z[front] + frame.intrinsics.cy
            x1, y1 = float(u.min()), float(v.min())
            x2, y2 = float(u.max()), float(v.max())
            x1, x2 = max(0.0, min(x1, w - 1)), max(0.0, min(x2, w - 1))
            y1, y2 = max(0.0, min(y1, h - 1)), max(0.0, min(y2, h - 1))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            # 区域内必须有足够的有效深度（否则反投影本来就没意义）
            xi1, xi2 = int(x1), int(np.ceil(x2))
            yi1, yi2 = int(y1), int(np.ceil(y2))
            n_valid = int(valid_d[yi1:yi2, xi1:xi2].sum())
            if n_valid < args.min_valid_px:
                continue

            dets.append(Detection2D(label=label, score=1.0,
                                    bbox=np.array([x1, y1, x2, y2])))

        if len(dets) >= 2:
            samples.append((frame, dets))
            used.append({"scene": i, "num_regions": len(dets),
                         "labels": sorted({d.label for d in dets})})

    if not samples:
        raise RuntimeError("没有收集到 GT 区域样本")
    log.info(f"使用场景：{[u['scene'] for u in used]}")
    for u in used:
        log.info(f"  场景 {u['scene']}: {u['num_regions']} 个 GT 区域，类别 {u['labels']}")
    return samples, used


def build_encoder(cfg, spec: str, pool: str, tta: str, clip_path: str, siglip_path: str):
    from roboground.perception import build_encoder as _build

    name = "siglip" if "siglip" in spec.lower() else "clip"
    path = siglip_path if name == "siglip" else clip_path
    cfg.set("perception.encoder_kwargs.model_id", path)
    enc = _build(cfg, name=name)
    enc.model_id = path
    enc.pool = pool
    enc.tta = tta
    return enc


def evaluate(samples, enc, template_key: str, null_texts) -> dict:
    from roboground.perception.encoders.siglip_encoder import SigLIPEncoder

    native = bool(getattr(enc, "is_calibrated", False))
    tpl = TEMPLATE_VARIANTS[template_key]

    all_labels = sorted({d.label for _, dets in samples for d in dets})
    texts = [tpl.format(l) for l in all_labels]

    self_s, other_s, null_s, top1, top3, ranks = [], [], [], [], [], []
    for frame, dets in samples:
        feats = enc.encode_regions(frame, dets)
        txt = enc.encode_text(texts)
        nul = enc.encode_text(null_texts)

        if native:
            sims = enc.pair_scores(feats, txt)
            sims_null = enc.pair_scores(feats, nul)
        else:
            sims = feats @ txt.T
            sims_null = feats @ nul.T

        for i, d in enumerate(dets):
            if i >= feats.shape[0]:
                continue
            j = all_labels.index(d.label)
            s_self = float(sims[i, j])
            s_other = float(max(sims[i, k] for k in range(len(texts)) if k != j))
            self_s.append(s_self)
            other_s.append(s_other)
            null_s.append(float(sims_null[i].max()))
            order = np.argsort(-sims[i]).tolist()
            rank = order.index(j) + 1
            ranks.append(rank)
            top1.append(int(rank == 1))
            top3.append(int(rank <= 3))

    ss, os_, ns = np.array(self_s), np.array(other_s), np.array(null_s)
    return {
        "n": int(ss.size),
        "self": float(ss.mean()), "other": float(os_.mean()), "null": float(ns.mean()),
        "separation": float((ss - os_).mean()),
        "margin_pos": float((ss - ns).mean()),
        "top1": float(np.mean(top1)), "top3": float(np.mean(top3)),
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean([1.0 / r for r in ranks])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--min-valid-px", type=int, default=200)
    ap.add_argument("--clip-path", default=r"G:\RoboGround\weights\clip-vit-base-patch32")
    ap.add_argument("--siglip-path", default=r"G:\RoboGround\weights\siglip-base-patch16-224")
    ap.add_argument("--encoders", nargs="*", default=["clip", "siglip"])
    ap.add_argument("--pools", nargs="*", default=["crop", "mask"])
    ap.add_argument("--ttas", nargs="*", default=["none", "multicrop"])
    ap.add_argument("--templates", nargs="*", default=["photo_of_a"])
    ap.add_argument("--quick", action="store_true", help="只跑 siglip+mask+multicrop")
    ap.add_argument("--output", default="runs/region_semantics.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)
    if args.quick:
        args.encoders, args.pools, args.ttas = ["siglip"], ["mask"], ["multicrop"]

    from roboground import load_config

    cfg = load_config("configs/perception_openvocab.yaml")
    null_texts = list(cfg.get("perception.encoder_kwargs.null_texts", []) or []) or [
        "a photo of nothing.", "an empty background.", "a photo of a blank wall.",
        "an out of focus image.", "a photo of the floor.", "a random texture.",
    ]

    samples, meta = collect_gt_regions(cfg, args)
    total = sum(len(d) for _, d in samples)
    log.info(f"共 {total} 个 GT 区域参与评测（标签来自数据集 GT，无检测噪声）")

    print()
    header = (f"{'编码器':<8}{'池化':<7}{'TTA':<11}{'模板':<12}{'n':>5}"
              f"{'self':>8}{'other':>8}{'分离度':>10}{'Top1':>7}{'Top3':>7}{'MRR':>7}{'排名':>6}")
    print(header)
    print("-" * len(header))

    rows = []
    for spec, pool, tta, tpl in itertools.product(
        args.encoders, args.pools, args.ttas, args.templates
    ):
        try:
            enc = build_encoder(cfg, spec, pool, tta, args.clip_path, args.siglip_path)
            r = evaluate(samples, enc, tpl, null_texts)
            r.update({"encoder": spec, "pool": pool, "tta": tta, "template": tpl})
            rows.append(r)
            print(f"{spec:<8}{pool:<7}{tta:<11}{tpl:<12}{r['n']:>5}{r['self']:>8.3f}"
                  f"{r['other']:>8.3f}{r['separation']:>+10.4f}{r['top1']:>7.2f}"
                  f"{r['top3']:>7.2f}{r['mrr']:>7.3f}{r['median_rank']:>6.1f}")
        except Exception as exc:
            log.warn(f"{spec}/{pool}/{tta}/{tpl} 失败：{type(exc).__name__}: {str(exc)[:140]}")

    print()
    if not rows:
        log.error("所有组合都失败")
        return 2

    best = max(rows, key=lambda r: (r["top1"], r["mrr"]))
    log.ok(f"最佳组合：{best['encoder']} + pool={best['pool']} + tta={best['tta']} "
           f"+ template={best['template']}")
    log.info(f"  Top-1={best['top1']:.3f}  Top-3={best['top3']:.3f}  "
             f"MRR={best['mrr']:.3f}  中位排名={best['median_rank']:.1f}  "
             f"分离度={best['separation']:+.4f}")

    # 与"基线"（每个编码器的第一个组合）对比
    for spec in args.encoders:
        spec_rows = [r for r in rows if r["encoder"] == spec]
        if len(spec_rows) < 2:
            continue
        base = spec_rows[0]
        gain = best["top1"] - base["top1"]
        log.info(f"  {spec}: 基线 Top1={base['top1']:.2f} "
                 f"({base['pool']}/{base['tta']}) → 最优 {best['top1']:.2f}  "
                 f"（{'提升' if gain > 0 else '变化'} {gain:+.2f}）")

    if args.output:
        ensure_dir(Path(args.output).parent)
        Path(args.output).write_text(
            json.dumps({"rows": rows, "best": best, "scenes": meta},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.ok(f"结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
