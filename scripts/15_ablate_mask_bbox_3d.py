#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""15 · SAM 掩码 vs bbox 掩码：3D 定位精度对比（补齐消融里缺失的那一项）。

背景：之前只量了"背景剔除率"
----------------------------
`scripts/09_test_real_backends.py` 已经量出 **SAM 掩码面积是 bbox 的 51%**
（即剔除了 48.8% 的背景像素）。但那只是**中间指标**，没有回答真正的问题：

> **换成 SAM 掩码之后，3D 定位到底准了多少？**

本脚本就补这一项，用 SUN RGB-D 的 **GT 3D 框**作为真值：
```
GT 框 → 投影出 2D 区域 → 两种掩码（bbox / SAM） → 反投影成 3D 点集
      → 估计 3D 包围盒 → 与 GT 框比：中心误差 / 3D IoU / 体积比
```

为什么用 GT 框做区域而不是检测框
--------------------------------
因为要**隔离变量**：这里要比较的是"掩码方式"，不是"检测质量"。
用 GT 区域 → 两种掩码拿到的是**同一个目标**，差异全部来自掩码本身。

预期结论（也是面试要讲的）
------------------------
bbox 会把框内的地面/墙/背景点一起反投影，导致：
- 3D 包围盒**被撑大** → 与 GT 的体积比 > 1；
- 中心**被拉偏** → 中心误差变大；
- 3D IoU 下降。

用法::

    python scripts/15_ablate_mask_bbox_3d.py --scenes 3
    python scripts/15_ablate_mask_bbox_3d.py --scenes 3 --max-boxes 12 --output runs/mask_ablation_3d.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.io import ensure_dir                        # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity     # noqa: E402

log = get_logger("ablate_mask3d")

EVAL_CLASSES = {
    "bed", "table", "sofa", "chair", "toilet", "desk", "cabinet",
    "nightstand", "bookshelf", "bathtub", "monitor", "box", "door",
}


def rotz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def gt_regions(cfg, args):
    """从 SUN RGB-D 取出 (帧, 2D区域, GT 3D框, 标签) 列表。"""
    from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene
    from roboground.types import Detection2D

    index = load_scene_index(
        str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz"))
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))

    samples = []
    used = []
    for pos in np.argsort(-counts)[:4000]:
        i = int(pos)
        if len(used) >= args.scenes:
            break
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = [str(x) for x in np.asarray(index["label_flat"][off0:off1]).ravel()]
        if len({l.lower() for l in labels} & EVAL_CLASSES) < 2:
            continue
        scene = load_sunrgbd_scene(index, i, max_depth=8.0, resize=(args.width, args.height))
        if scene is None or scene.boxes_3d.shape[0] == 0:
            continue

        frame = scene.to_frame()
        valid_d = (frame.depth_m > 0.1) & (frame.depth_m < 8.0)
        h, w = frame.shape

        entries = []
        for k in range(scene.boxes_3d.shape[0]):
            label = str(scene.labels[k]).lower()
            if label not in EVAL_CLASSES:
                continue
            box = np.asarray(scene.boxes_3d[k], dtype=np.float64)
            # 8 角点投影 → 2D AABB
            half = np.abs(box[3:6]) / 2.0
            signs = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=np.float64)
            corners = (signs * half) @ rotz(float(box[6]) if box.size > 6 else 0.0).T + box[:3]
            cam = frame.pose.world_to_cam(corners)
            z = cam[:, 2]
            front = z > 0.3
            if front.sum() < 3:
                continue
            u = cam[front, 0] * frame.intrinsics.fx / z[front] + frame.intrinsics.cx
            v = cam[front, 1] * frame.intrinsics.fy / z[front] + frame.intrinsics.cy
            x1, y1 = max(0.0, float(u.min())), max(0.0, float(v.min()))
            x2, y2 = min(w - 1.0, float(u.max())), min(h - 1.0, float(v.max()))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            xi1, xi2 = int(x1), int(np.ceil(x2))
            yi1, yi2 = int(y1), int(np.ceil(y2))
            if int(valid_d[yi1:yi2, xi1:xi2].sum()) < args.min_valid_px:
                continue
            entries.append((Detection2D(label=label, score=1.0,
                                        bbox=np.array([x1, y1, x2, y2])), box))

        if len(entries) >= 2:
            samples.append((frame, entries))
            used.append({"scene": i, "n": len(entries),
                         "labels": sorted({e[0].label for e in entries})})

    if not samples:
        raise RuntimeError("没有收集到 GT 区域样本")
    log.info(f"使用场景 {[u['scene'] for u in used]}")
    return samples, used


def estimate_box(points: np.ndarray, percentile: float = 2.0) -> Optional[np.ndarray]:
    """从点集估一个轴对齐 3D 框 `[cx,cy,cz,dx,dy,dz,yaw=0]`。"""
    if points.shape[0] < 5:
        return None
    lo = np.percentile(points, percentile, axis=0)
    hi = np.percentile(points, 100.0 - percentile, axis=0)
    size = np.clip(hi - lo, 1e-3, None)
    center = (lo + hi) / 2.0
    return np.concatenate([center, size, [0.0]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--min-valid-px", type=int, default=200)
    ap.add_argument("--max-boxes", type=int, default=10, help="每帧最多给 SAM 多少个框")
    ap.add_argument("--sam-model", default="facebook/sam-vit-base")
    ap.add_argument("--output", default="runs/mask_ablation_3d.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    from roboground import load_config
    from roboground.eval.metrics import box3d_iou
    from roboground.geometry.projection import backproject_detection
    from roboground.perception import build_segmenter

    cfg = load_config("configs/perception_openvocab.yaml")
    cfg.set("perception.segmenter_kwargs.model_id", args.sam_model)

    samples, meta = gt_regions(cfg, args)
    n_total = sum(len(e) for _, e in samples)
    log.info(f"共 {n_total} 个 GT 目标参与对比")

    segmenter = build_segmenter(cfg, name="sam")
    segmenter.warmup()

    # 两种掩码下，每个目标的指标
    rows: Dict[str, List[Dict[str, float]]] = {"bbox": [], "sam": []}
    sam_ms_total = 0.0
    sam_boxes = 0

    for frame, entries in samples:
        dets = [e[0] for e in entries]
        boxes_gt = [e[1] for e in entries]

        # ---- bbox 掩码 ----
        for det, gt in zip(dets, boxes_gt):
            obs = backproject_detection(det, frame, min_depth=0.1, max_depth=8.0)
            est = estimate_box(obs.points_world)
            if est is None:
                continue
            rows["bbox"].append(_metrics(est, gt, box3d_iou))

        # ---- SAM 掩码（只对前 N 个框做精细分割）----
        t0 = time.perf_counter()
        seg_dets = segmenter.segment(frame, dets[: args.max_boxes])
        sam_ms_total += (time.perf_counter() - t0) * 1000.0
        sam_boxes += min(len(dets), args.max_boxes)

        for det, gt in zip(seg_dets[: args.max_boxes], boxes_gt[: args.max_boxes]):
            if det.mask is None:
                continue
            obs = backproject_detection(det, frame, min_depth=0.1, max_depth=8.0)
            est = estimate_box(obs.points_world)
            if est is None:
                continue
            rows["sam"].append(_metrics(est, gt, box3d_iou))

    # ---- 汇总 ----
    print()
    header = (f"{'掩码方式':<10}{'n':>5}{'中心误差中位(m)':>17}{'中心误差均值(m)':>17}"
              f"{'3D IoU 均值':>13}{'体积比(估/GT)':>15}{'≤0.25m':>9}")
    print(header)
    print("-" * len(header))

    summary: Dict[str, Any] = {}
    for name in ("bbox", "sam"):
        rs = rows[name]
        if not rs:
            continue
        ce = np.array([r["center_err"] for r in rs])
        iou = np.array([r["iou"] for r in rs])
        vr = np.array([r["vol_ratio"] for r in rs])
        s = {
            "n": len(rs),
            "center_median_m": float(np.median(ce)),
            "center_mean_m": float(ce.mean()),
            "iou_mean": float(iou.mean()),
            "iou_median": float(np.median(iou)),
            "vol_ratio_mean": float(vr.mean()),
            "within_0.25m": float((ce <= 0.25).mean()),
            "within_0.50m": float((ce <= 0.50).mean()),
        }
        summary[name] = s
        print(f"{name:<10}{s['n']:>5}{s['center_median_m']:>17.3f}{s['center_mean_m']:>17.3f}"
              f"{s['iou_mean']:>13.3f}{s['vol_ratio_mean']:>15.2f}{s['within_0.25m']:>9.1%}")

    print()
    if "bbox" in summary and "sam" in summary:
        b, s = summary["bbox"], summary["sam"]
        d_center = b["center_median_m"] - s["center_median_m"]
        d_iou = s["iou_mean"] - b["iou_mean"]
        d_vol = b["vol_ratio_mean"] - s["vol_ratio_mean"]
        print("=" * 80)
        print("结论")
        print("=" * 80)
        print(f"  中心定位误差中位数：bbox {b['center_median_m']:.3f}m → SAM {s['center_median_m']:.3f}m "
              f"（{'降低' if d_center > 0 else '升高'} {abs(d_center) * 100:.1f} cm）")
        print(f"  3D IoU 均值       ：bbox {b['iou_mean']:.3f} → SAM {s['iou_mean']:.3f} "
              f"（{'+' if d_iou >= 0 else ''}{d_iou:.3f}）")
        print(f"  体积比(估/GT)     ：bbox {b['vol_ratio_mean']:.2f}× → SAM {s['vol_ratio_mean']:.2f}× "
              f"（越接近 1 越好，{'-' if d_vol > 0 else '+'}{abs(d_vol):.2f}）")
        if b["vol_ratio_mean"] > 1.2:
            print()
            print(f"  → bbox 掩码把 3D 包围盒撑到 GT 的 {b['vol_ratio_mean']:.2f} 倍，"
                  f"说明框内的背景点（地面/墙）被一起反投影进来了。")
            print(f"    SAM 把它压到 {s['vol_ratio_mean']:.2f} 倍 —— 这就是「该用精细掩码」的量化证据。")

    if sam_boxes:
        print()
        print(f"  SAM 耗时：{sam_ms_total / sam_boxes:.1f} ms/框（{sam_boxes} 个框）")

    if args.output:
        ensure_dir(Path(args.output).parent)
        Path(args.output).write_text(
            json.dumps({"summary": summary, "scenes": meta,
                        "sam_ms_per_box": round(sam_ms_total / max(sam_boxes, 1), 1)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        log.ok(f"结果已保存：{args.output}")
    return 0


def _metrics(est: np.ndarray, gt: np.ndarray, iou_fn) -> Dict[str, float]:
    ce = float(np.linalg.norm(est[:3] - gt[:3]))
    iou = float(iou_fn(est, gt))
    vol_est = float(np.prod(np.abs(est[3:6])))
    vol_gt = float(np.prod(np.abs(gt[3:6])))
    return {"center_err": ce, "iou": iou,
            "vol_ratio": vol_est / max(vol_gt, 1e-9)}


if __name__ == "__main__":
    raise SystemExit(main())
