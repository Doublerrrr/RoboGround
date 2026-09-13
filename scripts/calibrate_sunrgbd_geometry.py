#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SUN RGB-D 框几何完整标定（坐标系 × 朝向表示 × 尺寸约定）。

为什么要做三维标定
----------------
上一版只标定了**坐标系**，但框的几何还有两个自由度没定：
1. **朝向**：用完整 `basis`（3×3 旋转）还是只取绕 z 的 yaw？
   SUN RGB-D 的框可能带倾斜，用 yaw 近似会引入随物体尺寸放大的误差。
2. **尺寸**：`coeffs` 是**半边长**还是**全边长**？

三者不一起定下来，闭环误差里就分不清"坐标系错了"还是"尺寸差 2 倍"。
本脚本做三维网格搜索，用**角点闭环误差中位数**择优。

判据
----
```
GT 角点 --R_wc--> 相机系 --投影--> (u,v) --读深度 d--> 反投影 --> 相机系
        --R_wcᵀ--> 世界系 --与 GT 角点比较--> 误差
```
坐标系正确时误差应显著小于物体尺度；坐标系错误时误差通常是米级。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.synthetic import rotz  # noqa: E402

P = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0],
              [0.0, -1.0, 0.0]])          # cam → world（固定轴置换）

_SIGNS = np.array([list(s) for s in itertools.product((-1, 1), repeat=3)], dtype=np.float64)


# --------------------------------------------------------------------------
# 从 .mat 直接读取原始 GT（不经过我们自己的索引，避免索引阶段的假设污染标定）
# --------------------------------------------------------------------------
def load_raw_gt(meta_path: str, limit: int) -> List[Dict]:
    from scipy.io import loadmat  # noqa: PLC0415

    mat = loadmat(meta_path, squeeze_me=False)
    items = np.asarray(mat["SUNRGBDMeta"]).ravel()[:limit]

    def _s(obj, name):
        try:
            v = np.asarray(obj[name]).ravel()
            return str(v[0]).strip() if v.size else ""
        except Exception:
            return ""

    out: List[Dict] = []
    for it in items:
        gt = it["groundtruth3DBB"] if "groundtruth3DBB" in it.dtype.names else None
        boxes = []
        if gt is not None and np.asarray(gt).size > 0:
            for o in np.atleast_1d(gt).ravel():
                try:
                    centroid = np.asarray(o["centroid"], dtype=np.float64).ravel()[:3]
                    coeffs = np.asarray(o["coeffs"], dtype=np.float64).ravel()[:3]
                    basis = np.asarray(o["basis"], dtype=np.float64).reshape(3, 3)
                except Exception:
                    continue
                boxes.append((centroid, coeffs, basis, _s(o, "classname")))
        out.append({
            "seq": _s(it, "sequenceName"),
            "depthname": _s(it, "depthname"),
            "rgbname": _s(it, "rgbname"),
            "K": np.asarray(it["K"], dtype=np.float64).reshape(3, 3),
            "Rtilt": np.asarray(it["Rtilt"], dtype=np.float64).reshape(3, 3),
            "boxes": boxes,
        })
    return out


def load_depth(root: str, seq: str, depthname: str):
    from PIL import Image  # noqa: PLC0415

    path = Path(root) / seq / "depth" / depthname
    if not path.exists():
        return None
    try:
        arr = np.asarray(Image.open(path))
    except Exception:
        return None
    if arr.ndim != 2:
        return None
    d = arr.astype(np.float32) / 10000.0
    d[(d < 0.2) | (d > 8.0)] = 0.0
    return d


# --------------------------------------------------------------------------
# 角点生成：两种朝向表示 × 两种尺寸约定
# --------------------------------------------------------------------------
def corners_variant(
    centroid: np.ndarray,
    coeffs: np.ndarray,
    basis: np.ndarray,
    *,
    orientation: str,
    size_mode: str,
) -> np.ndarray:
    """按指定约定生成 8 个角点。

    orientation: "basis" 用完整 basis；"yaw" 只取绕 z 的 yaw。
    size_mode:   "half" 用 coeffs 作半边长；"full" 把 coeffs 当全边长。
    """
    half = coeffs if size_mode == "half" else coeffs / 2.0
    local = _SIGNS * half
    if orientation == "basis":
        R = basis
    else:
        yaw = float(np.arctan2(basis[1, 0], basis[0, 0]))
        R = rotz(yaw)
    return local @ R.T + centroid


def evaluate(
    gt_items: Sequence[Dict],
    root: str,
    rotation_fn,
    *,
    orientation: str,
    size_mode: str,
) -> Dict[str, float]:
    loop_errors: List[float] = []
    n_corners = 0
    n_hit = 0

    for item in gt_items:
        depth = load_depth(root, item["seq"], item["depthname"])
        if depth is None:
            continue
        height, width = depth.shape
        K = item["K"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R_wc = rotation_fn(item["Rtilt"])

        for centroid, coeffs, basis, _cls in item["boxes"]:
            if np.any(coeffs <= 0):
                continue
            cw = corners_variant(centroid, coeffs, basis,
                                 orientation=orientation, size_mode=size_mode)
            n_corners += cw.shape[0]

            cam = cw @ R_wc.T
            z = cam[:, 2]
            front = z > 0.2
            if not np.any(front):
                continue

            u = np.full(cw.shape[0], np.nan)
            v = np.full(cw.shape[0], np.nan)
            u[front] = cam[front, 0] * fx / z[front] + cx
            v[front] = cam[front, 1] * fy / z[front] + cy

            inb = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not np.any(inb):
                continue

            uu = np.clip(np.round(u[inb]).astype(int), 0, width - 1)
            vv = np.clip(np.round(v[inb]).astype(int), 0, height - 1)
            d = depth[vv, uu]
            good = d > 0.2
            n_hit += int(good.sum())
            if not np.any(good):
                continue

            u_g, v_g, d_g = uu[good], vv[good], d[good].astype(np.float64)
            back_cam = np.stack([
                (u_g - cx) / fx * d_g,
                (v_g - cy) / fy * d_g,
                d_g,
            ], axis=1)
            back_world = back_cam @ R_wc
            err = np.linalg.norm(back_world - cw[inb][good], axis=1)
            loop_errors.extend(err.tolist())

    err = np.asarray(loop_errors, dtype=np.float64)
    return {
        "valid_ratio": n_hit / max(n_corners, 1),
        "median": float(np.median(err)) if err.size else float("nan"),
        "mean": float(err.mean()) if err.size else float("nan"),
        "n": int(err.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="SUN RGB-D 框几何三维标定")
    # 注意：root 是数据集根目录（其下第一层就是 SUNRGBD/），
    # 因为 meta 里的 sequenceName 已经带了 "SUNRGBD/..." 前缀。
    ap.add_argument("--root", default=r"G:\sunrgbd_raw")
    ap.add_argument("--meta", default=r"G:\sunrgbd_raw\SUNRGBDtoolbox\Metadata\SUNRGBDMeta.mat")
    ap.add_argument("--limit", type=int, default=60, help="读取多少个场景的原始 GT")
    args = ap.parse_args()

    print(f"读取原始 GT：{args.meta}（前 {args.limit} 个场景）")
    gt_items = load_raw_gt(args.meta, args.limit)
    with_boxes = sum(1 for it in gt_items if it["boxes"])
    print(f"  含框场景 {with_boxes}/{len(gt_items)}，"
          f"框总数 {sum(len(it['boxes']) for it in gt_items)}\n")

    rotations = {
        "I":        lambda Rt: np.eye(3),
        "Pᵀ":       lambda Rt: P.T,
        "Pᵀ@Rtiltᵀ": lambda Rt: P.T @ Rt.T,
        "Pᵀ@Rtilt": lambda Rt: P.T @ Rt,
        "Rtiltᵀ@Pᵀ": lambda Rt: Rt.T @ P.T,
        "Rtilt@Pᵀ": lambda Rt: Rt @ P.T,
    }

    print(f"{'坐标系':<12} {'朝向':<6} {'尺寸':<5} {'valid':>6} {'median':>9} {'mean':>9} {'n':>6}")
    print("-" * 66)

    rows = []
    for rname, rfn in rotations.items():
        for orientation in ("basis", "yaw"):
            for size_mode in ("half", "full"):
                r = evaluate(gt_items, args.root, rfn,
                             orientation=orientation, size_mode=size_mode)
                rows.append((rname, orientation, size_mode, r))
                print(
                    f"{rname:<12} {orientation:<6} {size_mode:<5} "
                    f"{r['valid_ratio']:>6.2f} {r['median']:>9.3f} "
                    f"{r['mean']:>9.3f} {r['n']:>6d}"
                )

    cands = [row for row in rows if row[3]["n"] >= 50 and np.isfinite(row[3]["median"])]
    if not cands:
        print("\n⚠️ 样本不足，无法定标")
        return 1

    cands.sort(key=lambda row: row[3]["median"])
    rname, orientation, size_mode, best = cands[0]
    print(
        f"\n>>> 最佳组合：坐标系={rname}  朝向={orientation}  尺寸={size_mode}\n"
        f"    median 闭环误差 = {best['median']:.3f} m（n={best['n']}）"
    )
    print("\n前三名：")
    for rname, orientation, size_mode, r in cands[:3]:
        print(f"   {rname:<12} {orientation:<6} {size_mode:<5} median={r['median']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
