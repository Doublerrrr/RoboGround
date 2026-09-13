#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""坐标系统定标：用**角点重投影一致性**判定 SUN RGB-D 的框坐标系。

判据设计（为什么这个判据是决定性的）
----------------------------------
对 GT 框的 8 个角点，做一次闭环：
```
GT 角点(假设的世界系) --R_wc--> 相机系 --投影--> 像素(u,v)
      --读深度图--> d --反投影--> 相机系点 --R_wcᵀ--> 世界系点
      --与 GT 角点比较--> 误差
```
如果坐标系假设正确，这个闭环误差应当很小（只受深度噪声与遮挡影响，
通常在厘米级）；如果假设错误，误差会是米级甚至更离谱。

这个判据比"框中心 vs 表面深度"更强，因为：
- 不依赖"物体中心在表面附近"这种近似（大物体的中心离表面很远）；
- 角点是**几何上确定**的位置，闭环误差是有物理意义的量。

同时报告 `valid_ratio`（角点投影落在图内且深度有效的比例）作为辅助。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene  # noqa: E402
from roboground.data.synthetic import rotz  # noqa: E402

# 相机系 → 世界系 的固定轴置换：(x, y, z)_cam → (x, z, -y)_world
P = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0],
              [0.0, -1.0, 0.0]])


def box_corners(center: np.ndarray, size: np.ndarray, heading: float) -> np.ndarray:
    """有向盒的 8 个角点（世界系）。"""
    half = np.abs(size) / 2.0
    signs = np.array([list(s) for s in itertools.product((-1, 1), repeat=3)], dtype=np.float64)
    local = signs * half
    return local @ rotz(heading).T + center


def evaluate(
    name: str,
    rotation_for_scene: Callable[[int, np.ndarray], np.ndarray],
    scenes: Sequence,
    rtilts: Sequence[np.ndarray],
) -> Dict[str, float]:
    """返回该假设下的闭环误差统计。"""
    loop_errors: List[float] = []
    n_corners = 0
    n_hit = 0

    for scene, Rtilt in zip(scenes, rtilts):
        if scene.boxes_3d.shape[0] == 0:
            continue
        K = scene.intrinsics
        height, width = scene.shape
        R_wc = rotation_for_scene(0, Rtilt)

        for k in range(scene.boxes_3d.shape[0]):
            cx, cy, cz, dx, dy, dz = scene.boxes_3d[k, :6]
            heading = float(scene.boxes_3d[k, 6]) if scene.boxes_3d.shape[1] > 6 else 0.0
            corners_world = box_corners(
                np.array([cx, cy, cz], dtype=np.float64),
                np.array([dx, dy, dz], dtype=np.float64),
                heading,
            )
            n_corners += corners_world.shape[0]

            # 世界 → 相机
            corners_cam = corners_world @ R_wc.T
            z = corners_cam[:, 2]
            front = z > 0.2

            # 投影
            u = np.full(corners_world.shape[0], np.nan)
            v = np.full(corners_world.shape[0], np.nan)
            u[front] = corners_cam[front, 0] * K.fx / z[front] + K.cx
            v[front] = corners_cam[front, 1] * K.fy / z[front] + K.cy

            inb = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not np.any(inb):
                continue

            uu = np.clip(np.round(u[inb]).astype(int), 0, width - 1)
            vv = np.clip(np.round(v[inb]).astype(int), 0, height - 1)
            d = scene.depth_m[vv, uu]
            good = d > 0.2
            n_hit += int(good.sum())
            if not np.any(good):
                continue

            # 反投影（相机系）：p = d * [(u-cx)/fx, (v-cy)/fy, 1]
            uu_g, vv_g, d_g = uu[good], vv[good], d[good].astype(np.float64)
            back_cam = np.stack([
                (uu_g - K.cx) / K.fx * d_g,
                (vv_g - K.cy) / K.fy * d_g,
                d_g,
            ], axis=1)
            back_world = back_cam @ R_wc          # 相机→世界 = R_wcᵀ 作用在行向量上
            gt_world = corners_world[inb][good]
            err = np.linalg.norm(back_world - gt_world, axis=1)
            loop_errors.extend(err.tolist())

    err = np.asarray(loop_errors, dtype=np.float64)
    return {
        "valid_ratio": n_hit / max(n_corners, 1),
        "median": float(np.median(err)) if err.size else float("nan"),
        "mean": float(err.mean()) if err.size else float("nan"),
        "p25": float(np.percentile(err, 25)) if err.size else float("nan"),
        "n": int(err.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="用角点闭环一致性定标框坐标系")
    ap.add_argument("--index", default=r"G:\RoboGround\data\cache\sunrgbd_index.npz")
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--max-depth", type=float, default=8.0)
    args = ap.parse_args()

    index = load_scene_index(args.index)
    scenes, rtilts = [], []
    for i in range(min(args.scenes, len(index["sequences"]))):
        sc = load_sunrgbd_scene(index, i, max_depth=args.max_depth)
        if sc is None:
            continue
        scenes.append(sc)
        rtilts.append(np.asarray(index["Rtilt"][i], dtype=np.float64))
    print(f"已加载 {len(scenes)} 个场景\n")

    # 枚举所有合理组合（把 Rtilt 和固定置换以各种顺序复合）
    hypotheses: List[Tuple[str, Callable[[int, np.ndarray], np.ndarray]]] = [
        ("I            (框在相机系)", lambda i, Rt: np.eye(3)),
        ("Rtilt", lambda i, Rt: Rt),
        ("Rtiltᵀ", lambda i, Rt: Rt.T),
        ("P            (固定置换)", lambda i, Rt: P),
        ("Pᵀ", lambda i, Rt: P.T),
        ("P @ Rtilt", lambda i, Rt: P @ Rt),
        ("Pᵀ @ Rtilt", lambda i, Rt: P.T @ Rt),
        ("Rtilt @ P", lambda i, Rt: Rt @ P),
        ("Rtiltᵀ @ Pᵀ", lambda i, Rt: Rt.T @ P.T),
        ("P @ Rtiltᵀ", lambda i, Rt: P @ Rt.T),
        ("Pᵀ @ Rtiltᵀ", lambda i, Rt: P.T @ Rt.T),
        ("Rtilt @ Pᵀ", lambda i, Rt: Rt @ P.T),
        ("Rtiltᵀ @ P", lambda i, Rt: Rt.T @ P),
    ]

    print(f"{'假设':<24} {'valid':>6} {'median':>9} {'mean':>9} {'p25':>9} {'n':>6}")
    print("-" * 70)
    rows = []
    for name, fn in hypotheses:
        r = evaluate(name, fn, scenes, rtilts)
        rows.append((name, r))
        print(
            f"{name:<24} {r['valid_ratio']:>6.2f} {r['median']:>9.3f} "
            f"{r['mean']:>9.3f} {r['p25']:>9.3f} {r['n']:>6d}"
        )

    # 择优：闭环误差中位数最小（要求样本量足够）
    candidates = [(n, r) for n, r in rows if r["n"] >= 20 and np.isfinite(r["median"])]
    if not candidates:
        print("\n⚠️ 没有任何假设产生足够样本，请检查数据路径")
        return 1

    best_name, best = min(candidates, key=lambda kv: kv[1]["median"])
    print(f"\n>>> 最佳假设：{best_name}  (median 闭环误差 {best['median']:.3f} m)")
    print(
        "\n注：闭环误差不为 0 是正常的 —— GT 框角点与深度图的实际表面不完全重合"
        "（框是人标注/拟合的，深度有噪声，且有遮挡）。关键是**哪个假设误差最小**。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
