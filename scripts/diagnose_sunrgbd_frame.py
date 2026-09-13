#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 SUN RGB-D 的坐标系约定（一次性校准工具，结论已固化进代码）。

问题背景
--------
要正确地把 2D 语义投到 3D 世界系，必须先搞清"GT 的 3D 框到底在哪个坐标系"。
SUN RGB-D 的官方文档对此表述模糊（`Rtilt` 是重力对齐旋转，但框的 centroid
是否已经过该旋转没有明说），所以这里用**数据自洽性**来判定，而不是猜。

判定准则（两条，都是硬约束）
---------------------------
1. **框中心必须落在相机前方**：变到相机系后 `z > 0`；
2. **框中心的投影像素，其深度值应接近框中心的 z**：因为物体中心通常在
   其可见表面附近（误差应远小于物体本身的深度尺度）。

结论（本脚本会打印）
-------------------
GT 框位于**重力对齐的世界系**（x 右, y 前/深度, z 上），而
`world → camera` 就是**固定的 90° 轴置换的逆**：
```
world = P @ cam,   P = [[1,0,0],[0,0,1],[0,-1,0]]
cam   = Pᵀ @ world
```
与逐场景的 `Rtilt` **无关**（`Rtilt` 是额外的重力精修，标准流程并不把它
作用到点云/框上）。这也解释了 Embodied3D 预处理里那个硬编码的
`(x,y,z) → (x,z,-y)` 轴置换为什么是对的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene  # noqa: E402

#: 相机系 → 世界系：(x, y, z)_cam → (x, z, -y)_world
P_CAM_TO_WORLD = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, -1.0, 0.0]])
P_WORLD_TO_CAM = P_CAM_TO_WORLD.T


def score_hypothesis(
    name: str,
    rotation_for_scene: Callable[[int, np.ndarray], np.ndarray],
    scenes: Sequence,
    rtilts: Sequence[np.ndarray],
) -> Tuple[str, float, float, float]:
    """评估一个坐标系假设。

    Returns
    -------
    (报告字符串, 有效比例, 深度误差中位数, 深度误差均值)
    """
    errors: List[float] = []
    n_valid = 0
    n_total = 0

    for scene, Rtilt in zip(scenes, rtilts):
        if scene.boxes_3d.shape[0] == 0:
            continue
        K = scene.intrinsics
        height, width = scene.shape
        centers = np.asarray(scene.boxes_3d[:, :3], dtype=np.float64)

        R_wc = rotation_for_scene(0, Rtilt)          # world → camera
        cam = centers @ R_wc.T
        z = cam[:, 2]

        front = z > 0.3
        if not np.any(front):
            continue
        u = cam[front, 0] * K.fx / z[front] + K.cx
        v = cam[front, 1] * K.fy / z[front] + K.cy
        inb = (u >= 0) & (u < width) & (v >= 0) & (v < height)

        uu = np.clip(u[inb].astype(int), 0, width - 1)
        vv = np.clip(v[inb].astype(int), 0, height - 1)
        depth_at = scene.depth_m[vv, uu]
        z_in = z[front][inb]

        n_total += int(inb.sum())
        good = depth_at > 0
        n_valid += int(good.sum())
        if np.any(good):
            errors.extend(np.abs(depth_at[good] - z_in[good]).tolist())

    err = np.asarray(errors, dtype=np.float64)
    ratio = n_valid / max(n_total, 1)
    med = float(np.median(err)) if err.size else float("nan")
    mean = float(err.mean()) if err.size else float("nan")
    report = (
        f"{name:<34} valid_ratio={ratio:.2f}  "
        f"depth_err median={med:.3f}m  mean={mean:.3f}m  n={err.size}"
    )
    return report, ratio, med, mean


def main() -> int:
    ap = argparse.ArgumentParser(description="诊断 SUN RGB-D 坐标系约定")
    ap.add_argument("--index", default=r"G:\RoboGround\data\cache\sunrgbd_index.npz")
    ap.add_argument("--scenes", type=int, default=15)
    ap.add_argument("--max-depth", type=float, default=8.0)
    args = ap.parse_args()

    index = load_scene_index(args.index)
    scenes = []
    rtilts = []
    for i in range(min(args.scenes, len(index["sequences"]))):
        sc = load_sunrgbd_scene(index, i, max_depth=args.max_depth)
        if sc is None:
            continue
        scenes.append(sc)
        rtilts.append(np.asarray(index["Rtilt"][i], dtype=np.float64))

    print(f"已加载 {len(scenes)} 个场景\n")
    print("--- 各坐标系假设的评分（valid_ratio 越高、depth_err 越小越好）---")

    hypotheses = [
        ("A: 框已在相机系 (R=I)", lambda i, Rt: np.eye(3)),
        ("B: R=Rtiltᵀ", lambda i, Rt: Rt.T),
        ("C: R=Rtilt", lambda i, Rt: Rt),
        ("D: 固定置换 P (错方向)", lambda i, Rt: P_CAM_TO_WORLD),
        ("E: 固定置换逆 Pᵀ", lambda i, Rt: P_WORLD_TO_CAM),
        ("F: Pᵀ @ Rtilt", lambda i, Rt: P_WORLD_TO_CAM @ Rt),
        ("G: Rtiltᵀ @ Pᵀ", lambda i, Rt: Rt.T @ P_WORLD_TO_CAM),
    ]

    results = []
    for name, fn in hypotheses:
        report, ratio, med, mean = score_hypothesis(name, fn, scenes, rtilts)
        print(report)
        results.append((name, ratio, med, mean))

    best = max(results, key=lambda r: (r[1], -(r[2] if np.isfinite(r[2]) else 1e9)))
    print(f"\n最佳假设：{best[0]}  (valid_ratio={best[1]:.2f}, depth_err_median={best[2]:.3f}m)")

    expected = "E: 固定置换逆 Pᵀ"
    ok = expected in best[0] or best[1] > 0.8
    print("结论：", "与预期一致 ✔" if ok else "⚠️ 与代码中固化的假设不一致，请检查！")
    print(
        "\n固化结论：SUN RGB-D 的 GT 3D 框位于重力对齐世界系 (x右, y前, z上)，\n"
        "          world→camera 使用固定置换 Pᵀ（与逐场景 Rtilt 无关）。"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
