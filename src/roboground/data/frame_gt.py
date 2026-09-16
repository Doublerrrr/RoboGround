# -*- coding: utf-8 -*-
"""单帧（针孔）视角下的 GT 框投影与可见性判定。

为什么需要这个模块
==================
`pano_scene.box_to_pano` 判的是"物体在 **360° 全景**里看得见吗"。
要回答"**用一帧 vs 用 N 帧融合**，下游差多少"这类问题（全景价值消融），
必须有一个**口径完全一致**的针孔版本，否则两条臂数的不是同一件事：

    全景臂的 `visible_frac` 分母 = 包围盒表面采样点总数
    针孔臂也必须是同一个分母

如果针孔臂把"不在画面里的采样点"直接丢掉再算比例，它就会**虚高** ——
一个只露出一角的物体也能拿到 0.9，消融立刻失真。
所以这里刻意用同一套采样、同一个分母，把"画面外 / 超出量程 / 被挡住"
统统算作**没观测到**。

与全景版的唯一差别是**观测方向的定义**
=====================================
· 全景 `depth_m` 是**从光心起算的斜距** `r`，遮挡判据比 `r`；
· 针孔 `depth_m` 是**沿光轴的 z 深度**，遮挡判据比 `z`。

两者都各自与自己传感器的深度语义对齐，这一点在
`docs/2D3D-S数据集核验报告.md` 里有实测依据（同一批像素 `range/z` 最大差 2.86×，
拿 z 当 r 用会系统性错位）。量程门限则**统一用斜距** `r <= max_range_m`，
保证两条臂的信息边界一致。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.types import RGBDFrame

#: 遮挡容差（与 `pano_scene.box_to_pano` 保持同一组默认值）
DEFAULT_OCC_TOL_ABS = 0.30
DEFAULT_OCC_TOL_REL = 0.10
#: 可见性门限（与 `visible_gt` 同一个默认值）
DEFAULT_MIN_VISIBLE_FRAC = 0.25


def box_surface_points(box: np.ndarray, grid: int = 6) -> np.ndarray:
    """在轴对齐框的 6 个面上采样（只看表面，内部点没意义）。

    原先定义在 `pano_scene._box_surface_points`，为了让针孔版复用
    **同一套采样点**（同一个 `grid`、同样的面分布）而上移到公共位置。
    """
    box = np.asarray(box, dtype=np.float64).reshape(7)
    c, s = box[:3], box[3:6]
    lo, hi = c - s / 2.0, c + s / 2.0
    t = np.linspace(0.0, 1.0, int(grid))
    g = np.meshgrid(t, t, indexing="ij")
    pts = []
    for axis in range(3):                       # 每个轴上的两个面
        for val in (lo[axis], hi[axis]):
            p = np.empty((g[0].size, 3))
            p[:, axis] = val
            others = [a for a in range(3) if a != axis]
            p[:, others[0]] = lo[others[0]] + g[0].ravel() * (hi[others[0]] - lo[others[0]])
            p[:, others[1]] = lo[others[1]] + g[1].ravel() * (hi[others[1]] - lo[others[1]])
            pts.append(p)
    return np.concatenate(pts, axis=0)


def box_to_frame(frame: RGBDFrame, box: np.ndarray, *,
                 grid: int = 6,
                 occ_tol_abs: float = DEFAULT_OCC_TOL_ABS,
                 occ_tol_rel: float = DEFAULT_OCC_TOL_REL,
                 min_visible_frac: float = DEFAULT_MIN_VISIBLE_FRAC,
                 max_range_m: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """把一个世界系轴对齐 GT 框投到**针孔帧**上，并判定可见性。

    返回 `dict`；相机位于盒内时返回 `None`（不存在有意义的单框）。
    字段与 `pano_scene.box_to_pano` 对齐，便于两条臂直接比：

    · `uv`：`(u0, v0, u1, v1)`，**裁剪到画面内**（针孔没有接缝，
      所以永远只有一段；跨出边界的那部分采样点记为未观测）
    · `uv_parts`：长度 0 或 1 的列表，与全景版同为"落在图内的框"的语义
    · `range_m`：框表面点到光心的**中位斜距**
    · `z_median_m`：框表面点的中位 z 深度（诊断用）
    · `visible_frac`：采样点里"在画面内 + 有有效深度 + 没被更近的东西挡住"
      的比例，**分母是全部采样点**
    · `covered_frac`：只算"在画面内且有有效深度"的比例
    · `in_frustum_frac`：只算"在画面内"的比例
    · `visible`：`visible_frac >= min_visible_frac`

    ⚠️ 和全景一样，一个盒子总有约一半表面**背对相机**，
    所以"完全看得见"也只有 ~50%，不要把 `visible_frac` 读成
    "看到了物体的百分之多少"。
    """
    pts = box_surface_points(box, grid=grid)
    C = np.asarray(frame.pose.camera_center(), dtype=np.float64).reshape(3)
    d = pts - C[None, :]
    r = np.linalg.norm(d, axis=1)
    ok = r > 1e-6
    if ok.sum() < 4:
        return None
    pts, r = pts[ok], r[ok]

    # 相机在盒子里：盒子包住了光心，单框没有意义
    b = np.asarray(box, dtype=np.float64).reshape(7)
    lo, hi = b[:3] - b[3:6] / 2.0, b[:3] + b[3:6] / 2.0
    if bool(np.all(C >= lo) and np.all(C <= hi)):
        return None

    pc = np.asarray(frame.pose.world_to_cam(pts), dtype=np.float64)   # (N,3)
    z = pc[:, 2]
    K = frame.intrinsics
    H, W = frame.depth_m.shape[:2]

    front = z > 1e-6
    u = np.full(r.shape, -1.0)
    v = np.full(r.shape, -1.0)
    u[front] = K.fx * pc[front, 0] / z[front] + K.cx
    v[front] = K.fy * pc[front, 1] / z[front] + K.cy
    in_img = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if max_range_m is not None:
        # 统一量程口径：全景臂在融合时按**斜距**裁到 max_depth，
        # 针孔臂这里也必须按斜距裁，否则针孔白拿一段全景没有的信息。
        in_img &= (r <= float(max_range_m))

    obs = np.zeros(r.shape, dtype=np.float64)
    ui = np.where(in_img, u, 0.0).astype(np.int64)
    vi = np.where(in_img, v, 0.0).astype(np.int64)
    np.clip(ui, 0, W - 1, out=ui)
    np.clip(vi, 0, H - 1, out=vi)
    obs[in_img] = frame.depth_m[vi[in_img], ui[in_img]].astype(np.float64)

    covered = in_img & (obs > 0)
    tol = np.maximum(occ_tol_abs, occ_tol_rel * np.abs(z))
    not_occluded = obs >= (z - tol)
    vis = covered & not_occluded

    u_in = u[in_img]
    v_in = v[in_img]
    if u_in.size:
        u0 = int(np.clip(np.floor(u_in.min()), 0, W - 1))
        u1 = int(np.clip(np.floor(u_in.max()), 0, W - 1))
        v0 = int(np.clip(np.floor(v_in.min()), 0, H - 1))
        v1 = int(np.clip(np.floor(v_in.max()), 0, H - 1))
        parts: List[Tuple[int, int, int, int]] = (
            [(u0, v0, u1, v1)] if u1 >= u0 and v1 >= v0 else [])
    else:
        parts = []

    return {
        "uv": parts[0] if parts else (0, 0, 0, 0),
        "uv_parts": parts,
        "uv_wrapped": False,
        "range_m": float(np.median(r)),
        "z_median_m": float(np.median(z[front])) if bool(front.any()) else float("nan"),
        "visible_frac": float(vis.mean()),
        "covered_frac": float(covered.mean()),
        "in_frustum_frac": float(in_img.mean()),
        "n_samples": int(r.size),
        "visible": bool(vis.mean() >= float(min_visible_frac)),
    }


def visible_gt_in_frame(frame: RGBDFrame,
                        boxes: np.ndarray,
                        labels: Sequence[str], *,
                        min_visible_frac: float = DEFAULT_MIN_VISIBLE_FRAC,
                        max_range_m: Optional[float] = None,
                        grid: int = 6,
                        occ_tol_abs: float = DEFAULT_OCC_TOL_ABS,
                        occ_tol_rel: float = DEFAULT_OCC_TOL_REL
                        ) -> List[Dict[str, Any]]:
    """列出**这一帧里真的看得见**的 GT 物体（带投影框）。

    返回条目与 `pano_scene.visible_gt` 同构：`box` / `label` / `index` /
    `uv` / `uv_parts` / `visible_frac` / `range_m`，外加 `z_median_m`。
    `index` 是它在传入 `boxes` 里的下标 —— **跨臂对齐 GT 就靠它**。
    """
    boxes = np.asarray(boxes)
    out: List[Dict[str, Any]] = []
    if boxes.size == 0:
        return out
    boxes = boxes.reshape(-1, boxes.shape[-1] if boxes.ndim > 1 else 7)
    for i in range(boxes.shape[0]):
        info = box_to_frame(frame, boxes[i], grid=grid,
                            occ_tol_abs=occ_tol_abs, occ_tol_rel=occ_tol_rel,
                            min_visible_frac=min_visible_frac,
                            max_range_m=max_range_m)
        if info is None or info["visible_frac"] < float(min_visible_frac):
            continue
        out.append({"box": boxes[i], "label": str(labels[i]), "index": int(i),
                    **info})
    out.sort(key=lambda o: o["range_m"])
    return out
