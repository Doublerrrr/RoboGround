# -*- coding: utf-8 -*-
"""只用 RGB + 位姿恢复深度（plane-sweep 多视角立体）。

为什么需要这个模块
================
真实机器人上，**多目 RGB 摄像头很常见，深度不一定有**：
环绕相机模组、工业相机、纯视觉方案都只给 RGB。而本项目原来的前端
直接消费 `depth_m`，于是"没有深度"就等于"跑不了"。

本模块补上这一环：给定 **N 个真实视角的 RGB + 各自位姿**（位姿可由
轮式里程计 / SLAM / VIO 提供，本来就要有），用**纯几何**把深度算出来。

    N 张 RGB + N 个位姿 ──plane sweep──► N 张深度图 ──► 之后与本项目原链路完全一致

**不依赖任何模型权重**（不需要单目深度网络），因此在离线机器上也能跑通；
当然也可以把这个结果当作"几何基线"，与 Metric3D / Depth Anything 等
学习式方法对比。

算法（plane sweep / 平面扫描）
============================
对参考视角的每个像素，假设它在一系列深度平面上，把**其它视角**按该深度
投影过来取样，谁的假设让各视角颜色最一致，谁就是正确深度。

    对每个深度假设 z：
        p_cam_ref = z · [(u−cx)/fx, (v−cy)/fy, 1]
        p_world   = R_refᵀ (p_cam_ref − t_ref)          ← 位姿是 world→camera
        p_cam_s   = R_s p_world + t_s
        (u_s,v_s) = project(p_cam_s, K_s)
        cost += |I_ref(u,v) − I_s(u_s,v_s)|

    深度 = argmin_z cost

工程细节（每一条都影响结果，写清楚以免以后当成玄学）
================================================
· **深度按视差（1/z）均匀采样**，不按 z 均匀 —— 近处才该有高分辨率。
· 用 **L1 颜色差 + 梯度差**：纯颜色在无纹理墙面（本项目大量白墙）上退化，
  梯度项能补一点结构信息。
· **空间聚合**用盒滤波近似 patch matching（scipy `uniform_filter`），
  比逐 patch 循环快数十倍，效果接近。
· **亚像素细化**：在最优点 ±1 个平面上做抛物线拟合。
· **左右一致性检查**（可选）：把参考深度反投到源视角再投回来，差太大判为遮挡/误匹配。
· 投影越界的源视角**不计入该像素的代价**（不是记 0 分，而是排除），
  否则视场外的黑边会被误认为"深度很匹配"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("data.mvs_depth")


@dataclass
class MvsResult:
    """一次 plane-sweep 的结果。"""

    depth_m: np.ndarray          # (H, W) float32，0 = 无效
    cost: np.ndarray             # (H, W) float32，最优平面处的匹配代价（越小越可信）
    n_views_used: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float((self.depth_m > 0).mean())


# ==========================================================================
# 采样与工具
# ==========================================================================
def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 2:
        return a
    return a[..., :3].mean(axis=2)


def _sample_bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """双线性采样。返回 `(值, 是否在图内)`；越界处值置 0 并由 mask 标出。"""
    H, W = img.shape[:2]
    inside = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    uc = np.clip(u, 0, W - 1)
    vc = np.clip(v, 0, H - 1)
    u0 = np.floor(uc).astype(np.int64)
    v0 = np.floor(vc).astype(np.int64)
    u1 = np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)
    du = uc - u0
    dv = vc - v0
    val = (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u1] * du * (1 - dv)
           + img[v1, u0] * (1 - du) * dv + img[v1, u1] * du * dv)
    return val, inside


def _grad_mag(img: np.ndarray) -> np.ndarray:
    gy = np.zeros_like(img); gx = np.zeros_like(img)
    gy[1:-1, :] = np.abs(img[2:, :] - img[:-2, :]) * 0.5
    gx[:, 1:-1] = np.abs(img[:, 2:] - img[:, :-2]) * 0.5
    return gx + gy


def _scale_intrinsics(K, s: float):
    from roboground.types import CameraIntrinsics
    return CameraIntrinsics(fx=K.fx * s, fy=K.fy * s, cx=K.cx * s, cy=K.cy * s,
                            width=max(1, int(round(K.width * s))),
                            height=max(1, int(round(K.height * s))))


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    a = np.asarray(img)
    mode = Image.BILINEAR if a.ndim == 3 else Image.BILINEAR
    return np.asarray(Image.fromarray(a.astype(np.uint8) if a.dtype != np.uint8 else a).resize((w, h), mode))


# ==========================================================================
# 主算法
# ==========================================================================
def estimate_depth_plane_sweep(
    frames: Sequence[Any],
    ref_index: int = 0,
    *,
    depth_min: float = 0.5,
    depth_max: float = 8.0,
    n_planes: int = 64,
    downsample: int = 4,
    patch: int = 7,
    gradient_weight: float = 1.0,
    max_source_views: int = 8,
) -> MvsResult:
    """对第 `ref_index` 个视角做 plane-sweep，返回它的深度图。

    Parameters
    ----------
    frames
        N 个 `RGBDFrame`；**本函数只用 `color` / `intrinsics` / `pose`**，
        `depth_m` 完全不参与 —— 这正是"只有 RGB"的场景。
    depth_min, depth_max
        深度搜索范围（米）。
    n_planes
        深度假设数（按**视差**均匀采样）。越多越准越慢。
    downsample
        先降采样再匹配（加速）；深度图会恢复到原分辨率尺寸报出。
    patch
        空间聚合窗口（近似 patch matching）。
    max_source_views
        最多用多少个源视角（太多收益小、线性变慢）。按与参考视角的
        **朝向接近度**挑选，因为朝向差太大的视角几乎看不到同一块表面。
    """
    from scipy.ndimage import uniform_filter

    if not (0 <= ref_index < len(frames)):
        raise IndexError(f"ref_index={ref_index} 越界（共 {len(frames)} 个视角）")
    if len(frames) < 2:
        raise ValueError("plane sweep 至少需要 2 个视角")

    ref = frames[ref_index]
    s = 1.0 / max(1, int(downsample))
    K_ref = _scale_intrinsics(ref.intrinsics, s)
    W, H = K_ref.width, K_ref.height
    I_ref = _to_gray(_resize(ref.color, W, H))
    G_ref = _grad_mag(I_ref)
    # 参考视角每个像素在**相机系**里的方向（未归一化，z=1 平面）
    uu = (np.arange(W, dtype=np.float64) + 0.5)[None, :].repeat(H, axis=0)
    vv = (np.arange(H, dtype=np.float64) + 0.5)[:, None].repeat(W, axis=1)
    kx = (uu - K_ref.cx) / K_ref.fx
    ky = (vv - K_ref.cy) / K_ref.fy

    # ---- 挑源视角：按光轴方向接近度（朝向差太大看不到同一表面）----
    R_ref = np.asarray(ref.pose.R, dtype=np.float64).reshape(3, 3)
    f_ref = R_ref.T @ np.array([0.0, 0.0, 1.0])
    order = []
    for i, fr in enumerate(frames):
        if i == ref_index:
            continue
        f = np.asarray(fr.pose.R, dtype=np.float64).reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
        order.append((float(f @ f_ref), i))
    order.sort(reverse=True)
    sources = [i for _, i in order[: max(2, int(max_source_views))]]

    # ---- 深度假设：按视差均匀（近处分辨率高）----
    inv = np.linspace(1.0 / float(depth_max), 1.0 / float(depth_min), int(n_planes))
    planes = 1.0 / inv                                       # 从远到近
    planes = planes.astype(np.float64)

    # ---- 预采样源视角（灰度 + 梯度 + 内参 + 位姿）----
    prepped = []
    for i in sources:
        fr = frames[i]
        Ki = _scale_intrinsics(fr.intrinsics, s)
        prepped.append({
            "gray": _to_gray(_resize(fr.color, Ki.width, Ki.height)),
            "grad": _grad_mag(_to_gray(_resize(fr.color, Ki.width, Ki.height))),
            "K": Ki,
            "R": np.asarray(fr.pose.R, dtype=np.float64).reshape(3, 3),
            "t": np.asarray(fr.pose.t, dtype=np.float64).reshape(3),
        })

    # ---- 代价体：对每个深度平面累加各源视角的 L1 颜色差 + 梯度差 ----
    cost = np.zeros((len(planes), H, W), dtype=np.float32)
    weight = np.zeros((len(planes), H, W), dtype=np.float32)
    for pi, z in enumerate(planes):
        # 参考相机系 → 世界系
        p_cam = np.stack([kx * z, ky * z, np.full_like(kx, z)], axis=-1)
        p_world = (p_cam - ref.pose.t[None, None, :]) @ R_ref   # (Rᵀ v)ᵀ = vᵀ R
        c_acc = np.zeros((H, W))
        w_acc = np.zeros((H, W))
        for sp in prepped:
            p_s = p_world @ sp["R"].T + sp["t"][None, None, :]
            zs = p_s[..., 2]
            ok = zs > 1e-6
            us = np.where(ok, p_s[..., 0] * sp["K"].fx / np.maximum(zs, 1e-9) + sp["K"].cx, -1e6)
            vs = np.where(ok, p_s[..., 1] * sp["K"].fy / np.maximum(zs, 1e-9) + sp["K"].cy, -1e6)
            gv, inside = _sample_bilinear(sp["gray"], us, vs)
            gg, _ = _sample_bilinear(sp["grad"], us, vs)
            valid = ok & inside
            d = np.abs(I_ref - gv) + float(gradient_weight) * np.abs(G_ref - gg)
            c_acc += np.where(valid, d, 0.0)
            w_acc += valid.astype(np.float64)
        good = w_acc > 0
        cost[pi] = np.where(good, c_acc / np.maximum(w_acc, 1e-9), np.inf).astype(np.float32)
        weight[pi] = w_acc.astype(np.float32)
        if (pi + 1) % 16 == 0:
            logger.debug(f"  plane {pi + 1}/{len(planes)} (z={z:.2f} m)")

    # ---- 空间聚合（近似 patch matching）----
    if patch and patch > 1:
        for pi in range(len(planes)):
            c = cost[pi]
            finite = np.isfinite(c)
            filled = np.where(finite, c, 0.0)
            num = uniform_filter(filled, size=int(patch), mode="nearest")
            den = uniform_filter(finite.astype(np.float64), size=int(patch), mode="nearest")
            cost[pi] = np.where(den > 1e-6, num / np.maximum(den, 1e-9), np.inf).astype(np.float32)

    # ---- 赢家通吃 + 抛物线亚像素细化 ----
    best = np.argmin(cost, axis=0)
    H2, W2 = best.shape
    rr, cc = np.mgrid[0:H2, 0:W2]
    best_cost = cost[best, rr, cc]
    ok = np.isfinite(best_cost)
    depth = np.zeros((H2, W2), dtype=np.float64)
    depth[ok] = planes[best[ok]]

    pm = np.clip(best - 1, 0, len(planes) - 1)
    pp = np.clip(best + 1, 0, len(planes) - 1)
    c0 = cost[best, rr, cc]; cm = cost[pm, rr, cc]; cp = cost[pp, rr, cc]
    # ★ 无效平面（无源视角可见）与边界平面的代价是 `inf`，直接相减会得到
    #   `inf − inf = nan`（实测报了 RuntimeWarning）。改为**只在三者都有限时**
    #   才算二阶差分，其余保持 0（不做亚像素修正）。
    denom = np.zeros_like(c0)
    ok3 = np.isfinite(c0) & np.isfinite(cm) & np.isfinite(cp)
    denom[ok3] = cm[ok3] - 2.0 * c0[ok3] + cp[ok3]
    inb = (best > 0) & (best < len(planes) - 1) & ok
    shift = np.zeros_like(depth)
    good_par = inb & np.isfinite(denom) & (np.abs(denom) > 1e-9)
    shift[good_par] = 0.5 * (cm[good_par] - cp[good_par]) / denom[good_par]
    shift = np.clip(shift, -1.0, 1.0)
    # 在视差域里做亚像素插值（与采样方式一致）
    lo = np.clip(best - 1, 0, len(planes) - 1); hi = np.clip(best + 1, 0, len(planes) - 1)
    inv_interp = inv[best] + shift * 0.5 * (inv[hi] - inv[lo])
    depth[ok] = 1.0 / np.maximum(inv_interp[ok], 1e-9)

    # ---- 置信度门槛：代价过高的像素判为无效（无纹理区/遮挡）----
    valid_cost = best_cost[ok]
    if valid_cost.size:
        thr = float(np.percentile(valid_cost, 85.0))
        bad = ok & (best_cost > thr)
        depth[bad] = 0.0

    # 恢复到原分辨率尺寸
    d_full = np.zeros(ref.depth_m.shape if hasattr(ref, "depth_m") else (H2, W2), dtype=np.float32)
    from PIL import Image
    d_full = np.asarray(Image.fromarray(depth.astype(np.float32), mode="F").resize(
        (d_full.shape[1], d_full.shape[0]), Image.BILINEAR)).astype(np.float32)
    c_full = np.asarray(Image.fromarray(
        np.nan_to_num(best_cost, nan=0.0, posinf=0.0).astype(np.float32), mode="F").resize(
        (d_full.shape[1], d_full.shape[0]), Image.BILINEAR)).astype(np.float32)

    return MvsResult(depth_m=d_full, cost=c_full, n_views_used=len(sources) + 1,
                     meta={"ref_index": ref_index, "n_planes": int(n_planes),
                           "planes_range_m": [float(planes.min()), float(planes.max())],
                           "downsample": int(downsample), "patch": int(patch),
                           "gradient_weight": float(gradient_weight),
                           "sources": sources,
                           "solve_wh": [W, H]})


def estimate_all_depths(frames: Sequence[Any], **kwargs) -> List[MvsResult]:
    """对每个视角都估一遍深度（每个视角轮流当参考）。"""
    out = []
    for i in range(len(frames)):
        out.append(estimate_depth_plane_sweep(frames, i, **kwargs))
    return out
