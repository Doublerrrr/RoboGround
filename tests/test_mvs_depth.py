# -*- coding: utf-8 -*-
"""plane-sweep 多视角立体的测试（**纯合成，不需要数据集**）。

这些测试要钉住两件事，两件都曾经/可能被误解：

1. **实现是对的** —— 有基线时能把已知深度的平面恢复到厘米级；
2. **基线是硬前提** —— 相机光心重合时，多视角立体在**几何上不可能**，
   不是"效果差一点"，而是代价函数**恒为常数**。本项目用的 2D-3D-S `/data`
   恰好是**共光心**采集（光心离散 2.7e-06 m），所以那份数据**不能**用来
   验证"只有 RGB 也能出深度"这条路线。这一点必须写成测试，否则以后
   一定会有人再试一遍、再困惑一次。

测试用**解析构造**的纹理平面：纹理是世界坐标的函数，与 `mvs_depth.py`
的投影实现完全独立，所以"恢复出正确深度"是真检验，不是自证。
"""
from __future__ import annotations

import numpy as np
import pytest

from roboground.data.mvs_depth import estimate_depth_plane_sweep
from roboground.geometry.camera import look_at_pose
from roboground.types import CameraIntrinsics, RGBDFrame

W, H = 160, 120
Y0 = 3.0                     # 平面 y = 3.0（世界系），恒定的 z 深度
ZC = 1.2
K = CameraIntrinsics(fx=140.0, fy=140.0, cx=(W - 1) / 2.0, cy=(H - 1) / 2.0,
                     width=W, height=H)
_TEX = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)


def _camera(dx: float, *, y_plane: float = Y0) -> RGBDFrame:
    """在 (dx, 0, ZC) 处、朝 +y 看的相机；场景是一张 y=y_plane 的纹理平面。

    每个像素的颜色 = 纹理在**世界坐标**处的取值 ⇒ 两个相机看到的是
    同一个物理纹理，视差完全由几何决定。
    """
    pose = look_at_pose((dx, 0.0, ZC), (dx, 5.0, ZC))
    fwd = np.asarray(pose.R).T @ np.array([0.0, 0.0, 1.0])
    assert fwd[1] > 0.99, f"相机应朝 +y 看，实际 {fwd}"

    uu = (np.arange(W) + 0.5)[None, :].repeat(H, axis=0)
    vv = (np.arange(H) + 0.5)[:, None].repeat(W, axis=1)
    k = np.stack([(uu - K.cx) / K.fx, (vv - K.cy) / K.fy, np.ones_like(uu)], axis=-1)
    R = np.asarray(pose.R, dtype=np.float64)
    C = -R.T @ np.asarray(pose.t, dtype=np.float64)
    dw = k @ R                                        # 相机系方向 → 世界系
    s = (y_plane - C[1]) / dw[..., 1]                 # 与平面求交
    pw = C[None, None, :] + s[..., None] * dw
    xi = np.clip((pw[..., 0] + 2.0) / 4.0 * (_TEX.shape[1] - 1), 0,
                 _TEX.shape[1] - 1).astype(int)
    zi = np.clip((pw[..., 2] - 0.2) / 2.4 * (_TEX.shape[0] - 1), 0,
                 _TEX.shape[0] - 1).astype(int)
    return RGBDFrame(color=_TEX[zi, xi],
                     depth_m=np.full((H, W), float(y_plane), np.float32),
                     intrinsics=K, pose=pose, frame_id=f"c{dx:+.2f}")


def _run(baseline: float, n_views: int):
    frames = [_camera(0.0)] + [_camera(baseline * i) for i in range(1, n_views)]
    res = estimate_depth_plane_sweep(frames, 0, depth_min=1.5, depth_max=6.0,
                                     n_planes=48, downsample=1, patch=7,
                                     max_source_views=8)
    gt = frames[0].depth_m
    m = (res.depth_m > 0) & (gt > 0)
    err = np.abs(res.depth_m[m] - gt[m]) if m.sum() else np.array([np.inf])
    return res, err, float(m.sum())


# ==========================================================================
# 1) 实现正确性：有基线时应当恢复出厘米级深度
# ==========================================================================
def test_plane_sweep_recovers_known_depth_with_baseline():
    """★ 基线 0.30 m、4 个视角时，3.000 m 的平面应当恢复到厘米级。

    场景纹理按世界坐标生成，与 `mvs_depth.py` 无关，所以通过这条测试
    说明的是"plane-sweep 的投影、代价、WTA、亚像素细化这一整套是对的"。
    """
    res, err, n = _run(0.30, 4)
    assert n > 2000, f"有效像素太少（{n:.0f}），测试没有区分力"
    # 阈值按**本配置**（160×120、48 个深度平面）实测 0.069 m 留余量设定。
    # 提高分辨率/平面数会明显更准：320×240 + 80 平面实测中位 0.025 m。
    # 这里不追求"最好"，只要求"厘米级且不退化" —— 精度上限由采样决定，
    # 不该把它写死成一个过紧的数字，否则以后调参就会误报。
    assert float(np.median(err)) < 0.10, (
        f"中位误差 {np.median(err):.4f} m，超过 10 cm —— 实现可能有问题")
    assert float(np.median(res.depth_m[res.depth_m > 0])) == pytest.approx(3.0, abs=0.15)


def test_plane_sweep_accuracy_improves_with_baseline():
    """★ 精度随**基线**单调改善（多视角立体的基本原理）。

    基线越大视差越大、深度分辨率越高。这条同时说明测试**确实**在测几何，
    而不是在测某个常数。
    """
    errs = []
    for b in (0.05, 0.15, 0.30):
        _, err, n = _run(b, 4)
        assert n > 500, f"基线 {b} 下有效像素太少"
        errs.append(float(np.median(err)))
    assert errs[0] > errs[1] > errs[2], (
        f"基线 0.05/0.15/0.30 m 的中位误差应递减，实测 {errs}")
    assert errs[0] / max(errs[2], 1e-6) > 3.0, (
        f"基线从 5 cm 增到 30 cm，误差只从 {errs[0]:.4f} 降到 {errs[2]:.4f}，"
        "差异过小，说明基线没有真正起作用")


# ==========================================================================
# 2) 几何硬前提：共光心 ⇒ 不可能
# ==========================================================================
def test_cocentered_views_cannot_recover_depth():
    """★★ **共光心**的多视角无法恢复深度 —— 这是几何事实，不是实现缺陷。

    相机光心重合时，参考像素对应的射线**是固定的**；改变深度假设只是沿这条
    固定射线前后移动三维点，而它在另一个（同光心）相机里的投影方向**不变**，
    于是投影像素坐标与深度**无关**，代价函数恒为常数，argmin 无意义。

    这正是本项目数据集的处境：2D-3D-S `/data` 的 N 个视角光心离散只有
    2.7e-06 m。所以那份数据**不能**验证"只有 RGB 也能出深度"。
    """
    res, err, n = _run(0.00, 2)
    assert n > 2000, "即使共光心，也应有大量像素参与（只是估计值无意义）"
    assert float(np.median(err)) > 1.0, (
        f"共光心时竟然把深度估准了（中位误差 {np.median(err):.4f} m）？"
        "这不符合几何 —— 请检查是不是相机基线弄错了")
    # 代价函数没有极小值 ⇒ 估计值会塌到搜索边界附近
    est = res.depth_m[res.depth_m > 0]
    assert float(np.median(est)) > 3.5, (
        f"共光心时估计值应无意义地偏向搜索边界，实测中位 {np.median(est):.3f} m")


def test_projected_pixel_is_independent_of_depth_when_cocentered():
    """★★ 上面那条结论的**直接**证明：投影像素坐标与深度假设无关。

    在 40 倍深度范围（0.5 m → 20 m）内，参考像素投到同光心源相机的坐标
    变化应远小于 0.1 像素。这是"没有视差"的可判定判据。
    """
    ref = _camera(0.0)
    src = _camera(0.0)                      # 同一光心
    R_ref = np.asarray(ref.pose.R, dtype=np.float64)
    t_ref = np.asarray(ref.pose.t, dtype=np.float64)
    R_s = np.asarray(src.pose.R, dtype=np.float64)
    t_s = np.asarray(src.pose.t, dtype=np.float64)
    Kc = ref.intrinsics
    us = []
    for z in (0.5, 1.0, 2.0, 4.0, 8.0, 20.0):
        # 取一个离光轴较远的像素，让视差效应尽量显著
        u, v = 20.0, 20.0
        kx, ky = (u - Kc.cx) / Kc.fx, (v - Kc.cy) / Kc.fy
        p_cam = np.array([kx * z, ky * z, z])
        p_world = R_ref.T @ (p_cam - t_ref)
        p_s = R_s @ p_world + t_s
        us.append(p_s[0] * src.intrinsics.fx / p_s[2] + src.intrinsics.cx)
    spread = float(np.max(us) - np.min(us))
    assert spread < 0.1, (
        f"共光心下投影坐标随深度变化了 {spread:.4f} 像素 —— 应有视差吗？")


# ==========================================================================
# 3) 边界与健壮性
# ==========================================================================
def test_plane_sweep_input_validation():
    f = _camera(0.0)
    with pytest.raises(ValueError):
        estimate_depth_plane_sweep([f], 0)                 # 至少 2 个视角
    with pytest.raises(IndexError):
        estimate_depth_plane_sweep([f, _camera(0.3)], 5)   # ref_index 越界


def test_plane_sweep_returns_full_resolution_depth():
    """返回的深度图应与参考视角同尺寸，且 0 表示无效。"""
    res, _, _ = _run(0.30, 3)
    assert res.depth_m.shape == (H, W)
    assert res.cost.shape == (H, W)
    assert res.n_views_used >= 2
    assert np.all(res.depth_m >= 0), "无效像素必须是 0，不能是负数或 NaN"
    assert np.isfinite(res.depth_m).all()


def test_depth_is_zero_outside_search_range_is_not_reported_as_valid():
    """搜索范围之外的像素不应给出**范围外**的深度值。"""
    frames = [_camera(0.0), _camera(0.3)]
    res = estimate_depth_plane_sweep(frames, 0, depth_min=1.5, depth_max=6.0,
                                     n_planes=32, downsample=1, patch=5,
                                     max_source_views=2)
    valid = res.depth_m[res.depth_m > 0]
    assert valid.size > 0
    assert valid.min() >= 1.5 - 1e-3 and valid.max() <= 6.0 + 1e-3, (
        f"深度估计跑出了搜索范围 [{valid.min():.3f}, {valid.max():.3f}]")
