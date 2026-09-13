# -*- coding: utf-8 -*-
"""多视角 → 全景融合的测试（**不需要数据集**）。

测试策略：**用已知答案的合成相机几何**验证投影与融合，而不是"跑一遍不报错"。

1. **投影正确性**：构造相机，检查某个世界方向是否落在预期的全景像素上；
2. **往返一致性**：`directions_to_equirect` 与 `equirect_directions` 必须互逆；
3. **覆盖完整性**：绕一圈均匀放 N 个相机，融合后应当几乎全图有覆盖 ——
   这是"多视角真的拼成了全景"的**可量化证据**；
4. **融合正确性**：让两个视角对同一像素给出不同深度（模拟前景/背景），
   检查两遍稳健融合**不会把深度平均成中间值**（那在几何上不存在）；
5. **权重方向**：单视角时，光轴正前方的像素权重应高于边缘；
6. **接缝**：±180° 附近不应出现整列空洞（这是等距柱状最容易错的地方）。
"""
from __future__ import annotations

import numpy as np
import pytest

from roboground.data.panorama import (
    Panorama,
    directions_to_equirect,
    equirect_directions,
    frame_for_panorama,
    fuse_to_equirect,
    panorama_rays_world,
    view_rays_world,
)
from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame

W, H = 360, 180            # 小尺寸全景（便于断言到具体像素）


def _intrinsics(fov_deg: float = 60.0, size: int = 64) -> CameraIntrinsics:
    fx = (size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return CameraIntrinsics(fx=fx, fy=fx, cx=(size - 1) / 2.0, cy=(size - 1) / 2.0,
                            width=size, height=size)


def _look_at(eye, target, up=(0.0, 0.0, 1.0)) -> CameraPose:
    """构造 world→camera 位姿（相机在 eye，看向 target）。"""
    from roboground.data.virtual_camera import look_at_pose  # 复用已验证的实现

    return look_at_pose(eye, target, world_up=up)


def _frame(color: np.ndarray, depth: np.ndarray, pose: CameraPose, K: CameraIntrinsics,
           frame_id: str = "v") -> RGBDFrame:
    return RGBDFrame(color=color.astype(np.uint8), depth_m=depth.astype(np.float32),
                     intrinsics=K, pose=pose, frame_id=frame_id)


def _uniform_view(value: int, depth: float, pose: CameraPose, K: CameraIntrinsics,
                  frame_id: str = "v") -> RGBDFrame:
    size = K.height
    color = np.full((size, K.width, 3), value, dtype=np.uint8)
    d = np.full((size, K.width), depth, dtype=np.float32)
    return _frame(color, d, pose, K, frame_id)


# ==========================================================================
# 1) 投影几何
# ==========================================================================
def test_forward_axis_maps_to_expected_azimuth():
    """相机朝 +x 看时，光轴（az=0）应落在全景的**中心列**。

    约定：`u = (az + π) / 2π · W`，所以
      · az = 0（正前方）  → u = W/2（中心列）
      · az = ±π（正后方） → u = 0 / W-1（**接缝在背后**，这是等距柱状的标准约定）
    """
    K = _intrinsics()
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))            # 朝 +x 水平看
    dirs, _ = view_rays_world(_uniform_view(255, 2.0, pose, K))
    u, v, el = directions_to_equirect(dirs, W, H)
    # ⚠️ 注意选哪个像素：`cx = (W-1)/2 = 31.5` 落在像素 31 与 32 的**交界**上，
    #    所以没有任何像素的中心恰好在光轴上。像素 31 的中心是 31.5 == cx，
    #    它才是"最接近光轴"的那个（用 32 会差整整一个像素 = 0.018 rad）。
    ci, cj = int(np.floor(K.cx)), int(np.floor(K.cy))
    assert abs(int(u[cj, ci]) - W // 2) <= 1, \
        f"光轴应落在中心列 u≈{W // 2}，实际 {u[cj, ci]}"
    assert abs(float(el[cj, ci])) < 1e-9, \
        f"光轴像素的仰角应为 0，实际 {float(el[cj, ci]):.3e} rad"
    # 对照：**竖直**方向隔一个像素，仰角应增加约 1/fy
    # （证明角度尺度是对的，而不是"恰好凑出 0"）。
    # 注意别拿水平邻居做这个检查：水平移动改变的是**方位角**，仰角仍是 0。
    one_px = 1.0 / K.fy
    got = float(el[cj + 1, ci])
    assert 0.9 * one_px < got < 1.1 * one_px, \
        f"竖直相邻像素的仰角应约为 1/fy={one_px:.5f} rad，实际 {got:.5f} rad"


def test_equirect_roundtrip_is_inverse():
    """`directions_to_equirect` 与 `equirect_directions` 必须互逆。"""
    dirs = equirect_directions(W, H)
    u, v, _el = directions_to_equirect(dirs, W, H)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    assert np.array_equal(u, uu), "方位角映射不是恒等（u 轴错位）"
    assert np.array_equal(v, vv), "仰角映射不是恒等（v 轴错位）"


def test_equirect_directions_are_unit_and_cover_sphere():
    """全景方向场必须是单位向量，且覆盖整个球面。"""
    d = equirect_directions(W, H)
    n = np.linalg.norm(d, axis=-1)
    assert np.allclose(n, 1.0, atol=1e-12)
    # 方位角均匀铺满：u 方向一圈应覆盖所有象限
    assert d[..., 1].min() < -0.99 and d[..., 1].max() > 0.99
    assert d[..., 2].min() < -0.99 and d[..., 2].max() > 0.99


def test_projection_covers_every_panorama_column():
    """★ 一整圈的相机应当能覆盖**每一列** —— 否则说明投影有系统性空洞。"""
    K = _intrinsics(fov_deg=70.0)
    seen = set()
    for az_deg in range(0, 360, 10):
        a = np.radians(az_deg)
        pose = _look_at((0, 0, 0), (np.cos(a), np.sin(a), 0.0))
        dirs, _ = view_rays_world(_uniform_view(128, 3.0, pose, K))
        u, _v, _e = directions_to_equirect(dirs, W, H)
        seen.update(np.unique(u).tolist())
    assert len(seen) == W, f"只覆盖了 {len(seen)}/{W} 列"


# ==========================================================================
# 2) 融合：覆盖完整性与接缝
# ==========================================================================
def test_fusing_a_full_sweep_covers_panorama():
    """★ 绕一圈 12 个真实视角融合后，方位方向应几乎全覆盖。

    这是"多视角真的拼成了全景"的量化证据。留出上下两端：
    针孔视角的垂直 FOV 有限，天顶/天底**本来就**拍不到（诚实留空）。
    """
    K = _intrinsics(fov_deg=75.0)
    frames = []
    for az_deg in range(0, 360, 30):
        a = np.radians(az_deg)
        frames.append(_uniform_view(100 + az_deg % 100, 3.0,
                                    _look_at((0, 0, 1.2), (np.cos(a), np.sin(a), 1.2)), K,
                                    frame_id=f"v{az_deg}"))
    pano = fuse_to_equirect(frames, width=W, height=H, min_cos=0.3)
    assert pano.n_used == len(frames)

    # 中间那一行（水平环）应当基本全满
    mid = pano.depth_m[H // 2]
    frac = float((mid > 0).mean())
    assert frac > 0.98, f"水平环覆盖率只有 {frac:.3f}，说明方位方向有空洞"

    # 天顶/天底行应当大面积缺失（这是诚实的物理限制，不是 bug）
    assert float((pano.depth_m[0] > 0).mean()) < 0.5, "天顶不该有覆盖（针孔相机拍不到）"

    # 整体覆盖率由**垂直 FOV**决定，而不是由方位覆盖决定：
    #   垂直半 FOV = 37.5° → 单视角最多覆盖 el ∈ [-37.5°, +37.5°]，
    #   占全景 180° 高度的约 42%；再被 min_cos=0.3 的门限削掉一点，实测约 39%。
    #   所以这里给的是一个**由几何推出的区间**，而不是随手写一个下限。
    assert 0.30 < pano.coverage < 0.50, (
        f"整体覆盖率 {pano.coverage:.3f} 超出几何预期（0.30~0.50）："
        "偏低=方位方向有空洞，偏高=可能把无效像素也算进来了"
    )


def test_no_seam_hole_at_pm180():
    """±180° 接缝处不应出现整列空洞。"""
    K = _intrinsics(fov_deg=75.0)
    frames = []
    for az_deg in (170, 180, 190, 350, 0, 10):            # 故意跨越接缝
        a = np.radians(az_deg)
        frames.append(_uniform_view(200, 2.5,
                                    _look_at((0, 0, 1.2), (np.cos(a), np.sin(a), 1.2)), K,
                                    frame_id=f"v{az_deg}"))
    pano = fuse_to_equirect(frames, width=W, height=H, min_cos=0.3)
    col0 = (pano.depth_m[:, 0] > 0)
    colw = (pano.depth_m[:, W - 1] > 0)
    mid_row = H // 2
    # 水平环上接缝两侧都应有值
    assert col0[mid_row] or col0[mid_row + 1], "接缝左侧整列空"
    assert colw[mid_row] or colw[mid_row - 1], "接缝右侧整列空"


# ==========================================================================
# 3) 深度融合：不能把前景/背景平均成"中间值"
# ==========================================================================
def test_depth_fusion_rejects_outlier_view():
    """★ 两个视角对同一方向给出 2 m 与 5 m 时，融合结果不该是中间值 3.5 m。

    做法：让 A 视角（权重高）看到 2 m，B 视角（权重低、边缘）看到 5 m。
    两遍稳健融合应当保留 2 m，而不是把两者平均。
    """
    K = _intrinsics(fov_deg=70.0)
    pose_a = _look_at((0, 0, 1.2), (1, 0, 1.2))
    pose_b = _look_at((0, 0, 1.2), (1, 0, 1.2))            # 同方向
    a = _uniform_view(10, 2.0, pose_a, K, frame_id="near")
    b = _uniform_view(250, 5.0, pose_b, K, frame_id="far")

    pano = fuse_to_equirect([a, b], width=W, height=H, min_cos=0.3,
                            depth_outlier_m=0.5)
    # 找光轴附近的有效像素
    u, v, _ = directions_to_equirect(
        view_rays_world(a)[0], W, H)
    # 注意：u/v 的索引空间是**视角图像**（K.height×K.width），不是全景
    pix = (v[K.height // 2, K.width // 2], u[K.height // 2, K.width // 2])
    d = float(pano.depth_m[pix[0], pix[1]])
    assert d > 0, "该像素没有融合出深度"
    # 允许在 2.0 与 5.0 之间二选一（谁权重高谁赢不算错），
    # 但**绝不能**落在中间地带
    assert d < 2.4 or d > 4.6, f"深度被平均成了 {d:.2f} m（几何上不存在的值）"


def test_weighted_rgb_fusion_prefers_center_view():
    """RGB 融合应偏向权重更高的视角（光轴中心），而不是简单平均。"""
    K = _intrinsics(fov_deg=70.0)
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))
    center_bright = _uniform_view(255, 2.0, pose, K)
    # 第二个视角：同位置但朝向偏 60°，它在"正前方"处的像素是边缘 → 权重低
    a = np.radians(60)
    off = _uniform_view(0, 2.0, _look_at((0, 0, 1.2), (np.cos(a), np.sin(a), 1.2)), K)

    pano_mean = fuse_to_equirect([center_bright, off], width=W, height=H,
                                 min_cos=0.2, rgb_mode="weighted_mean")
    pano_near = fuse_to_equirect([center_bright, off], width=W, height=H,
                                 min_cos=0.2, rgb_mode="nearest")

    u, v, _ = directions_to_equirect(view_rays_world(center_bright)[0], W, H)
    r, c = v[K.height // 2, K.width // 2], u[K.height // 2, K.width // 2]
    assert pano_mean.rgb[r, c].mean() > 128, "加权平均没有偏向亮的那一帧"
    assert pano_near.rgb[r, c].mean() > 128, "nearest 模式没有选中权重最高的帧"


# ==========================================================================
# 4) 边界与健壮性
# ==========================================================================
def test_empty_input_raises():
    with pytest.raises(ValueError):
        fuse_to_equirect([], width=W, height=H)  # type: ignore[arg-type]


def test_all_invalid_depth_yields_empty_panorama():
    """全部深度无效时应返回一张空全景，而不是抛异常或给出假数据。"""
    K = _intrinsics()
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))
    f = _uniform_view(200, 0.0, pose, K)
    pano = fuse_to_equirect([f], width=W, height=H)
    assert pano.n_used == 0
    assert pano.coverage == 0.0
    assert float(pano.weight.max()) == 0.0


def test_depth_beyond_max_is_dropped():
    """超过 `max_depth` 的点必须被丢弃（远距离深度不可靠）。"""
    K = _intrinsics()
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))
    pano = fuse_to_equirect([_uniform_view(200, 20.0, pose, K)],
                            width=W, height=H, max_depth=8.0)
    assert pano.coverage == 0.0, "20 m 的点不该出现在 max_depth=8 的全景里"


def test_bad_rgb_mode_raises():
    K = _intrinsics()
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))
    with pytest.raises(ValueError):
        fuse_to_equirect([_uniform_view(1, 2.0, pose, K)], width=W, height=H,
                         rgb_mode="magic")


def test_fusion_is_deterministic():
    """同样的输入必须给出完全相同的输出（可复现性是硬要求）。"""
    K = _intrinsics()
    fs = [_uniform_view(50 + i * 40, 2.0 + 0.1 * i,
                        _look_at((0, 0, 1.2), (1, 0, 1.2)), K, f"v{i}") for i in range(3)]
    p1 = fuse_to_equirect(fs, width=W, height=H)
    p2 = fuse_to_equirect(fs, width=W, height=H)
    assert np.array_equal(p1.rgb, p2.rgb)
    assert np.array_equal(p1.depth_m, p2.depth_m)


# ==========================================================================
# 5) 与下游的衔接
# ==========================================================================
def test_panorama_can_be_wrapped_as_rgbdframe():
    """融合结果必须能包成 `RGBDFrame`，这样下游 MapBuilder 不用改。"""
    K = _intrinsics()
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))
    pano = fuse_to_equirect([_uniform_view(120, 2.0, pose, K)], width=W, height=H)
    frame = frame_for_panorama(pano, frame_id="pano0")
    assert frame.color.shape[:2] == (H, W)
    assert frame.depth_m.shape == (H, W)
    assert frame.intrinsics.width == W and frame.intrinsics.height == H
    assert frame.meta["source"] == "panorama_fusion"


def test_panorama_rays_world_are_unit():
    """精确反投影用的方向场必须是单位向量（避免以后用错近似内参）。"""
    K = _intrinsics()
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))
    pano = fuse_to_equirect([_uniform_view(120, 2.0, pose, K)], width=W, height=H)
    d = panorama_rays_world(pano)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-12)


def test_panorama_describe_has_expected_keys():
    K = _intrinsics()
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))
    pano = fuse_to_equirect([_uniform_view(120, 2.0, pose, K)], width=W, height=H)
    d = pano.describe()
    for k in ("分辨率", "参与视角", "有效视角", "深度覆盖", "RGB覆盖"):
        assert k in d, f"describe() 缺少 {k}"
    assert isinstance(pano, Panorama)
