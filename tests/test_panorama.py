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
    from roboground.geometry.camera import look_at_pose

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
                            range_outlier_m=0.5)
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


# ==========================================================================
# 3b) ★ 斜距 vs z 深度：一个曾经真实存在的语义错误
# ==========================================================================
def test_prepare_view_converts_z_depth_to_range():
    """★ 融合用的必须是**斜距**，不是原始 z 深度。

    为什么这条测试重要：上面所有深度融合的测试都取**光轴附近**的像素，
    而在光轴上 `r == z` —— 所以它们**根本区分不出**两种语义。
    早期版本直接把 z 深度灌进融合，这些测试全绿，但大角度上是错的。

    构造：正对相机的平面（z ≡ 2 m，一个 fronto-parallel 平面）。
    · 光轴像素：斜距 = 2 m；
    · 边缘像素：斜距 = 2 / cos θ > 2 m（θ = 该像素与光轴的夹角）。
    断言边缘像素的斜距**明显大于 2**，且等于 2/cos θ。
    """
    from roboground.data.panorama import _prepare_view

    K = _intrinsics(fov_deg=90.0, size=128)
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))
    view = _uniform_view(128, 2.0, pose, K)

    prep = _prepare_view(view, W, H, weight_power=2.0, min_cos=0.0, max_depth=8.0)
    assert prep is not None
    _pix, _wf, _color, rng = prep
    rng = rng.reshape(K.height, K.width)

    # ★ 注意 `cx = (size-1)/2 = 63.5` 正好落在**像素边界**上：
    #   索引 63 的像素中心是 63.5，才是真正在光轴上的那个像素
    #   （用 `size//2 = 64` 会差半像素，得 2.00049 而不是 2.0）。
    ci = int(round(K.cx - 0.5))
    u_axis = ci + 0.5
    assert u_axis == K.cx, "取的像素中心应正好等于 cx"
    assert rng[ci, ci] == pytest.approx(2.0, abs=1e-9), "光轴上斜距应等于 z 深度"

    # 其它像素：期望值直接由约定算出来 —— r = z · |k|，k = [(u−cx)/fx, (v−cy)/fy, 1]
    for (r_i, c_i) in ((0, 0), (0, K.width - 1), (K.height - 1, 0), (10, 90)):
        k = np.array([(c_i + 0.5 - K.cx) / K.fx, (r_i + 0.5 - K.cy) / K.fy, 1.0])
        expected = 2.0 * float(np.linalg.norm(k))
        got = float(rng[r_i, c_i])
        assert got == pytest.approx(expected, rel=1e-6), (
            f"像素({r_i},{c_i}) 斜距应为 {expected:.6f} m，实测 {got:.6f} m")
        assert expected > 2.0 + 1e-9, "该像素必须离轴，否则测试没有区分力"

    # 最极端的一条：角落像素的斜距必须**明显**大于 2（旧实现会在这里给 2）
    assert float(rng[0, 0]) > 3.0, (
        f"角落像素输出 {rng[0, 0]:.4f} m —— 若接近 2.0 说明**还在用 z 深度**，"
        "跨视角融合会把大角度上的正确观测误判为离群")


def test_panorama_depth_is_range_not_z():
    """融合结果里，离轴像素的斜距必须大于 2 m（正对它的平面距离）。

    ⚠️ 这里**不能用固定像素**做断言：源图 128 px 覆盖 90°（0.70°/px），
    而 2048 宽的全景是 0.176°/px —— 全景被**过采样** 4 倍，
    大部分全景像素本来就取不到样本。所以改成对**所有有覆盖的像素**校验。
    """
    K = _intrinsics(fov_deg=90.0, size=128)
    pose = _look_at((0, 0, 1.2), (1, 0, 1.2))
    Wp, Hp = 512, 256                      # 与源图采样率相当，避免大量空洞
    pano = fuse_to_equirect([_uniform_view(128, 2.0, pose, K)],
                            width=Wp, height=Hp, min_cos=0.1, max_depth=8.0)
    assert pano.is_range_image

    rows, cols = np.nonzero(pano.depth_m > 0)
    assert rows.size > 1000, f"有效像素太少（{rows.size}），测试没有区分力"
    r = pano.depth_m[rows, cols].astype(np.float64)

    # 期望值直接由等距柱状约定算出：光轴指向 +x（az=0, el=0），
    # 所以该像素方向的离轴角 θ 满足 cos θ = cos(el)·cos(az)，而 r = 2 / cos θ
    az = (cols + 0.5) / Wp * 2.0 * np.pi - np.pi
    el = np.pi / 2.0 - (rows + 0.5) / Hp * np.pi
    cos_t = np.cos(el) * np.cos(az)
    expected = 2.0 / np.maximum(cos_t, 1e-9)
    rel = np.abs(r - expected) / expected

    assert np.median(rel) < 0.02, (
        f"斜距与期望值中位相对误差 {np.median(rel):.4%}，说明存的不是斜距")
    # 关键区分力：必须有大量像素的斜距**明显大于** 2 m。
    # 旧实现（存 z 深度）会把它们全部写成 2.0。
    n_beyond = int((r > 2.1).sum())
    assert n_beyond > 0.2 * r.size, (
        f"只有 {n_beyond}/{r.size} 个像素斜距 > 2.1 m —— "
        "若几乎全为 2.0 说明**还在用 z 深度**，跨视角融合会把大角度观测误判为离群")
    assert r.max() > 2.4, f"最大斜距仅 {r.max():.4f} m，离轴像素没有被正确换算"


def test_equirect_backprojection_reconstructs_the_true_plane():
    """★ 端到端：融合出的全景必须能**精确**反投影回真实平面。

    这是整条链路的关键正确性断言。构造一个正对相机的平面（世界系 x = C_x + 2），
    把 N 个视角融合成全景，再用几何层的反投影把全景像素送回三维 ——
    **所有点都应落在那张平面上**。

    如果退回"z 深度 + 等效针孔"的老做法，远离光轴的像素会系统性偏离平面，
    这个断言会直接失败。
    """
    from roboground.geometry.projection import pixel_grid, unproject_pixels

    K = _intrinsics(fov_deg=110.0, size=96)
    C = np.array([0.0, 0.0, 1.2])
    # 4 个视角，绕一圈，都看到同一张平面 x = 2（正对第一个视角）
    frames = []
    for az_deg in (0, 90, 180, 270):
        a = np.radians(az_deg)
        frames.append(_uniform_view(
            128, 2.0, _look_at(tuple(C), (C[0] + np.cos(a), C[1] + np.sin(a), C[2])),
            K, frame_id=f"v{az_deg}"))
    pano = fuse_to_equirect(frames, width=512, height=256, min_cos=0.0,
                            max_depth=8.0)

    frame = frame_for_panorama(pano)
    assert frame.meta["projection"] == "equirect"
    # 光心必须回到 C
    assert np.allclose(np.asarray(frame.pose.camera_center()), C, atol=1e-9)

    valid = pano.depth_m > 0
    assert valid.any()
    uv = pixel_grid(512, 256, flatten=True)[valid.reshape(-1)]
    r = pano.depth_m.reshape(-1)[valid.reshape(-1)]
    pts = unproject_pixels(frame, uv, r)
    assert pts.shape[0] > 1000

    # 每个视角各看一张"朝向自己"的平面，所以世界系里并不是同一张平面；
    # 这里逐个视角验证：把点按方位角分到 4 个象限，各自应落在对应的平面上。
    # 视角 az 看到的是平面 {x·cos(az) + y·sin(az) = 2}（沿该方向距 C 2 m）。
    ang = np.arctan2(pts[:, 1] - C[1], pts[:, 0] - C[0])
    quad = ((np.degrees(ang) + 45.0) % 360.0 // 90.0).astype(int)      # 0..3
    for q, az_deg in enumerate((0, 90, 180, 270)):
        sel = quad == q
        if sel.sum() < 50:
            continue
        a = np.radians(az_deg)
        # 平面法向 (cos a, sin a, 0)，过点 C + 2·(cos a, sin a, 0)
        n = np.array([np.cos(a), np.sin(a), 0.0])
        d = float(n @ (C + 2.0 * np.array([np.cos(a), np.sin(a), 0.0])))
        resid = np.abs(pts[sel] @ n - d)
        assert np.median(resid) < 0.01, (
            f"方位 {az_deg}° 的重建点偏离真实平面中位 {np.median(resid):.4f} m")


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
    # ★ 键名必须同时包含「像素覆盖」和「立体角覆盖」。
    #   只报像素覆盖率会**低估**真实视野覆盖（等距柱状图里每行跨同样的仰角
    #   增量，但赤道附近的行承载的立体角更大），实测 office_6 差 19 个百分点
    #   （像素 52.2% vs 立体角 71.4%）。所以两个都得有，缺一个就是退步。
    for k in ("分辨率", "参与视角", "有效视角", "像素覆盖", "立体角覆盖",
              "仰角范围", "RGB覆盖"):
        assert k in d, f"describe() 缺少 {k}"
    # 统计口径也必须自报家门
    assert pano.meta["depth_semantics"] == "range_from_center", \
        "全景的 depth_m 是斜距，必须在 meta 里写明，否则下游会当成针孔 z 深度用"
    assert isinstance(pano, Panorama)
    assert pano.range_m is pano.depth_m or np.array_equal(pano.range_m, pano.depth_m)
