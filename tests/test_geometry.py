"""几何层测试：相机模型、投影/反投影、内参变换。"""

from __future__ import annotations

import numpy as np
import pytest

from roboground.geometry.camera import (
    camera_ray_directions,
    pixel_grid,
    points_in_camera_frustum,
    transform_points,
)
from roboground.geometry.projection import (
    backproject_detection,
    crop_intrinsics,
    depth_to_points_camera,
    depth_to_points_world,
    filter_depth_ghosts,
    project_points_to_image,
    raw_depth_to_meters,
    sample_intrinsics_like,
)
from roboground.types import CameraIntrinsics, CameraPose, Detection2D, RGBDFrame


# ==========================================================================
# 基础算子
# ==========================================================================
def test_pixel_grid_shapes_and_order():
    flat = pixel_grid(4, 3, flatten=True)
    assert flat.shape == (12, 2)
    assert flat[0].tolist() == [0.0, 0.0]
    assert flat[1].tolist() == [1.0, 0.0]      # 同一行内先走列
    assert flat[4].tolist() == [0.0, 1.0]      # 第 2 行

    grid = pixel_grid(4, 3, flatten=False)
    assert grid.shape == (3, 4, 2)


def test_ray_directions_are_unit_length(intrinsics):
    dirs = camera_ray_directions(intrinsics, 160, 120)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
    assert np.all(dirs[:, 2] > 0)             # 全部朝前


def test_transform_points_matches_manual():
    pts = np.array([[1.0, 0.0, 0.0]])
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.0, 0.0, 2.0])
    out = transform_points(pts, R, t)
    assert np.allclose(out, [[0.0, 1.0, 2.0]])


# ==========================================================================
# 深度 → 点
# ==========================================================================
def test_depth_to_points_camera_flat_plane(intrinsics):
    depth = np.full((120, 160), 2.5, dtype=np.float32)
    pts = depth_to_points_camera(depth, intrinsics, min_depth=0.1, max_depth=8.0)
    assert pts.shape[0] == 120 * 160
    assert np.allclose(pts[:, 2], 2.5, atol=1e-6)         # z 恒为深度

    # 中心像素应落在光轴上
    center_px = np.array([79.5, 59.5])
    expected_x = (center_px[0] - intrinsics.cx) * 2.5 / intrinsics.fx
    expected_y = (center_px[1] - intrinsics.cy) * 2.5 / intrinsics.fy
    assert abs(expected_x) < 1e-6 and abs(expected_y) < 1e-6


def test_depth_to_points_camera_respects_range():
    # 内参尺寸必须与深度图一致（否则会被 geometry 层主动拒绝）
    K = CameraIntrinsics(fx=20.0, fy=20.0, cx=10.0, cy=10.0, width=20, height=20)
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    depth[0, 0] = 0.0        # 无效
    depth[1, 1] = 99.0       # 过远
    pts, mask = depth_to_points_camera(depth, K, min_depth=0.1,
                                       max_depth=8.0, return_mask=True)
    assert mask.shape == (400,)
    assert mask.sum() == 400 - 2


def test_depth_to_points_world_identity_equals_camera():
    K = CameraIntrinsics(fx=20.0, fy=20.0, cx=10.0, cy=10.0, width=20, height=20)
    depth = np.full((20, 20), 3.0, dtype=np.float32)
    cam = depth_to_points_camera(depth, K)
    world = depth_to_points_world(depth, K, CameraPose.identity())
    assert np.allclose(cam, world)


def test_depth_to_points_world_applies_pose():
    K = CameraIntrinsics(fx=20.0, fy=20.0, cx=10.0, cy=10.0, width=20, height=20)
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    # 相机在世界上 (1,0,0)，朝向不变 → 世界点 = 相机点 + (1,0,0)
    pose = CameraPose(np.eye(3), -np.array([1.0, 0.0, 0.0]))
    world = depth_to_points_world(depth, K, pose)
    cam = depth_to_points_camera(depth, K)
    assert np.allclose(world - cam, np.array([1.0, 0.0, 0.0]))


def test_depth_shape_mismatch_with_intrinsics_raises(intrinsics):
    """内参尺寸与深度图不一致时必须早失败（这是最隐蔽的 bug 之一）。"""
    depth = np.full((50, 50), 1.0, dtype=np.float32)
    with pytest.raises(ValueError, match="不一致"):
        depth_to_points_camera(depth, intrinsics)


# ==========================================================================
# 投影闭环
# ==========================================================================
def test_project_unproject_roundtrip(intrinsics):
    depth = np.full((120, 160), 2.0, dtype=np.float32)
    pts_world = depth_to_points_world(depth, intrinsics, CameraPose.identity())
    uv, z, valid = project_points_to_image(pts_world, CameraPose.identity(), intrinsics, 160, 120)
    assert valid.all()
    assert np.allclose(z, 2.0, atol=1e-5)

    # 投影回的像素应重新反投影出原来的点
    flat_depth = depth.reshape(-1)
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    d = flat_depth[v * 160 + u]
    back = np.stack([
        (u - intrinsics.cx) * d / intrinsics.fx,
        (v - intrinsics.cy) * d / intrinsics.fy,
        d,
    ], axis=1)
    assert np.allclose(back, pts_world, atol=0.05)


def test_points_behind_camera_are_invalid(intrinsics):
    pts_behind = np.array([[0.0, 0.0, -1.0]])       # 相机后方
    uv, z, valid = project_points_to_image(pts_behind, CameraPose.identity(), intrinsics, 160, 120)
    assert not valid[0]
    assert np.isnan(uv[0, 0])


def test_points_outside_fov_are_invalid(intrinsics):
    pts_far_side = np.array([[100.0, 0.0, 1.0]])    # 视野外
    _, _, valid = project_points_to_image(pts_far_side, CameraPose.identity(), intrinsics, 160, 120)
    assert not valid[0]


# ==========================================================================
# 反投影检测
# ==========================================================================
def test_backproject_detection_bbox(intrinsics):
    depth = np.full((120, 160), 2.0, dtype=np.float32)
    frame = RGBDFrame(color=np.zeros((120, 160, 3), np.uint8), depth_m=depth,
                      intrinsics=intrinsics, pose=CameraPose.identity(), frame_id="t")
    det = Detection2D(label="cup", score=0.9, bbox=np.array([70.0, 50.0, 90.0, 70.0]))

    obs = backproject_detection(det, frame)
    assert obs.num_points == 20 * 20
    assert np.allclose(obs.points_world[:, 2], 2.0)
    # 中心的 x 应该接近 (80-cx)*d/fx
    assert abs(obs.centroid[0] - (80 - intrinsics.cx) * 2.0 / intrinsics.fx) < 0.02


def test_backproject_detection_with_mask(intrinsics):
    depth = np.full((20, 20), 1.5, dtype=np.float32)
    frame = RGBDFrame(color=np.zeros((20, 20, 3), np.uint8), depth_m=depth,
                      intrinsics=CameraIntrinsics(10, 10, 10, 10, 20, 20),
                      pose=CameraPose.identity(), frame_id="t")
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    det = Detection2D(label="box", score=0.8, bbox=np.array([0.0, 0.0, 19.0, 19.0]), mask=mask)

    obs = backproject_detection(det, frame)
    assert obs.num_points == 25          # 只取掩码内的像素，而不是整个 bbox


def test_backproject_detection_all_invalid_depth(intrinsics):
    depth = np.zeros((20, 20), dtype=np.float32)      # 全无效
    frame = RGBDFrame(color=np.zeros((20, 20, 3), np.uint8), depth_m=depth,
                      intrinsics=CameraIntrinsics(10, 10, 10, 10, 20, 20),
                      pose=CameraPose.identity(), frame_id="t")
    det = Detection2D(label="box", score=0.8, bbox=np.array([1.0, 1.0, 5.0, 5.0]))
    obs = backproject_detection(det, frame)
    assert obs.num_points == 0
    assert obs.centroid is None


# ==========================================================================
# 深度预处理
# ==========================================================================
def test_raw_depth_to_meters():
    raw = np.array([[0, 1000, 2500]], dtype=np.uint16)
    out = raw_depth_to_meters(raw, 1000.0)
    assert out[0, 0] == 0.0
    assert abs(out[0, 1] - 1.0) < 1e-6
    assert abs(out[0, 2] - 2.5) < 1e-6


def test_filter_depth_ghosts_removes_far_outliers():
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    depth[0, 0] = 50.0                  # 幽灵点
    out = filter_depth_ghosts(depth, percentile=95.0, min_depth=0.1, max_depth=8.0)
    assert out[0, 0] == 0.0
    assert np.allclose(out[1:, :], 2.0)
    assert out.shape == depth.shape     # 形状必须保持


def test_filter_depth_ghosts_all_invalid_is_safe():
    depth = np.zeros((5, 5), dtype=np.float32)
    out = filter_depth_ghosts(depth)
    assert out.shape == (5, 5)
    assert out.sum() == 0.0


# ==========================================================================
# 内参变换
# ==========================================================================
def test_sample_intrinsics_like_scales_correctly(intrinsics):
    scaled = sample_intrinsics_like(intrinsics, 320, 240)
    assert abs(scaled.fx - intrinsics.fx * 2.0) < 1e-6
    assert abs(scaled.cx - intrinsics.cx * 2.0) < 1e-6
    assert scaled.width == 320 and scaled.height == 240


def test_crop_intrinsics_shifts_principal_point(intrinsics):
    cropped = crop_intrinsics(intrinsics, x0=10, y0=5)
    assert abs(cropped.cx - (intrinsics.cx - 10)) < 1e-9
    assert abs(cropped.cy - (intrinsics.cy - 5)) < 1e-9


# ==========================================================================
# 视野判定与位姿工具
# ==========================================================================
def test_points_in_camera_frustum(intrinsics):
    pts = np.array([[0.0, 0.0, 2.0], [100.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    inside, uv = points_in_camera_frustum(pts, CameraPose.identity(), intrinsics, 160, 120)
    assert inside[0] and not inside[1] and not inside[2]


def test_camera_pose_inverse_roundtrip():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([1.0, 2.0, 3.0])
    pose = CameraPose(R, t)
    inv = pose.inverse()

    pts = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(inv.cam_to_world(pose.cam_to_world(pts)), pts, atol=1e-9)
    assert np.allclose(pose.cam_to_world(pose.world_to_cam(pts)), pts, atol=1e-9)


def test_camera_center_formula():
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    pose = CameraPose(R, t)
    # cam = p + t = 0 → p = -t
    assert np.allclose(pose.camera_center(), -t)


def test_from_sunrgbd_permutation_semantics():
    """SUN RGB-D 位姿：世界 +y（前方）必须映射到相机 +z（深度）。"""
    Rtilt = np.eye(3)
    pose = CameraPose.from_sunrgbd(Rtilt)
    cam = pose.world_to_cam(np.array([[0.0, 2.0, 0.0]]))
    assert np.allclose(cam, [[0.0, 0.0, 2.0]], atol=1e-9)

    # 世界 +z（上）→ 相机 -y（上，因为相机 y 朝下）
    cam_up = pose.world_to_cam(np.array([[0.0, 0.0, 1.0]]))
    assert np.allclose(cam_up, [[0.0, -1.0, 0.0]], atol=1e-9)
