"""合成场景渲染器测试：验证"渲染 → 反投影 → 落在 GT 框内"这个闭环。

这是整个项目里**最有价值的一类测试**：它同时校验了
- 射线-盒求交的正确性（渲染）
- 位姿约定的正确性（world ↔ camera）
- 反投影公式的正确性（深度 → 3D 点）
三者任何一个错了，闭环都会失败。
"""

from __future__ import annotations

import numpy as np
import pytest

from roboground.data.synthetic import (
    SceneObject,
    SyntheticRoom,
    default_intrinsics,
    make_synthetic_frame,
    make_synthetic_sequence,
    ray_box_intersection,
    ray_rotated_box_intersection,
    rotz,
)
from roboground.geometry.projection import depth_to_points_world


# ==========================================================================
# 射线-盒求交
# ==========================================================================
def test_ray_box_intersection_hit_and_miss():
    origins = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])       # 一个向前、一个向后
    t = ray_box_intersection(origins, dirs, np.array([-0.5, -0.5, 1.0]),
                             np.array([0.5, 0.5, 2.0]))
    assert abs(t[0] - 1.0) < 1e-9        # 命中近面 z=1
    assert np.isinf(t[1])                # 朝反方向 → 不命中


def test_ray_box_intersection_inside_box_returns_zero():
    t = ray_box_intersection(np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]),
                             np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    assert t[0] == 0.0                    # 起点在盒内 → 参数为 0


def test_ray_rotated_box_respects_heading():
    """把一个长方体绕 z 转 90° 后，原本命中的射线应该打不中。"""
    origins = np.array([[0.0, 0.0, 0.0]])
    dirs = np.array([[1.0, 0.0, 0.0]])          # 沿 +x 看
    center = np.array([2.0, 0.0, 0.0])
    size = np.array([0.2, 2.0, 0.2])            # 细长条沿 y 方向

    # heading=0：长边沿 y，从 +x 看过去很薄但能命中
    t0 = ray_rotated_box_intersection(origins, dirs, center, size, 0.0)
    assert np.isfinite(t0[0])

    # heading=90°：长边转向 x，从 +x 看过去会先撞到远端面（仍命中，但参数不同）
    t90 = ray_rotated_box_intersection(origins, dirs, center, size, np.pi / 2)
    assert np.isfinite(t90[0])
    assert abs(t90[0] - (2.0 - 1.0)) < 1e-6     # 近端面在 x=1.0


def test_rotz_matrix():
    """synthetic.rotz 是 3D 的绕 z 旋转（BEV 的 2×2 版本在 eval.metrics 里）。"""
    R = rotz(np.pi / 2)
    assert R.shape == (3, 3)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-9)   # z 不变


# ==========================================================================
# 渲染 → 反投影闭环
# ==========================================================================
def _room_with_one_box() -> SyntheticRoom:
    return SyntheticRoom(
        width=4.0, depth=5.0, height=2.6,
        objects=[SceneObject("box", [0.0, 2.0, 0.5], [0.6, 0.6, 0.6], 0.0, (200, 60, 60))],
    )


def test_render_produces_valid_depth():
    room = _room_with_one_box()
    K = default_intrinsics(160, 120)
    from roboground.data.synthetic import _look_forward_pose

    pose = _look_forward_pose(np.array([0.0, 0.0, 1.0]))
    res = room.render(pose, K)

    assert res.depth_m.shape == (120, 160)
    valid = res.depth_m > 0
    assert valid.sum() > 1000                    # 有大量有效深度
    assert res.depth_m[valid].min() > 0.1
    # 地面/墙/物体都在 1~6m 范围内
    assert res.depth_m[valid].max() < 8.0


def test_render_instance_id_matches_gt_boxes():
    room = _room_with_one_box()
    K = default_intrinsics(160, 120)
    from roboground.data.synthetic import _look_forward_pose

    pose = _look_forward_pose(np.array([0.0, 0.0, 1.0]))
    res = room.render(pose, K)

    assert res.labels == ["box"]
    assert res.boxes_3d.shape[0] == 1
    assert (res.instance_id == 0).sum() > 100    # 盒体占了足够多像素


def test_unprojected_instance_pixels_fall_inside_gt_box():
    """核心闭环断言：属于某个实例的像素，反投影后必须落在其 GT 盒内。"""
    room = SyntheticRoom(
        width=4.0, depth=5.0, height=2.6,
        objects=[
            SceneObject("box", [0.0, 2.0, 0.5], [0.6, 0.6, 0.6], 0.0, (200, 60, 60)),
            SceneObject("cube", [0.9, 2.5, 0.4], [0.5, 0.5, 0.5], 0.3, (60, 60, 200)),
        ],
    )
    K = default_intrinsics(200, 150)
    from roboground.data.synthetic import _look_forward_pose

    pose = _look_forward_pose(np.array([0.0, 0.0, 1.1]))
    res = room.render(pose, K)

    points_world = depth_to_points_world(res.depth_m, K, pose)
    # depth_to_points_world 会过滤无效深度，所以索引要同步过滤
    valid_mask = (res.depth_m > 0.1) & (res.depth_m < 12.0)
    inst = res.instance_id.reshape(-1)[valid_mask.reshape(-1)]
    assert points_world.shape[0] == inst.shape[0]

    for inst_id in np.unique(inst):
        if inst_id < 0:
            continue
        sel = points_world[inst == inst_id]
        box = res.boxes_3d[int(inst_id)]
        center, size, heading = box[:3], box[3:6], box[6]

        # 转到盒的局部系
        rel = sel - center[None, :]
        c, s = np.cos(-heading), np.sin(-heading)
        lx = rel[:, 0] * c - rel[:, 1] * s
        ly = rel[:, 0] * s + rel[:, 1] * c
        lz = rel[:, 2]
        half = np.abs(size) / 2.0 + 1e-3         # 容忍浮点误差
        inside = (np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1]) & (np.abs(lz) <= half[2])

        assert inside.mean() > 0.95, (
            f"实例 {inst_id}({res.labels[int(inst_id)]}) 只有 {inside.mean():.1%} 的像素落在 GT 盒内"
        )


def test_render_from_behind_returns_no_instance():
    """相机背对物体时，不应看到任何实例（验证 front 判定）。"""
    room = _room_with_one_box()
    K = default_intrinsics(160, 120)
    from roboground.data.synthetic import _look_forward_pose

    # 相机在物体后方、朝远离物体的方向看（朝 -y）
    pose = _look_forward_pose(np.array([0.0, 4.0, 1.0]))
    # _look_forward_pose 恒朝 +y 看；这里手动翻转 180°
    from roboground.types import CameraPose

    flip = np.diag([-1.0, -1.0, 1.0])
    pose = CameraPose(flip @ pose.R, flip @ pose.t)

    res = room.render(pose, K)
    assert (res.instance_id >= 0).sum() == 0


def test_render_empty_room():
    room = SyntheticRoom(objects=[])
    K = default_intrinsics(80, 60)
    from roboground.data.synthetic import _look_forward_pose

    res = room.render(_look_forward_pose(np.array([0.0, 0.0, 1.0])), K)
    assert res.boxes_3d.shape[0] == 0
    assert res.labels == []


# ==========================================================================
# 房间生成
# ==========================================================================
def test_random_room_is_deterministic():
    a = SyntheticRoom.random(seed=3, num_objects=4)
    b = SyntheticRoom.random(seed=3, num_objects=4)
    assert [o.label for o in a.objects] == [o.label for o in b.objects]
    assert np.allclose(a.gt_boxes_3d(), b.gt_boxes_3d())


def test_random_room_objects_inside_bounds():
    room = SyntheticRoom.random(seed=5, num_objects=6)
    for obj in room.objects:
        assert -room.width / 2 <= obj.center[0] <= room.width / 2
        assert 0 <= obj.center[1] <= room.depth
        assert obj.center[2] >= 0


def test_random_room_no_overlap_when_requested():
    room = SyntheticRoom.random(seed=9, num_objects=5, allow_overlap=False)
    for i in range(len(room.objects)):
        for j in range(i + 1, len(room.objects)):
            a, b = room.objects[i], room.objects[j]
            overlap = np.all(np.abs(a.center - b.center) < (a.size + b.size) / 2)
            assert not overlap, f"{a.label} 与 {b.label} 重叠"


def test_static_surfaces_count():
    room = SyntheticRoom(objects=[])
    assert len(room.static_surfaces()) == 6      # 地/顶 + 四墙


# ==========================================================================
# 便捷函数
# ==========================================================================
def test_make_synthetic_frame_has_gt_meta():
    frame = make_synthetic_frame(seed=1, width=80, height=60, num_objects=4)
    assert frame.color.shape == (60, 80, 3)
    assert frame.depth_m.shape == (60, 80)
    assert "boxes_3d" in frame.meta and "labels" in frame.meta
    assert frame.meta["boxes_3d"].shape[1] == 7


def test_make_synthetic_sequence_shares_gt():
    frames = make_synthetic_sequence(seed=2, num_frames=3, width=80, height=60, num_objects=4)
    assert len(frames) == 3
    boxes0 = frames[0].meta["boxes_3d"]
    for f in frames[1:]:
        assert np.allclose(f.meta["boxes_3d"], boxes0)   # 同一房间 → GT 一致
    # 位姿必须不同（否则"多视角"没意义）
    assert not np.allclose(frames[0].pose.t, frames[1].pose.t)


def test_synthetic_depth_respects_max_depth():
    frame = make_synthetic_frame(seed=4, width=80, height=60, num_objects=3)
    assert frame.depth_m.max() <= 12.0
