# -*- coding: utf-8 -*-
"""`data.frame_gt`：单帧（针孔）GT 可见性判定的解析单元测试。

为什么这个模块值得单独测
========================
它是"**用一帧 vs 用多帧融合**"这条消融的**分母**。分母错了，
整条消融的结论就是错的，而且错得很隐蔽（两条臂各自看着都"合理"）。
所以这里不测"跑得通"，只测**能解析手算出来的数**：

1. 光心在原点、光轴 +z、深度图是一张 **z=const 的平面** ——
   这种场景下每个采样点的遮挡关系都能手推。
2. 遮挡判据必须比 **z 深度**，不是比斜距 `r`。
   项目已经在这件事上栽过一次（见 `docs/2D3D-S数据集核验报告.md`：
   同一批像素 `range/z` 最大差 2.86×）。这里放一个**反证**：
   同一个场景若按斜距判，可见率会从 ~0.5 掉到 ~0。
"""
from __future__ import annotations

import numpy as np
import pytest

from roboground.data.frame_gt import (
    box_surface_points,
    box_to_frame,
    visible_gt_in_frame,
)
from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame


# ==========================================================================
# 构造用的小工具
# ==========================================================================
def _frame(depth_value: float, *, W: int = 640, H: int = 480,
           fx: float = 150.0, fy: float = 150.0,
           R: np.ndarray | None = None, t: np.ndarray | None = None,
           ) -> RGBDFrame:
    """一张**常量深度**的针孔帧（depth = z 平面的距离）。

    `fx=150, W=640` → 水平半视场 `atan(320/150) = 64.9°`，
    60° 离轴的物体仍能落在画面里 —— 这正是"z 与 r 差得最多"的区域。
    """
    K = CameraIntrinsics(fx=fx, fy=fy, cx=W / 2.0, cy=H / 2.0, width=W, height=H)
    pose = CameraPose.from_rt(np.eye(3) if R is None else R,
                              np.zeros(3) if t is None else t)
    return RGBDFrame(
        color=np.zeros((H, W, 3), dtype=np.uint8),
        depth_m=np.full((H, W), float(depth_value), dtype=np.float32),
        intrinsics=K, pose=pose, frame_id="synthetic",
        meta={"source": "test"},
    )


def _box(center, size=(1.0, 1.0, 1.0)) -> np.ndarray:
    return np.array([*center, *size, 0.0], dtype=np.float64)


# ==========================================================================
# 1) 采样点：数量与位置
# ==========================================================================
def test_box_surface_points_count_and_on_faces():
    box = _box((1.0, 2.0, 3.0), (0.8, 0.6, 0.4))
    for grid in (2, 3, 6):
        pts = box_surface_points(box, grid=grid)
        assert pts.shape == (6 * grid * grid, 3)
        lo = box[:3] - box[3:6] / 2.0
        hi = box[:3] + box[3:6] / 2.0
        # 每个点至少落在**一个**坐标的边界面上，且三个坐标都在盒内
        on_face = np.isclose(pts, lo[None, :], atol=1e-9) | \
            np.isclose(pts, hi[None, :], atol=1e-9)
        assert bool(on_face.any(axis=1).all())
        assert bool(((pts >= lo[None, :] - 1e-9) & (pts <= hi[None, :] + 1e-9)).all())


def test_box_surface_points_accepts_flat_and_degenerate_input():
    pts = box_surface_points(np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]), grid=4)
    assert pts.shape == (6 * 16, 3)
    assert np.allclose(pts, [0.0, 0.0, 3.0])   # 退化成**一个点**，不是零向量
    assert box_surface_points(_box((0, 0, 3)), grid=1).shape == (6, 3)


# ==========================================================================
# 2) 正前方：可见率能**手算**出来
# ==========================================================================
def test_front_facing_box_visible_fraction_is_analytic():
    """深度平面 z=3.0、盒子 `[-0.5,0.5]³` 中心在 `(0,0,3)`，grid=6。

    采样点共 `6·36 = 216` 个，z 只能取 `linspace(2.5, 3.5, 6)`
    = `{2.5, 2.7, 2.9, 3.1, 3.3, 3.5}`。遮挡判据是
    `obs >= z − max(0.30, 0.10·z)`，而 `obs ≡ 3.0`：

    · z=2.5 → 容差 0.30 → 门限 2.20 ✓
    · z=2.7 → 0.30 → 2.40 ✓   · z=2.9 → 0.30 → 2.60 ✓
    · z=3.1 → 0.31 → 2.79 ✓   · z=3.3 → 0.33 → 2.97 ✓
    · z=3.5 → 0.35 → 3.15 ✗

    即 **5 档 z 可见、1 档不可见**。z 的分布：

    · 近端面 z=2.5：36 个 → 全可见（36）
    · 远端面 z=3.5：36 个 → 全不可见（0）
    · 4 个侧面：每档 z 各 6 个点 → 4 × 5 × 6 = 120

    合计 `(36 + 120) / 216 = 0.7222…`

    —— "正对相机、完全没被挡的盒子"也只有约七成采样点可见，
    因为分母里含背对相机的远端面。这是**口径**，不是缺陷；
    也提醒不要把 `visible_frac` 读成"看到了物体的百分之多少"。
    这里刻意把 `occ_tol_rel` 也算进去：若只算 `occ_tol_abs`，
    z=3.3 那一档会被误判为不可见，手算值就会错成 156/216 之外的数。
    """
    frame = _frame(3.0)
    info = box_to_frame(frame, _box((0.0, 0.0, 3.0), (1.0, 1.0, 1.0)))
    assert info is not None
    assert info["n_samples"] == 216
    assert info["in_frustum_frac"] == pytest.approx(1.0)
    assert info["covered_frac"] == pytest.approx(1.0)
    assert info["visible_frac"] == pytest.approx((36 + 120) / 216.0, abs=1e-9)
    assert info["visible"] is True
    # 斜距恒 ≥ z 深度，且中位 z 深度应正好是盒子中心
    assert info["z_median_m"] == pytest.approx(3.0, abs=1e-6)
    assert info["range_m"] > info["z_median_m"]
    # 投影框落在画面内，且是"图内"的那一个
    u0, v0, u1, v1 = info["uv"]
    assert 0 <= u0 <= u1 < 640 and 0 <= v0 <= v1 < 480
    assert info["uv_parts"] == [info["uv"]]
    assert info["uv_wrapped"] is False


# ==========================================================================
# 3) ★ 遮挡判据是 z 深度，不是斜距（反证）
# ==========================================================================
def test_occlusion_compares_z_depth_not_range():
    """60° 离轴、深度平面 z=1.8、盒子 z∈[1.8,2.2]。

    · 按 **z** 判：近侧面 z=1.8 的观测=1.8 ≥ 1.8−0.30 → 可见；
      远侧面 z=2.2 的观测=1.8 < 2.2−0.30=1.9 → 被挡。
      于是可见率落在 0.3~0.7 之间。
    · 按 **斜距** 判：这些点的 r ≈ 3.7，观测值 1.8 远小于 3.7−0.37 → 全判遮挡，
      可见率 ≈ 0。

    这里把两种口径都算出来对比。断言"z 判据可见、r 判据不可见"，
    等价于断言实现里**没有**把 z 当 r 用。
    """
    frame = _frame(1.8)
    box = _box((3.4641, 0.0, 2.0), (0.4, 0.4, 0.4))     # 中心 r=4.0, z=2.0
    info = box_to_frame(frame, box)
    assert info is not None
    assert info["in_frustum_frac"] == pytest.approx(1.0)
    assert info["visible_frac"] > 0.30, info

    # —— 反证：同一场景按斜距判会得到什么 ——
    pts = box_surface_points(box, grid=6)
    r = np.linalg.norm(pts, axis=1)
    obs = 1.8
    range_based = float((obs >= (r - np.maximum(0.30, 0.10 * r))).mean())
    assert range_based < 0.05, f"斜距判据居然也判可见（{range_based:.3f}）—— 场景没构造好"


# ==========================================================================
# 4) 近处遮挡物 → 全判不可见
# ==========================================================================
def test_nearby_wall_hides_the_box():
    frame = _frame(1.0)                     # 一堵墙在 z=1.0
    info = box_to_frame(frame, _box((0.0, 0.0, 3.0), (1.0, 1.0, 1.0)))
    assert info is not None
    # `covered` 的含义是"这个方向**有有效深度**"，墙把每个方向都填满了 → 1.0
    assert info["covered_frac"] == pytest.approx(1.0)
    # 但观测距离 1.0 远小于盒子自身 z∈[2.5,3.5] → 全部判为被挡
    assert info["visible_frac"] == pytest.approx(0.0)
    assert info["visible"] is False


# ==========================================================================
# 5) 画外 / 背后 / 相机在盒内 / 量程
# ==========================================================================
def test_box_behind_camera_is_not_visible():
    frame = _frame(3.0)
    info = box_to_frame(frame, _box((0.0, 0.0, -3.0), (1.0, 1.0, 1.0)))
    assert info is not None
    assert info["in_frustum_frac"] == pytest.approx(0.0)
    assert info["visible_frac"] == pytest.approx(0.0)
    assert info["uv_parts"] == []


def test_box_outside_horizontal_fov_is_not_visible():
    # 65° 半视场；放到 80° 离轴 → 全部出画
    frame = _frame(3.0)
    c = 4.0 * np.array([np.sin(np.radians(80)), 0.0, np.cos(np.radians(80))])
    info = box_to_frame(frame, _box(c, (0.3, 0.3, 0.3)))
    assert info is not None
    assert info["in_frustum_frac"] == pytest.approx(0.0)
    assert info["visible"] is False


def test_box_at_sixty_degrees_is_inside_fov():
    """对照组：60° 离轴仍在画面内（证明上一条不是"全都出画"）。"""
    frame = _frame(3.0)
    c = 4.0 * np.array([np.sin(np.radians(60)), 0.0, np.cos(np.radians(60))])
    info = box_to_frame(frame, _box(c, (0.3, 0.3, 0.3)))
    assert info is not None
    assert info["in_frustum_frac"] == pytest.approx(1.0)


def test_camera_inside_box_returns_none():
    frame = _frame(3.0)
    assert box_to_frame(frame, _box((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))) is None


def test_max_range_uses_slant_range():
    """量程门限按**斜距**裁：60° 离轴的点 z=2.0 但 r=4.0。

    `max_range_m=3.0` 时这些点应被判"超出量程 → 没观测"，
    即使它们的 z 深度只有 2.0。这一条保证针孔臂与全景臂
    （融合时按斜距裁 `max_depth`）的信息边界一致。
    """
    frame = _frame(2.0)
    box = _box((3.4641, 0.0, 2.0), (0.4, 0.4, 0.4))
    loose = box_to_frame(frame, box, max_range_m=8.0)
    tight = box_to_frame(frame, box, max_range_m=3.0)
    assert loose is not None and tight is not None
    assert loose["visible_frac"] > 0.4
    assert tight["visible_frac"] == pytest.approx(0.0)
    assert tight["visible"] is False


# ==========================================================================
# 6) 位姿真的被用上了（世界系 → 相机系）
# ==========================================================================
def test_pose_yaw_decides_which_box_is_in_view():
    """两个盒子分别在世界 `+z` 与 `+x` 方向 3 m 处，两个位姿各看见一个。

    `R` 是 **world→camera**。`R = I` 时相机沿世界 `+z` 看；
    `R = Ry(−90°)` 把世界 `+x` 映到相机 `+z`，于是相机沿世界 `+x` 看。
    断言"各看见一个"，等价于断言位姿真的参与了投影
    （若把 `R` 忽略掉，两个位姿会给出同一组结果）。
    """
    box_z = _box((0.0, 0.0, 3.0), (0.5, 0.5, 0.5))
    box_x = _box((3.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    a = np.radians(-90.0)
    Ry = np.array([[np.cos(a), 0.0, np.sin(a)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(a), 0.0, np.cos(a)]])

    straight = _frame(3.0)
    yawed = _frame(3.0, R=Ry)

    for frame, near, far, tag in ((straight, box_z, box_x, "R=I"),
                                  (yawed, box_x, box_z, "R=Ry(-90)")):
        i_near = box_to_frame(frame, near)
        i_far = box_to_frame(frame, far)
        assert i_near is not None and i_far is not None
        assert i_near["in_frustum_frac"] == pytest.approx(1.0), tag
        assert i_far["in_frustum_frac"] == pytest.approx(0.0), tag


def test_translation_is_honoured():
    """`t` 是 **world→camera** 平移：`t=(0,0,-1)` 把光心搬到世界 `z=+1`。

    用**退化成一个点**的盒子，斜距才能手算：
    所有采样点都在 `(0,0,3)`，`range_m` 恰好是 3.0 / 2.0 / 4.0。
    若换成有体积的盒子，中位斜距不会整移动 1 m（只有光轴上的点才会），
    —— 那是几何事实，不是实现偏差，所以这里不拿它当断言。

    注意方向：光心是 `−Rᵀt`，`t` 取 `+z` 时相机往 `−z` 跑，距离**变大**。
    """
    dot = _box((0.0, 0.0, 3.0), (0.0, 0.0, 0.0))
    a = box_to_frame(_frame(3.0), dot)
    b = box_to_frame(_frame(3.0, t=np.array([0.0, 0.0, -1.0])), dot)
    c = box_to_frame(_frame(3.0, t=np.array([0.0, 0.0, 1.0])), dot)
    for info, want_r, want_z in ((a, 3.0, 3.0), (b, 2.0, 2.0), (c, 4.0, 4.0)):
        assert info is not None
        assert info["range_m"] == pytest.approx(want_r, abs=1e-9)
        assert info["z_median_m"] == pytest.approx(want_z, abs=1e-9)


# ==========================================================================
# 7) visible_gt_in_frame：下标 / 标签 / 排序 / 门限
# ==========================================================================
def test_visible_gt_in_frame_keeps_input_indices_and_sorts_by_range():
    frame = _frame(6.0)                     # 远墙，两个盒子都在它前面
    boxes = np.stack([
        _box((0.0, 0.0, 5.0), (1.0, 1.0, 1.0)),      # 0 远
        _box((0.0, 0.0, 0.0), (4.0, 4.0, 4.0)),      # 1 相机在盒内 → 丢
        _box((0.0, 0.0, 2.0), (1.0, 1.0, 1.0)),      # 2 近
    ])
    labels = ["far", "wrap", "near"]
    vis = visible_gt_in_frame(frame, boxes, labels, max_range_m=8.0)
    assert [v["index"] for v in vis] == [2, 0]
    assert [v["label"] for v in vis] == ["near", "far"]
    assert all(np.allclose(v["box"], boxes[v["index"]]) for v in vis)


def test_visible_gt_in_frame_threshold_is_monotone():
    """门限越高，入选越少；门限超过实测最大值后必须一个都不剩。

    （`min_visible_frac` 曾经在 `box_to_pano` 里被硬编码成 0.25，
    导致"筛选用一个门限、条目里的 `visible` 用另一个" —— 这里一并守住。）
    """
    frame = _frame(3.0)
    boxes = np.stack([_box((0.0, 0.0, 3.0), (1.0, 1.0, 1.0)),
                      _box((1.8, 0.0, 3.0), (1.0, 1.0, 1.0))])
    labels = ["a", "b"]
    allv = visible_gt_in_frame(frame, boxes, labels,
                               min_visible_frac=0.0, max_range_m=8.0)
    fracs = {v["index"]: v["visible_frac"] for v in allv}
    assert fracs, "门限为 0 时应全部入选，否则后面的单调性断言没有意义"
    top = max(fracs.values())

    counts = [len(visible_gt_in_frame(frame, boxes, labels,
                                      min_visible_frac=t, max_range_m=8.0))
              for t in (0.0, top * 0.5, top, top + 1e-6)]
    assert counts[0] == len(fracs)
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0


def test_visible_gt_in_frame_handles_empty_input():
    frame = _frame(3.0)
    assert visible_gt_in_frame(frame, np.zeros((0, 7)), []) == []
    assert visible_gt_in_frame(frame, np.zeros((7,)), []) == []


def test_boxes_are_injected_within_the_image():
    """注入用的 `uv_parts` 必须落在图像内（`Detection2D` 的硬约束）。"""
    frame = _frame(3.0)
    vis = visible_gt_in_frame(frame, np.stack([_box((0.0, 0.0, 3.0))]),
                              ["a"], max_range_m=8.0)
    assert len(vis) == 1
    for (u0, v0, u1, v1) in vis[0]["uv_parts"]:
        assert 0 <= u0 <= u1 < 640
        assert 0 <= v0 <= v1 < 480
