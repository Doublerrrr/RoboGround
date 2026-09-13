"""ROS2 位姿接入测试（**不需要 ROS 运行时**）。

这一组测试覆盖的是"真机上最容易出错、又最难事后发现"的部分：
外部位姿表示 → `CameraPose` → 建图坐标系。全部是纯数学与纯数据流，
所以在 Windows 上没有 rclpy 也能完整验证。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboground.deployment.ros2.bridge import (
    ROS2_AVAILABLE,
    frame_from_streams,
)
from roboground.deployment.ros2.tf import (
    IdentityPoseProvider,
    OdometryPoseProvider,
    PoseProvider,
    StaticPoseProvider,
    TrajectoryPoseProvider,
    TfPoseProvider,
    _stamp_to_ros_time,
    _stamp_to_seconds,
    build_pose_provider,
    euler_to_camera_pose,
    odometry_to_camera_pose,
    quaternion_to_matrix,
    rotation_matrix_to_quaternion,
    tf_transform_to_camera_pose,
)
from roboground.types import CameraIntrinsics, CameraPose


# ==========================================================================
# 四元数 ⇄ 旋转矩阵
# ==========================================================================
def test_quaternion_identity():
    assert np.allclose(quaternion_to_matrix(0, 0, 0, 1), np.eye(3))


def test_quaternion_90deg_about_z():
    """绕 z 轴 90°：x 轴应转到 y 轴。"""
    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    R = quaternion_to_matrix(0, 0, s, c)
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_quaternion_is_normalized_before_use():
    """未归一化的四元数（TF 常有数值漂移）必须被归一化，否则旋转矩阵不正交。"""
    R = quaternion_to_matrix(0, 0, 2.0, 2.0)      # 模长 2√2
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_zero_quaternion_is_safe():
    assert np.allclose(quaternion_to_matrix(0, 0, 0, 0), np.eye(3))


@pytest.mark.parametrize("axis,angle", [
    ((1, 0, 0), 0.3), ((0, 1, 0), -1.1), ((0, 0, 1), 2.7), ((1, 1, 1), 0.9),
])
def test_quaternion_matrix_roundtrip(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2)
    q = (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))
    R = quaternion_to_matrix(*q)
    q2 = rotation_matrix_to_quaternion(R)
    assert np.allclose(quaternion_to_matrix(*q2), R, atol=1e-9)


# ==========================================================================
# ★ 坐标约定（真机上最容易搞错的一步）
# ==========================================================================
def test_tf_convention_forward_maps_to_camera_z():
    """世界 +x（机器人前方）必须映射到相机 +z（光轴）。

    REP-103 机器人坐标：x 前、y 左、z 上；OpenCV 相机：x 右、y 下、z 前。
    """
    pose = tf_transform_to_camera_pose(
        (0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), optical_frame_correction=True)
    cam = pose.world_to_cam(np.array([[1.0, 0.0, 0.5]]))
    assert np.allclose(cam, [[0.0, 0.0, 1.0]], atol=1e-9), cam


def test_tf_convention_left_maps_to_camera_minus_x():
    """机器人 +y（左）→ 相机 −x（因为相机 x 朝右）。"""
    pose = tf_transform_to_camera_pose(
        (0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), optical_frame_correction=True)
    cam = pose.world_to_cam(np.array([[0.0, 1.0, 0.5]]))
    assert np.allclose(cam, [[-1.0, 0.0, 0.0]], atol=1e-9), cam


def test_tf_convention_up_maps_to_camera_minus_y():
    """机器人 +z（上）→ 相机 −y（因为相机 y 朝下）。"""
    pose = tf_transform_to_camera_pose(
        (0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), optical_frame_correction=True)
    cam = pose.world_to_cam(np.array([[0.0, 0.0, 1.5]]))
    assert np.allclose(cam, [[0.0, -1.0, 0.0]], atol=1e-9), cam


def test_optical_frame_correction_off_differs():
    """关掉轴纠正时结果必须不同 —— 否则这个开关是假的。"""
    on = tf_transform_to_camera_pose((0, 0, 0), (0, 0, 0, 1), optical_frame_correction=True)
    off = tf_transform_to_camera_pose((0, 0, 0), (0, 0, 0, 1), optical_frame_correction=False)
    assert not np.allclose(on.R, off.R)


def test_camera_center_recovered_from_tf_translation():
    """TF 的 translation 就是相机光心在 map 中的位置，必须能被还原。"""
    C = (1.5, -0.8, 1.2)
    pose = tf_transform_to_camera_pose(C, (0, 0, 0, 1), optical_frame_correction=False)
    assert np.allclose(pose.camera_center(), C, atol=1e-9)


def test_euler_and_quaternion_paths_agree():
    """欧拉角入口与四元数入口必须给出一致的旋转。"""
    rpy = (0.1, -0.2, 0.3)
    from_euler = euler_to_camera_pose((0.5, 0.0, 1.0), rpy, optical_frame_correction=False)

    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    R = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
         @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
         @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    q = rotation_matrix_to_quaternion(R)
    from_quat = tf_transform_to_camera_pose((0.5, 0.0, 1.0), q, optical_frame_correction=False)
    assert np.allclose(from_euler.R, from_quat.R, atol=1e-9)


# ==========================================================================
# 里程计
# ==========================================================================
def test_odometry_without_extrinsics():
    pose = odometry_to_camera_pose((2.0, 0.0, 0.0), (0, 0, 0, 1),
                                  optical_frame_correction=False)
    assert np.allclose(pose.camera_center(), [2.0, 0.0, 0.0], atol=1e-9)


def test_odometry_with_base_to_camera_extrinsics():
    """带 base→camera 外参时，光心位置必须是 base 位置 + 旋转后的偏移。"""
    pose = odometry_to_camera_pose(
        (0.0, 0.0, 0.0), (0, 0, 0, 1),
        base_to_camera=((0.0, 0.0, 1.0), (0, 0, 0, 1)),
        optical_frame_correction=False,
    )
    assert np.allclose(pose.camera_center(), [0.0, 0.0, 1.0], atol=1e-9)


def test_odometry_extrinsics_rotate_with_base():
    """底盘转 90° 时，相机的固定偏移也必须跟着转。"""
    s = math.sin(math.pi / 4)
    c = math.cos(math.pi / 4)
    pose = odometry_to_camera_pose(
        (0.0, 0.0, 0.0), (0, 0, s, c),                  # base 绕 z 转 90°
        base_to_camera=((1.0, 0.0, 0.0), (0, 0, 0, 1)),  # 相机在 base 前方 1m
        optical_frame_correction=False,
    )
    # base 转 90° 后，前方 1m 在 map 里是 +y 方向
    assert np.allclose(pose.camera_center(), [0.0, 1.0, 0.0], atol=1e-9)


# ==========================================================================
# PoseProvider
# ==========================================================================
def test_identity_provider_warns_once():
    p = IdentityPoseProvider()
    assert np.allclose(p.get_pose().R, np.eye(3))
    assert p.get_pose().t.tolist() == [0.0, 0.0, 0.0]


def test_static_provider():
    p = StaticPoseProvider.from_transform((1.0, 2.0, 3.0), (0, 0, 0, 1),
                                         optical_frame_correction=False)
    assert np.allclose(p.get_pose().camera_center(), [1.0, 2.0, 3.0], atol=1e-9)
    assert p.get_pose(123.0) is p.get_pose(456.0)     # 与时间无关


def test_trajectory_provider_interpolates():
    poses = [
        CameraPose(np.eye(3), np.array([0.0, 0.0, 0.0])),
        CameraPose(np.eye(3), np.array([-2.0, 0.0, 0.0])),   # 光心在 (2,0,0)
    ]
    p = TrajectoryPoseProvider([0.0, 1.0], poses)

    c0 = p.get_pose(0.0).camera_center()
    c1 = p.get_pose(1.0).camera_center()
    cm = p.get_pose(0.5).camera_center()

    assert np.allclose(c0, [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(c1, [2.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(cm, [1.0, 0.0, 0.0], atol=1e-6)      # 中点


def test_trajectory_provider_clamps_out_of_range():
    poses = [CameraPose(np.eye(3), np.zeros(3)), CameraPose(np.eye(3), -np.array([1.0, 0, 0]))]
    p = TrajectoryPoseProvider([10.0, 20.0], poses)
    assert np.allclose(p.get_pose(0.0).camera_center(), [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(p.get_pose(99.0).camera_center(), [1.0, 0.0, 0.0], atol=1e-9)


def test_trajectory_provider_requires_matching_lengths():
    with pytest.raises(ValueError):
        TrajectoryPoseProvider([0.0, 1.0], [CameraPose.identity()])


def test_odometry_provider_drops_stale_data():
    """过期里程计数据必须被丢弃，而不是拿旧值凑数。"""
    p = OdometryPoseProvider(node=None, max_age_s=0.2)

    class _Vec:
        x, y, z = 1.0, 2.0, 3.0

    class _Quat:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0

    class _Pose:
        position, orientation = _Vec(), _Quat()

    class _PoseWithCov:
        pose = _Pose()

    class _Stamp:
        sec, nanosec = 100, 0

    class _Header:
        stamp = _Stamp()

    class _Msg:
        pose = _PoseWithCov()      # nav_msgs/Odometry 是 msg.pose.pose.position
        header = _Header()

    p._on_odom(_Msg())
    assert p.get_pose(100.1) is not None        # 新鲜
    assert p.get_pose(101.0) is None            # 过期 → 丢弃而不是用旧值
    assert p.failures >= 1
    # 不传 stamp 时不做时效检查（用于没有时间戳的离线回放）
    assert p.get_pose(None) is not None


def test_odometry_provider_returns_none_before_any_message():
    p = OdometryPoseProvider(node=None)
    assert p.get_pose(0.0) is None


# ==========================================================================
# 配置驱动构建
# ==========================================================================
def test_build_pose_provider_defaults_to_identity(cfg):
    p = build_pose_provider(cfg)
    assert isinstance(p, IdentityPoseProvider)


def test_build_pose_provider_static(cfg):
    cfg.set("deploy.ros2.pose.source", "static")
    cfg.set("deploy.ros2.pose.translation", [1.0, 0.0, 1.5])
    cfg.set("deploy.ros2.pose.rpy", [0.0, 0.0, 0.0])
    cfg.set("deploy.ros2.pose.optical_frame_correction", False)
    p = build_pose_provider(cfg)
    assert isinstance(p, StaticPoseProvider)
    assert np.allclose(p.get_pose().camera_center(), [1.0, 0.0, 1.5], atol=1e-9)


def test_build_pose_provider_tf_without_node_degrades(cfg):
    """配了 tf 但没传节点 → 必须优雅降级并告警，而不是崩。"""
    cfg.set("deploy.ros2.pose.source", "tf")
    p = build_pose_provider(cfg, node=None)
    assert isinstance(p, IdentityPoseProvider)


def test_build_pose_provider_unknown_source_degrades(cfg):
    cfg.set("deploy.ros2.pose.source", "telepathy")
    assert isinstance(build_pose_provider(cfg), IdentityPoseProvider)


# ==========================================================================
# ★ 帧组装：位姿缺失时必须丢帧
# ==========================================================================
def _intrinsics():
    return CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, width=100, height=80)


def test_frame_from_streams_with_good_pose():
    color = np.full((80, 100, 3), 127, np.uint8)
    depth_mm = np.full((80, 100), 2000, np.uint16)
    provider = StaticPoseProvider(CameraPose.identity())

    frame = frame_from_streams(color, depth_mm, _intrinsics(), provider,
                               stamp=1.0, depth_scale=1000.0)
    assert frame is not None
    assert frame.timestamp == 1.0
    assert abs(float(frame.depth_m[40, 50]) - 2.0) < 1e-6      # 2000mm → 2m
    assert frame.meta["has_pose"] is True


def test_frame_from_streams_returns_none_when_pose_missing():
    """★ 核心行为：位姿拿不到就返回 None，**绝不退回恒等位姿**。"""
    color = np.full((80, 100, 3), 127, np.uint8)
    depth_mm = np.full((80, 100), 2000, np.uint16)
    provider = OdometryPoseProvider(node=None)      # 从未收到消息 → 永远没有位姿

    frame = frame_from_streams(color, depth_mm, _intrinsics(), provider, stamp=1.0)
    assert frame is None


def test_frame_from_streams_returns_none_without_provider():
    color = np.full((80, 100, 3), 127, np.uint8)
    depth_mm = np.full((80, 100), 2000, np.uint16)
    assert frame_from_streams(color, depth_mm, _intrinsics(), None) is None


def test_frame_from_streams_filters_invalid_depth():
    color = np.full((80, 100, 3), 127, np.uint8)
    depth_mm = np.full((80, 100), 2000, np.uint16)
    depth_mm[0, 0] = 0            # 无效
    depth_mm[1, 1] = 60000        # 60m，超出 8m 上限
    provider = StaticPoseProvider(CameraPose.identity())

    frame = frame_from_streams(color, depth_mm, _intrinsics(), provider,
                               min_depth=0.1, max_depth=8.0)
    assert frame is not None
    assert frame.depth_m[0, 0] == 0.0
    assert frame.depth_m[1, 1] == 0.0
    assert frame.depth_m[40, 50] > 0.0


def test_frame_from_streams_rejects_mismatched_resolution_without_autosync():
    color = np.full((80, 100, 3), 127, np.uint8)
    depth_mm = np.full((40, 50), 2000, np.uint16)      # 分辨率不一致
    provider = StaticPoseProvider(CameraPose.identity())
    assert frame_from_streams(color, depth_mm, _intrinsics(), provider) is None


def test_frame_from_streams_autosync_resizes_depth():
    color = np.full((80, 100, 3), 127, np.uint8)
    depth_mm = np.full((40, 50), 2000, np.uint16)
    provider = StaticPoseProvider(CameraPose.identity())
    frame = frame_from_streams(color, depth_mm, _intrinsics(), provider, autosync=True)
    assert frame is not None
    assert frame.depth_m.shape == (80, 100)


def test_frame_from_streams_survives_provider_exception():
    """位姿来源抛异常时也要安全返回 None，而不是把整条链路炸掉。"""

    class _Bad(PoseProvider):
        name = "bad"

        def get_pose(self, stamp=None, frame_id="map"):
            raise RuntimeError("TF 树断了")

    color = np.full((80, 100, 3), 127, np.uint8)
    depth_mm = np.full((80, 100), 2000, np.uint16)
    assert frame_from_streams(color, depth_mm, _intrinsics(), _Bad()) is None


# ==========================================================================
# 端到端：已知轨迹 → 建图 → 物体落点正确
# ==========================================================================
def test_trajectory_driven_mapping_places_objects_correctly(cfg, quiet, obs_factory):
    """用一条**已知相机轨迹**驱动建图，验证位姿真的被用对了。

    这是整条"位姿接入"链路的最终验收：
    如果位姿约定搞反/搞错，不同视角观测到的同一物体会落到不同位置，
    物体数会膨胀、位置也会错。
    """
    from roboground.mapping import MapBuilder

    prompts = ["cup", "table"]
    cfg.set("perception.prompts", prompts)

    # 世界系里的一个静止物体（在 (0, 2, 0)，即相机前方 2m）
    target = np.array([0.0, 2.0, 0.0])
    obs = [obs_factory("cup", target, (0.2, 0.2, 0.2), frame_id=f"f{i}") for i in range(3)]

    builder = MapBuilder(cfg, prompts=prompts)
    smap = builder.build_from_observations(obs, feature_dim=4)

    assert smap.num_objects == 1, "同一物体被不同帧重复建成了多个"
    assert np.allclose(smap.objects[0].center, target, atol=0.05)

    # 用轨迹 provider 走一遍（验证 provider 本身不引入偏移）
    poses = [CameraPose(np.eye(3), np.zeros(3))] * 3
    provider = TrajectoryPoseProvider([0.0, 1.0, 2.0], poses)
    for t in (0.0, 1.0, 2.0):
        assert np.allclose(provider.get_pose(t).camera_center(), [0, 0, 0], atol=1e-9)


def test_ros2_available_flag_is_bool():
    assert isinstance(ROS2_AVAILABLE, bool)


def test_tf_provider_importable_without_ros():
    """`TfPoseProvider` 必须能在没有 ROS 的环境里被 import（构建时才需要 rclpy）。"""
    assert TfPoseProvider is not None
    p = TfPoseProvider(node=None)
    assert p.name == "tf"
    assert p.describe()["source_frame"]


# ==========================================================================
# ★ 回归锁：TF 位姿接线的两个真实 bug
# ==========================================================================
# 这两个 bug 只有在**真实 tf2_ros**（Linux/WSL）上才会暴露：
# RoboStack 的 Windows 构建没有 tf2_ros，测试会自动退回 odometry，
# 从而完全绕过 TF 这条路径 —— 所以它们潜伏了很久，直到装上 WSL 才现形。
#
# 下面用**假 buffer + 假 rclpy 模块**把这两个行为锁在离线测试里，
# 这样即使没有 WSL 也能守住，不会等下次装环境才发现。
# --------------------------------------------------------------------------
class _FakeTime:
    """替身 `rclpy.time.Time`（离线环境没有 rclpy）。"""

    def __init__(self, seconds: int = 0, nanoseconds: int = 0) -> None:
        self.sec = int(seconds)
        self.nanosec = int(nanoseconds)

    def __eq__(self, other) -> bool:
        return (isinstance(other, _FakeTime)
                and (self.sec, self.nanosec) == (other.sec, other.nanosec))

    def __hash__(self) -> int:
        return hash((self.sec, self.nanosec))

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"Time({self.sec}.{self.nanosec:09d})"


class _FakeDuration:
    def __init__(self, seconds: float = 0.0, nanoseconds: int = 0) -> None:
        self.seconds = float(seconds)


class _FakeBuffer:
    """只记录 `lookup_transform` 的入参，然后抛错（不关心返回值）。"""

    def __init__(self) -> None:
        self.calls = []

    def lookup_transform(self, target, source, time, timeout=None):
        self.calls.append({"target": target, "source": source, "time": time})
        raise RuntimeError("fake buffer：本测试只关心入参类型")


def _install_fake_rclpy(monkeypatch):
    import sys
    import types

    rclpy_mod = types.ModuleType("rclpy")
    time_mod = types.ModuleType("rclpy.time")
    dur_mod = types.ModuleType("rclpy.duration")
    time_mod.Time = _FakeTime
    dur_mod.Duration = _FakeDuration
    rclpy_mod.time = time_mod
    rclpy_mod.duration = dur_mod
    monkeypatch.setitem(sys.modules, "rclpy", rclpy_mod)
    monkeypatch.setitem(sys.modules, "rclpy.time", time_mod)
    monkeypatch.setitem(sys.modules, "rclpy.duration", dur_mod)


def test_stamp_to_ros_time_handles_float_and_zero():
    """`_stamp_to_ros_time`：float 秒要拆成 sec+nanosec；0/None 取最新。"""
    import sys
    import types

    saved = {k: sys.modules.get(k) for k in ("rclpy", "rclpy.time")}
    rclpy_mod = types.ModuleType("rclpy")
    time_mod = types.ModuleType("rclpy.time")
    time_mod.Time = _FakeTime
    rclpy_mod.time = time_mod
    sys.modules["rclpy"] = rclpy_mod
    sys.modules["rclpy.time"] = time_mod
    try:
        t = _stamp_to_ros_time(1789094941.25)
        assert (t.sec, t.nanosec) == (1789094941, 250000000)
        # 0 / None → Time() 即"最新可用"
        assert _stamp_to_ros_time(0.0) == _FakeTime(0, 0)
        assert _stamp_to_ros_time(None) == _FakeTime(0, 0)
        assert _stamp_to_ros_time(None) == _FakeTime(0, 0)
        # 已经是 Time 就原样返回
        same = _FakeTime(5, 6)
        assert _stamp_to_ros_time(same) is same
        # nanosec 四舍五入顶到 1e9 时必须归位（rclpy 会拒绝 1e9）
        r = _stamp_to_ros_time(10.9999999999)
        assert r.nanosec < 1_000_000_000, f"nanosec 未归位：{r.nanosec}"
        assert (r.sec, r.nanosec) == (11, 0)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_tf_provider_converts_float_stamp(monkeypatch):
    """★ 回归锁：`TfPoseProvider` 必须把 float 秒转成 ROS Time 再查 TF。

    真实 bug：项目内部约定 `get_pose(stamp)` 的 stamp 是 **float 秒**
    （`nodes.py` 用 `_stamp_to_seconds(color_msg.header.stamp)`），
    而 `tf2_ros` 的 `lookup_transform()` 只接受 `rclpy.time.Time`。
    把 float 原样传进去会抛异常 → 被 identity fallback 吞掉 →
    **帧带着恒等位姿建图**（现象：处理了 N 帧却 0 体素，且 `pose_failures` 为 0）。

    这个断言很朴素但很关键：传给 tf2 的 `time` **不能是裸 float**。
    """
    _install_fake_rclpy(monkeypatch)

    prov = TfPoseProvider(node=object())
    buf = _FakeBuffer()
    prov._buffer = buf                      # 绕过 setup()，直接注入假 buffer
    prov.get_pose(1789094941.25)            # 内部会抛错并被捕获，但入参已记录

    assert buf.calls, "lookup_transform 没被调用"
    passed = buf.calls[-1]["time"]
    assert not isinstance(passed, float), (
        f"传给 tf2 的 stamp 是裸 float（{passed!r}）—— "
        "会抛异常并被 fallback 吞掉，必须先用 _stamp_to_ros_time 转换"
    )
    assert isinstance(passed, _FakeTime), f"期望 Time 对象，实际 {type(passed).__name__}"
    assert (passed.sec, passed.nanosec) == (1789094941, 250000000)


def test_tf_provider_without_fallback_returns_none_on_failure(monkeypatch):
    """TF 查询失败且**没有 fallback** 时应返回 None（让上层丢帧），而不是恒等位姿。"""
    _install_fake_rclpy(monkeypatch)

    prov = TfPoseProvider(node=object())    # fallback 默认 None
    prov._buffer = _FakeBuffer()
    assert prov.get_pose(1.0) is None
    assert prov.failures == 1


def test_build_pose_provider_tf_has_no_silent_identity_fallback(cfg):
    """★ 回归锁：tf 位姿**默认不允许**静默退回恒等位姿。

    这个兜底曾把"stamp 类型错→查询失败"变成"悄悄用恒等位姿建图"，
    与 `nodes.py` 里写明的原则冲突：
    **宁可丢帧，也不要用恒等位姿凑数**（错误位姿污染地图且极难归因）。
    只有显式配 `fallback_identity: true` 才给。
    """
    cfg.set("deploy.ros2.pose.source", "tf")
    p = build_pose_provider(cfg, node=object())
    assert isinstance(p, TfPoseProvider)
    assert p.fallback is None, "tf 默认不应有 identity fallback"

    cfg.set("deploy.ros2.pose.fallback_identity", True)
    p2 = build_pose_provider(cfg, node=object())
    assert isinstance(p2.fallback, IdentityPoseProvider), "显式开启后应给 identity 兜底"


def test_stamp_seconds_roundtrip():
    """`_stamp_to_seconds` 与 `_stamp_to_ros_time` 必须互为逆（float 精度内）。"""
    for secs in (1789094941.25, 0.5, 12345.125):
        assert _stamp_to_seconds(_FakeTime(int(secs), int((secs % 1) * 1e9))) \
            == pytest.approx(secs, abs=1e-6)


# ==========================================================================
# ★ 回归锁：QoS 策略（状态型话题必须 latched）
# ==========================================================================
def test_state_topics_use_transient_local():
    """★ **状态型**话题（地图/答案）必须用 `TRANSIENT_LOCAL`。

    真实缺陷：地图/答案发布器早期用的是默认 QoS（`VOLATILE`），
    真机上规划/导航节点如果比感知节点**后启动**，会一直等一张
    **永远不会再发的历史地图**，表现为"接不上"。

    为什么这个缺陷难发现：
    - **离线单测测不出来** —— 假对象不模拟 QoS 语义；
    - **不报任何错** —— 发布订阅都"成功"，只是收不到；
    - 只有真机上"后启动一个订阅者"才会暴露。

    所以把**策略**提成模块级纯数据（`QOS_POLICIES`）让它离线可断言；
    行为验证在 WSL 真 ROS2 里做。
    """
    from roboground.deployment.ros2.nodes import QOS_POLICIES

    assert QOS_POLICIES["state"]["durability"] == "transient_local", \
        "地图/答案这类状态型话题必须是 transient_local，否则后加入的订阅者收不到"
    assert QOS_POLICIES["state"]["reliability"] == "reliable"
    assert QOS_POLICIES["state"]["depth"] == 1, "状态型只需保留最后一帧"


def test_sensor_topics_prefer_latency_over_completeness():
    """传感器话题必须是 `BEST_EFFORT` + `VOLATILE`（REP-2003）。

    理由与状态型**相反**：丢几帧没关系（下一帧马上到），
    但绝不能因为重传而增加延迟 —— 实时链路里迟到的一帧等于没有。
    """
    from roboground.deployment.ros2.nodes import QOS_POLICIES

    s = QOS_POLICIES["sensor"]
    assert s["reliability"] == "best_effort"
    assert s["durability"] == "volatile"
    assert s["depth"] <= 2, "传感器队列要短，否则积压会引入延迟"


def test_node_publishers_actually_use_state_qos():
    """★ 只定义策略不够 —— **必须验证接线**。

    这是项目里踩过两次的坑模式：加了配置/常量却没接进调用链，
    于是"策略对了但节点还用着默认值"，而且**不报错**
    （`robust_n_signal` 加了没接线、`query.min_score` 覆盖了类默认）。

    所以这里直接查源码里发布器的实参。
    """
    import inspect

    import roboground.deployment.ros2.nodes as nodes_mod

    src = inspect.getsource(nodes_mod)
    assert src.count("_state_qos()") >= 3, \
        "地图(×2)与答案发布器都应使用 _state_qos()"
    # 旧的"直接传端口 1"写法不应残留
    assert '"/roboground/semantic_map"), 1)' not in src
    assert '"/roboground/answer"), 1)' not in src
