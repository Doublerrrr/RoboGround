"""位姿来源抽象：把"相机在世界哪里"从 ROS 里解耦出来。

为什么单独抽一层
---------------
`PerceptionNode` 原来把 `_pose` 留空、退化成 `CameraPose.identity()`，
这会让地图变成"以第一帧相机为原点"的**局部地图**，在真机上是不能用的。

但直接把 `tf2_ros` 调用写进节点里有两个问题：
1. **无法测试** —— 没有 ROS2 就一行都跑不了；
2. **刚性** —— 真机上位姿可能来自 TF、里程计、SLAM、或者固定安装位。

所以这里定义 `PoseProvider` 接口，并给出四种实现：

| Provider | 位姿来源 | 适用 |
|---|---|---|
| `TfPoseProvider` | `tf2` 查 `map → camera_link` | 有完整 TF 树的机器人（推荐） |
| `OdometryPoseProvider` | `nav_msgs/Odometry` | 只有轮式/视觉里程计 |
| `StaticPoseProvider` | 固定变换 | 相机固定安装、或离线回放 |
| `IdentityPoseProvider` | 恒等 | 占位/单元测试（**真机禁用**） |

**所有把外部数据转成 `CameraPose` 的数学都放在本文件里**，
它们不依赖 ROS，因此可以被完整单元测试覆盖。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from roboground.types import CameraPose
from roboground.utils.logging import get_logger

logger = get_logger("deployment.ros2.pose")


# ==========================================================================
# 数学：外部表示 → CameraPose
# ==========================================================================
#: 光学系 → 机器人相机架（camera_link）的轴纠正矩阵。
#:
#: 推导（这是真机上最容易搞反的一步，所以把推导写在代码里）：
#: - 光学系 x（右） = camera_link −y（因为 link 的 +y 是"左"）
#: - 光学系 y（下） = camera_link −z（因为 link 的 +z 是"上"）
#: - 光学系 z（前） = camera_link +x（因为 link 的 +x 是"前"）
#:
#: 即 `p_link = R_OPT @ p_optical`：
#: ```
#: R_OPT @ [1,0,0] = [0,-1,0]   （右 → −左）
#: R_OPT @ [0,1,0] = [0,0,-1]   （下 → −上）
#: R_OPT @ [0,0,1] = [1,0,0]    （前 → +前）
#: ```
#: ⚠️ 它的转置是**错的**（会把"右"映射成"上"）。
#: `tests/test_ros2_pose.py` 里有三个测试专门守这条约定。
R_OPTICAL_TO_LINK = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """四元数 (x, y, z, w) → 3×3 旋转矩阵。

    ROS 的 `geometry_msgs/Quaternion` 就是这个顺序（**w 在最后**）。
    输入会先做归一化 —— 真机上 TF 传来的四元数常有轻微数值漂移，
    不归一化会让旋转矩阵不再正交，误差随层级累积。
    """
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3×3 旋转矩阵 → 四元数 (x, y, z, w)。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def tf_transform_to_camera_pose(
    translation: Sequence[float],
    rotation_quaternion: Sequence[float],
    *,
    optical_frame_correction: bool = True,
) -> CameraPose:
    """把 TF 的 `map → camera` 变换转成 `CameraPose`（world → camera）。

    Parameters
    ----------
    translation
        TF 里的平移 `(x, y, z)`，含义是 **camera 原点在 map 中的位置 C**。
    rotation_quaternion
        TF 里的旋转四元数 `(x, y, z, w)`，含义是**把 camera 系旋转到 map 系**。
    optical_frame_correction
        是否做"机器人相机架 → 光学系"的轴纠正。

        这是**真机上最容易搞错的一步**：
        - ROS 的机器人坐标约定是 **x 前、y 左、z 上**（REP-103）；
        - 而 OpenCV/本项目相机约定是 **x 右、y 下、z 前（光轴）**。

        两者差一个固定旋转。如果 TF 树里已经发布了 `*_optical_frame`
        （绝大多数相机驱动都会发布，例如 `camera_color_optical_frame`），
        就应该直接查那个 frame，并把本参数设为 `False`。

        `True` 时套用的纠正矩阵是 `R_opt`：
        ```
        cam_x(右)  = -robot_y
        cam_y(下)  = -robot_z
        cam_z(前)  =  robot_x
        ```
    """
    C = np.asarray(translation, dtype=np.float64).reshape(3)
    x, y, z, w = (float(v) for v in rotation_quaternion)
    R_cam_to_map = quaternion_to_matrix(x, y, z, w)      # camera → map

    if optical_frame_correction:
        # robot(x前,y左,z上) → camera(x右,y下,z前)
        R_cam_to_map = R_cam_to_map @ R_OPTICAL_TO_LINK

    # CameraPose 存的是 world → camera：p_cam = R (p_map - C)
    R_map_to_cam = R_cam_to_map.T
    t = -R_map_to_cam @ C
    return CameraPose(R_map_to_cam, t)


def odometry_to_camera_pose(
    position: Sequence[float],
    orientation_quaternion: Sequence[float],
    *,
    base_to_camera: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    optical_frame_correction: bool = True,
) -> CameraPose:
    """`nav_msgs/Odometry` 的位姿 → `CameraPose`。

    Parameters
    ----------
    position, orientation_quaternion
        里程计给出的 **base_link 在 map 中的位姿**。
    base_to_camera
        可选的 `(translation, quaternion)` 外参（base_link → camera）。
        没有它就只能假设相机与底盘重合 —— 那会引入一个固定偏移，
        对建图是可接受的（整体平移不影响相对几何），但不该忽略。
    """
    C_base = np.asarray(position, dtype=np.float64).reshape(3)
    bx, by, bz, bw = (float(v) for v in orientation_quaternion)
    R_base_to_map = quaternion_to_matrix(bx, by, bz, bw)

    if base_to_camera is not None:
        t_bc = np.asarray(base_to_camera[0], dtype=np.float64).reshape(3)
        qx, qy, qz, qw = (float(v) for v in base_to_camera[1])
        R_cam_to_base = quaternion_to_matrix(qx, qy, qz, qw)
        C_cam = C_base + R_base_to_map @ t_bc
        R_cam_to_map = R_base_to_map @ R_cam_to_base
    else:
        C_cam = C_base
        R_cam_to_map = R_base_to_map

    if optical_frame_correction:
        R_cam_to_map = R_cam_to_map @ R_OPTICAL_TO_LINK

    R_map_to_cam = R_cam_to_map.T
    return CameraPose(R_map_to_cam, -R_map_to_cam @ C_cam)


def euler_to_camera_pose(
    translation: Sequence[float],
    rpy: Sequence[float],
    *,
    optical_frame_correction: bool = True,
) -> CameraPose:
    """用欧拉角 (roll, pitch, yaw) 描述旋转的便捷入口（手写标定时常用）。"""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
    return tf_transform_to_camera_pose(
        translation, (qx, qy, qz, qw), optical_frame_correction=optical_frame_correction
    )


# ==========================================================================
# Provider 接口
# ==========================================================================
class PoseProvider:
    """位姿来源的统一接口。"""

    name = "pose_provider"

    def get_pose(self, stamp: Any = None, frame_id: str = "map") -> Optional[CameraPose]:
        """获取该时刻的相机位姿；拿不到就返回 None（**不要造一个假位姿**）。

        Parameters
        ----------
        stamp
            ROS 时间戳（或秒数）。TF 必须按时间戳查询 —— 用"最新"会导致
            位姿与图像不同步，表现为建图时的"重影"。
        frame_id
            目标坐标系（通常 `map` 或 `odom`）。
        """
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}


class IdentityPoseProvider(PoseProvider):
    """恒等位姿（**仅用于占位与单元测试，真机禁用**）。"""

    name = "identity"

    def __init__(self, *, warn: bool = True) -> None:
        self._warned = False
        self._warn = warn

    def get_pose(self, stamp=None, frame_id="map") -> CameraPose:
        if self._warn and not self._warned:
            logger.warn(
                "使用 IdentityPoseProvider：所有帧的相机位姿都是恒等变换。"
                "建出来的地图是**以第一帧相机为原点的局部地图**，真机上不可用。"
            )
            self._warned = True
        return CameraPose.identity()


class StaticPoseProvider(PoseProvider):
    """固定位姿（相机固定安装，或离线回放已知轨迹）。"""

    name = "static"

    def __init__(self, pose: CameraPose) -> None:
        self.pose = pose

    @classmethod
    def from_transform(cls, translation, quaternion, *, optical_frame_correction=True):
        return cls(tf_transform_to_camera_pose(
            translation, quaternion, optical_frame_correction=optical_frame_correction))

    @classmethod
    def from_euler(cls, translation, rpy, *, optical_frame_correction=True):
        return cls(euler_to_camera_pose(
            translation, rpy, optical_frame_correction=optical_frame_correction))

    def get_pose(self, stamp=None, frame_id="map") -> CameraPose:
        return self.pose


class TrajectoryPoseProvider(PoseProvider):
    """按时间戳在一段已知轨迹上插值（离线回放 / 仿真用）。

    真机上不用它，但它是**验证"位姿接入是否正确"的最佳工具**：
    喂一条已知轨迹，就能检查建出来的地图是否与几何真值一致。
    """

    name = "trajectory"

    def __init__(self, stamps: Sequence[float], poses: Sequence[CameraPose]) -> None:
        if len(stamps) != len(poses) or len(stamps) == 0:
            raise ValueError("stamps 与 poses 必须等长且非空")
        self.stamps = np.asarray(stamps, dtype=np.float64)
        self.poses = list(poses)
        order = np.argsort(self.stamps)
        self.stamps = self.stamps[order]
        self.poses = [self.poses[int(i)] for i in order]

    def get_pose(self, stamp: Any = None, frame_id: str = "map") -> Optional[CameraPose]:
        if stamp is None:
            return self.poses[-1]
        t = float(stamp)
        if t <= self.stamps[0]:
            return self.poses[0]
        if t >= self.stamps[-1]:
            return self.poses[-1]
        i = int(np.searchsorted(self.stamps, t))
        t0, t1 = self.stamps[i - 1], self.stamps[i]
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)

        # 平移线性插值 + 旋转用四元数 slerp 的简化实现（小角度下线性近似足够）
        p0, p1 = self.poses[i - 1], self.poses[i]
        q0 = rotation_matrix_to_quaternion(p0.R)
        q1 = rotation_matrix_to_quaternion(p1.R)
        q = _slerp(q0, q1, alpha)
        C0 = p0.camera_center()
        C1 = p1.camera_center()
        C = (1 - alpha) * C0 + alpha * C1
        return tf_transform_to_camera_pose(C, q, optical_frame_correction=False)


def _slerp(q0, q1, alpha: float):
    """四元数球面线性插值（含最短路径与符号处理）。"""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(a, b))
    if dot < 0.0:                      # 走最短路径
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:                   # 几乎重合 → 线性插值即可
        out = a + alpha * (b - a)
    else:
        theta0 = math.acos(dot)
        theta = theta0 * alpha
        s0 = math.sin(theta0 - theta) / math.sin(theta0)
        s1 = math.sin(theta) / math.sin(theta0)
        out = s0 * a + s1 * b
    n = float(np.linalg.norm(out))
    return tuple((out / n).tolist()) if n > 1e-12 else tuple(a.tolist())


# ==========================================================================
# ROS2 实现（需要 rclpy，运行时才 import）
# ==========================================================================
class TfPoseProvider(PoseProvider):
    """从 TF 树查询 `map → camera_frame`。

    真机推荐用法：

    ```python
    provider = TfPoseProvider(node, target_frame="map",
                             source_frame="camera_color_optical_frame")
    provider.setup()                 # 创建 Buffer / TransformListener
    pose = provider.get_pose(stamp)  # 按图像时间戳查询（**不要用最新**）
    ```

    ⚠️ 两个真机必踩的坑（已在实现里处理）：
    1. **必须按时间戳查询**。用 `lookup_transform(..., Time())`（最新）会让位姿
       与图像不同步，表现为建图"重影"。所以接口强制要求传 `stamp`。
    2. **帧名要用 `*_optical_frame`**。大多数相机驱动会发布它，
       直接查它就不用做轴纠正（`optical_frame_correction=False`）。
       如果只能查到 `camera_link`，则要打开纠正。
    """

    name = "tf"

    def __init__(
        self,
        node: Any,
        *,
        target_frame: str = "map",
        source_frame: str = "camera_color_optical_frame",
        timeout_s: float = 0.1,
        optical_frame_correction: bool = False,
        fallback: Optional[PoseProvider] = None,
    ) -> None:
        self.node = node
        self.target_frame = target_frame
        self.source_frame = source_frame
        self.timeout_s = float(timeout_s)
        self.optical_frame_correction = bool(optical_frame_correction)
        self.fallback = fallback
        self._buffer = None
        self._listener = None
        self.failures = 0

    def setup(self) -> None:
        """创建 TF Buffer 与 Listener（需要 rclpy 已初始化）。"""
        from tf2_ros import Buffer, TransformListener  # noqa: PLC0415

        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self.node)
        # ⚠️ 这里**只说明"监听器建好了"，不等于"查得到"**。
        # 曾经这句话是"TF 已就绪"，读日志的人会以为位姿可用，
        # 而实际上 /tf 可能一条都还没收到（真机启动顺序很常见），
        # 于是前几帧被静默丢弃。明确的措辞 + `wait_ready` 才是对的。
        logger.info(
            f"TF 监听器已创建：{self.target_frame} → {self.source_frame}"
            "（等待 /tf 数据；配合 wait_ready 可避免首帧被丢）")

    def can_transform(self) -> bool:
        """当前是否**真的**能查到这条变换（启动期就绪判断用）。"""
        if self._buffer is None:
            return False
        try:
            return bool(self._buffer.can_transform(
                self.target_frame, self.source_frame, _ros_time_now(self.node)))
        except Exception:
            return False

    def wait_ready(self, timeout_s: float = 5.0, *, spin_once=None,
                   poll_s: float = 0.05) -> bool:
        """等 TF 数据真正可用；返回是否就绪。

        Parameters
        ----------
        timeout_s
            最长等待时间（秒）。真机启动顺序经常是"感知节点先起、TF 树后到"，
            不等一下就开跑会把开头若干帧全丢掉（而丢帧是**静默**的，
            只是地图少几帧观测）。
        spin_once
            处理一次回调的函数（例如 `lambda d: rclpy.spin_once(node, timeout_sec=d)`）。
            TF 数据靠订阅回调写入 buffer，所以等待期间必须让节点转起来。
            不在构造期（还没 start spin）时传它即可。
        """
        if self._buffer is None:
            self.setup()
        deadline = time.time() + max(0.0, float(timeout_s))
        while True:
            if self.can_transform():
                logger.info(
                    f"TF 已就绪：{self.target_frame} → {self.source_frame} 可查")
                return True
            if time.time() >= deadline:
                logger.warn(
                    f"等待 {timeout_s:.1f}s 后仍查不到 "
                    f"{self.target_frame} → {self.source_frame}；"
                    "后续帧若位姿查不到会被**丢弃**（不会伪造位姿）。"
                    "检查：TF 树里有没有这两个 frame、启动顺序、以及 "
                    "deploy.ros2.pose.ready_timeout_s")
                return False
            if spin_once is not None:
                try:
                    spin_once(poll_s)
                    continue
                except Exception:      # pragma: no cover - 具体实现相关
                    pass
            time.sleep(poll_s)

    def get_pose(self, stamp: Any = None, frame_id: Optional[str] = None) -> Optional[CameraPose]:
        if self._buffer is None:
            self.setup()
        if stamp is None:
            logger.warn("TfPoseProvider 需要图像时间戳；缺失时退回最新变换（可能导致重影）")
            stamp = _ros_time_now(self.node)

        try:
            tf_msg = self._buffer.lookup_transform(
                frame_id or self.target_frame, self.source_frame,
                # ⚠️ 必须转换：内部 stamp 是 float 秒，tf2 要的是 Time 消息。
                # 直接传 float 会抛异常 → 被下面的 fallback 吞掉 → 恒等位姿建图。
                _stamp_to_ros_time(stamp),
                timeout=_duration(self.timeout_s),
            )
        except Exception as exc:
            self.failures += 1
            logger.debug(f"TF 查询失败（{exc}）")
            return self.fallback.get_pose(stamp) if self.fallback else None

        tr = tf_msg.transform.translation
        rot = tf_msg.transform.rotation
        return tf_transform_to_camera_pose(
            (tr.x, tr.y, tr.z), (rot.x, rot.y, rot.z, rot.w),
            optical_frame_correction=self.optical_frame_correction,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name, "target_frame": self.target_frame,
            "source_frame": self.source_frame, "failures": self.failures,
        }


class OdometryPoseProvider(PoseProvider):
    """订阅 `nav_msgs/Odometry`，缓存最新位姿。"""

    name = "odometry"

    def __init__(
        self,
        node: Any,
        *,
        topic: str = "/odom",
        base_to_camera: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        optical_frame_correction: bool = True,
        max_age_s: float = 0.5,
    ) -> None:
        self.node = node
        self.topic = topic
        self.base_to_camera = base_to_camera
        self.optical_frame_correction = bool(optical_frame_correction)
        self.max_age_s = float(max_age_s)
        self._latest = None
        self._stamp = None
        self.failures = 0

    def setup(self) -> None:
        from nav_msgs.msg import Odometry  # noqa: PLC0415

        self.node.create_subscription(Odometry, self.topic, self._on_odom, 10)
        logger.info(f"里程计已订阅：{self.topic}（等第一条 /odom 到达前会丢帧）")

    def can_transform(self) -> bool:
        """是否已经收到过一条里程计（启动期就绪判断用）。"""
        return self._latest is not None

    def wait_ready(self, timeout_s: float = 5.0, *, spin_once=None,
                   poll_s: float = 0.05) -> bool:
        """等第一条 `/odom` 到达；返回是否就绪。语义同 `TfPoseProvider.wait_ready`。"""
        deadline = time.time() + max(0.0, float(timeout_s))
        while True:
            if self.can_transform():
                logger.info(f"里程计已就绪：已收到 {self.topic} 的数据")
                return True
            if time.time() >= deadline:
                logger.warn(
                    f"等待 {timeout_s:.1f}s 后仍没收到 {self.topic}；"
                    "后续帧会因位姿不可用被**丢弃**（不会伪造位姿）。"
                    "检查：话题名 `deploy.ros2.pose.odom_topic` 与启动顺序")
                return False
            if spin_once is not None:
                try:
                    spin_once(poll_s)
                    continue
                except Exception:      # pragma: no cover
                    pass
            time.sleep(poll_s)

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # 注意：这里**不立刻转 CameraPose**，因为 stamp 可能过期；
        # 转换放到 get_pose 里，保证"拿到的是被查询时刻的位姿"
        self._latest = ((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        self._stamp = _stamp_to_seconds(msg.header.stamp)

    def get_pose(self, stamp: Any = None, frame_id: str = "map") -> Optional[CameraPose]:
        if self._latest is None:
            self.failures += 1
            return None
        if stamp is not None and self._stamp is not None:
            age = abs(float(stamp) - self._stamp)
            if age > self.max_age_s:
                self.failures += 1
                logger.debug(f"里程计数据过期（{age:.3f}s > {self.max_age_s}s），丢弃")
                return None
        return odometry_to_camera_pose(
            self._latest[0], self._latest[1],
            base_to_camera=self.base_to_camera,
            optical_frame_correction=self.optical_frame_correction,
        )

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "topic": self.topic, "failures": self.failures}


# ==========================================================================
# 小工具（把 rclpy 调用集中在这里，方便 mock）
# ==========================================================================
def _ros_time_now(node: Any):
    try:
        return node.get_clock().now().to_msg()
    except Exception:
        return None


def _duration(seconds: float):
    from rclpy.duration import Duration  # noqa: PLC0415

    return Duration(seconds=float(seconds))


def _stamp_to_seconds(stamp: Any) -> Optional[float]:
    if stamp is None:
        return None
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        try:
            return float(stamp)
        except Exception:
            return None


def _stamp_to_ros_time(stamp: Any):
    """把内部约定的 stamp 转成 **tf2 需要的 `rclpy.time.Time`**。

    为什么必须转（这是实测踩出来的 bug）
    ------------------------------------
    项目内部 `PoseProvider.get_pose(stamp)` 的 stamp 是 **float 秒** ——
    `nodes.py` 里写的是 `stamp = _stamp_to_seconds(color_msg.header.stamp)`，
    `TrajectoryPoseProvider` 也是直接 `float(stamp)`。

    但 `tf2_ros.Buffer.lookup_transform()` 只接受 `rclpy.time.Time`。
    把 float 原样传进去会**抛异常**，而 `TfPoseProvider` 又配了
    `IdentityPoseProvider` 兜底 —— 于是异常被静默吞掉、**退回恒等位姿**，
    帧就带着错误位姿建图（表现为"处理了 N 帧但 0 体素"）。

    症状之所以隐蔽，是因为 `pose_failures` 计数为 0：
    位姿"拿到了"（其实是恒等阵），只是完全不对。

    接受三种输入，因为调用方来源不一：
    - `None` 或 0 → 取最新可用变换（tf2 里 `Time()` 即"latest"）
    - 已有 `.sec/.nanosec` 的 ROS 时间消息 → 直接用
    - float/int 秒 → 拆成 sec + nanosec
    """
    from rclpy.time import Time  # noqa: PLC0415

    if stamp is None:
        return Time()
    if isinstance(stamp, Time):
        return stamp
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return Time(seconds=int(stamp.sec), nanoseconds=int(stamp.nanosec))
    secs = float(stamp)
    if not (secs > 0.0):          # 0 / NaN / 负数 → 没有有效时间戳
        return Time()
    whole = int(secs)
    nanos = int(round((secs - whole) * 1e9))
    # 四舍五入可能把 nanos 顶到 1e9，rclpy 会拒绝，这里归位
    if nanos >= 1_000_000_000:
        whole += 1
        nanos -= 1_000_000_000
    return Time(seconds=whole, nanoseconds=nanos)


def build_pose_provider(cfg, node: Any = None) -> PoseProvider:
    """按配置构建位姿来源。

    `configs/*.yaml` 里：
    ```yaml
    deploy:
      ros2:
        pose:
          source: tf                # tf | odometry | static | identity
          target_frame: map
          source_frame: camera_color_optical_frame
          optical_frame_correction: false
    ```
    """
    pose_cfg = dict(cfg.get("deploy.ros2.pose", {}) or {})
    source = str(pose_cfg.get("source", "identity")).lower()

    if source == "identity":
        return IdentityPoseProvider()

    if source == "static":
        tr = pose_cfg.get("translation", [0.0, 0.0, 0.0])
        rpy = pose_cfg.get("rpy", [0.0, 0.0, 0.0])
        return StaticPoseProvider.from_euler(
            tr, rpy,
            optical_frame_correction=bool(pose_cfg.get("optical_frame_correction", True)),
        )

    if source in {"tf", "tf2"}:
        if node is None:
            logger.warn("配置要求 TF 位姿但没有传入 ROS 节点，退回 identity")
            return IdentityPoseProvider()
        return TfPoseProvider(
            node,
            target_frame=str(pose_cfg.get("target_frame", "map")),
            source_frame=str(pose_cfg.get("source_frame", "camera_color_optical_frame")),
            timeout_s=float(pose_cfg.get("timeout_s", 0.1)),
            optical_frame_correction=bool(pose_cfg.get("optical_frame_correction", False)),
            # ⚠️ **默认不给 fallback**，这是刻意的，别改回去。
            #
            # 之前这里是 `fallback=IdentityPoseProvider(warn=False)`，它造成过一个
            # 非常难查的 bug：`lookup_transform` 因为 stamp 类型不对而抛异常，
            # 被这个兜底静默吞掉 → 返回**恒等位姿** → 帧带着错误位姿建图
            # （现象是"处理了 N 帧却 0 体素"，而 `pose_failures` 还是 0，
            #  因为位姿"拿到了"，只是完全不对）。
            #
            # 这也直接违反 `nodes.py` 里已经写明的原则：
            # **"宁可丢帧，也不要用恒等位姿凑数"** ——
            # 错误位姿会污染整张地图且事后极难归因，丢帧只是少一点观测。
            #
            # 真需要兜底就显式配 `deploy.ros2.pose.fallback_identity: true`，
            # 并接受"位姿可能是错的"这个后果。
            fallback=(IdentityPoseProvider(warn=True)
                      if bool(pose_cfg.get("fallback_identity", False)) else None),
        )

    if source in {"odometry", "odom"}:
        if node is None:
            logger.warn("配置要求里程计位姿但没有传入 ROS 节点，退回 identity")
            return IdentityPoseProvider()
        b2c = pose_cfg.get("base_to_camera")
        return OdometryPoseProvider(
            node,
            topic=str(pose_cfg.get("odom_topic", "/odom")),
            base_to_camera=tuple(b2c) if b2c else None,
            optical_frame_correction=bool(pose_cfg.get("optical_frame_correction", True)),
        )

    logger.warn(f"未知的位姿来源 {source!r}，退回 identity")
    return IdentityPoseProvider()
