"""ROS2 节点：感知 / 语义地图 / 推理。

三个节点的职责
-------------
```
PerceptionNode   订阅 RGB-D + CameraInfo → 反投影成 3D → 累积体素特征场
                 （对应 S1+S2 的实时部分，跑在快档）
MapNode          维护 SemanticMap，周期性发布摘要与物体列表
                 （中间层，负责"地图 → 下游可用语义"）
VLMNode          订阅自然语言问题 → 规则引擎/VLM 推理 → 发布结构化答案
                 （跑在慢档，可独立降级）
```

导入策略
--------
本模块在**没有 rclpy 的环境里也能 import**（类会退化成占位符，
实例化时抛出带操作指引的 `ImportError`）。这样：
- 单元测试可以在无 ROS 的机器上覆盖 `bridge` 层；
- 用户不会因为 `import roboground.deployment.ros2.nodes` 就炸掉整个流程。

启动方式见 `roboground/deployment/ros2/README.md`。
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional

import numpy as np

from roboground.deployment.ros2.bridge import (
    ROS2_AVAILABLE,
    answer_to_dict,
    camera_info_to_intrinsics,
    map_to_dict,
    require_ros2,
    to_json,
)
from roboground.deployment.ros2.tf import _stamp_to_seconds
from roboground.utils.logging import get_logger

logger = get_logger("deployment.ros2.nodes")


# ==========================================================================
# QoS 策略（**模块级纯数据**，离线可测）
# ==========================================================================
# 为什么把"策略"和"rclpy 的 QoSProfile 对象"分开：
# 下面整段真实实现都在 `if ROS2_AVAILABLE:` 里，离线环境下函数根本不存在，
# 于是 **QoS 配置在离线测试里完全测不到** —— 而它恰恰是一个真实缺陷的所在
# （状态型话题用了默认 VOLATILE，真机上后启动的订阅者收不到地图）。
#
# 把策略提成纯数据之后：
#   · 离线可以断言「地图必须是 transient_local」（防止被改回去）；
#   · ROS2 环境下再验证**行为**（后加入的订阅者真的收到地图）。
#
# 依据 REP-2003（ROS 2 QoS 约定）：
#   · 传感器数据 → BEST_EFFORT + VOLATILE + KEEP_LAST(小)：宁可丢帧不要延迟
#   · 状态/参数 → RELIABLE + TRANSIENT_LOCAL + KEEP_LAST(1)：后加入者必须拿到当前值
QOS_POLICIES: Dict[str, Dict[str, Any]] = {
    "sensor": {
        "reliability": "best_effort",
        "durability": "volatile",
        "history": "keep_last",
        "depth": 2,
        "used_by": "彩色图 / 深度图 / CameraInfo",
        "why": "丢几帧没关系（下一帧马上到），但绝不能因重传增加延迟",
    },
    "state": {
        "reliability": "reliable",
        "durability": "transient_local",   # ★ latched：后加入者立刻收到当前状态
        "history": "keep_last",
        "depth": 1,
        "used_by": "语义地图 / 问答答案",
        "why": "发布的是「当前状态」而非数据流；VOLATILE 会让后启动的规划节点"
               "永远收不到地图 —— 这是真机上的实际故障模式",
    },
}


# ==========================================================================
# 有 ROS2 时的真实实现
# ==========================================================================
if ROS2_AVAILABLE:  # pragma: no cover - 需要 ROS2 运行时
    import rclpy  # noqa: PLC0415
    from rclpy.node import Node  # noqa: PLC0415
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: PLC0415
    from sensor_msgs.msg import CameraInfo, Image  # noqa: PLC0415
    from std_msgs.msg import String  # noqa: PLC0415

    _RELIABILITY = {"best_effort": ReliabilityPolicy.BEST_EFFORT,
                    "reliable": ReliabilityPolicy.RELIABLE}
    _DURABILITY = {"volatile": DurabilityPolicy.VOLATILE,
                   "transient_local": DurabilityPolicy.TRANSIENT_LOCAL}
    _HISTORY = {"keep_last": HistoryPolicy.KEEP_LAST}

    def _qos(name: str) -> "QoSProfile":
        """按 `QOS_POLICIES[name]` 构建 rclpy 的 QoSProfile。"""
        p = QOS_POLICIES[name]
        return QoSProfile(
            reliability=_RELIABILITY[p["reliability"]],
            durability=_DURABILITY[p["durability"]],
            history=_HISTORY[p["history"]],
            depth=int(p["depth"]),
        )

    def _sensor_qos() -> "QoSProfile":
        """传感器数据的 QoS（见 `QOS_POLICIES['sensor']`）。"""
        return _qos("sensor")

    def _state_qos() -> "QoSProfile":
        """**状态型**话题的 QoS（见 `QOS_POLICIES['state']`）。

        ★ 这是一个真实缺陷的修复：地图/答案早期用默认 QoS（`VOLATILE`），
        真机上规划/导航节点若比感知节点**后启动**，会一直等一张
        永远不会再发的历史地图，表现为"接不上"。

        这个缺陷在离线单测里**测不出来**（假对象不模拟 QoS 语义），
        也不会报任何错 —— 只有真机上"后启动一个订阅者"才会暴露。
        """
        return _qos("state")

    def _image_to_numpy(msg: "Image") -> np.ndarray:
        """`sensor_msgs/Image` → numpy（支持 rgb8/bgr8/16UC1/32FC1）。"""
        encoding = str(getattr(msg, "encoding", "rgb8")).lower()
        height, width = int(msg.height), int(msg.width)

        if encoding in {"rgb8", "bgr8"}:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            arr = arr.reshape(height, int(msg.step))[:, : width * 3].reshape(height, width, 3)
            if encoding == "bgr8":
                arr = arr[:, :, ::-1]        # BGR → RGB
            return np.ascontiguousarray(arr)

        if encoding in {"16uc1", "mono16"}:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint16)
            return arr.reshape(height, int(msg.step) // 2)[:, :width].copy()

        if encoding in {"32fc1", "32fc"}:
            arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
            return arr.reshape(height, int(msg.step) // 4)[:, :width].copy()

        if encoding in {"mono8", "8uc1"}:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            return arr.reshape(height, int(msg.step))[:, :width].copy()

        raise ValueError(f"不支持的图像编码：{encoding}")

    class PerceptionNode(Node):
        """订阅 RGB-D，实时把 2D 语义升维到 3D 并累积进语义地图。

        与"能跑通"版本相比，这里补齐了真机必需的三件事：

        1. **相机位姿**（`pose_provider`）—— 由 TF / 里程计 / 静态外参提供。
           缺了它建出来的只是"以第一帧相机为原点"的局部地图，
           不同时刻的帧会被叠在同一位置（视觉上表现为重影）。
        2. **三路消息的时间同步** —— 彩色、深度、内参来自三个独立话题，
           靠"缓存最新"配出来的帧在机器人运动时会有明显错位。
           这里用 `message_filters.ApproximateTimeSynchronizer` 按时间戳配对。
        3. **位姿按图像时间戳查询** —— TF 用"最新"会让位姿与图像不同步。
        """

        def __init__(self, cfg=None, *, node_name: str = "roboground_perception") -> None:
            """构造感知节点。

            Parameters
            ----------
            cfg
                配置对象。给 `None` 时**由本节点自己声明 ROS2 参数并据其构造配置**
                （`params.declare_params`）。推荐给 `None`：
                `--params-file` 的顶层键就是节点名，只有"声明参数的节点"与
                "干活的节点"是同一个，参数文件才能正确匹配 —— 详见
                `roboground.deployment.ros2.params` 模块 docstring
                （实测证据在 `scripts/wsl/42_probe_param_matching.py`）。
            """
            super().__init__(node_name)
            from roboground.deployment.ros2.tf import build_pose_provider  # noqa: PLC0415
            from roboground.mapping import MapBuilder  # noqa: PLC0415

            if cfg is None:
                from roboground.deployment.ros2.params import declare_params  # noqa: PLC0415

                cfg, applied = declare_params(self)
                if applied:
                    self.get_logger().info(
                        "参数文件/命令行覆盖了 " + str(len(applied)) + " 项："
                        + ", ".join(f"{k}={v}" for k, v in sorted(applied.items())))
            self.cfg = cfg
            self.builder = MapBuilder(cfg)
            self.topics = dict(cfg.get("deploy.ros2.topics", {}) or {})
            # 把配置里的热参数读进运行期字段（在线调参的落点见 refresh_params）
            self.refresh_params()

            self._frame_count = 0
            self.pose_failures = 0
            self.sync_drops = 0

            # ---- 位姿来源 ----
            self.pose_provider = build_pose_provider(cfg, self)
            self.pose_ready = False
            try:
                if hasattr(self.pose_provider, "setup"):
                    self.pose_provider.setup()
                # ★ 等位姿源真正可用再开始收数据。
                #
                # 真机启动顺序常常是"感知节点先起、TF 树/里程计后到"，
                # 不等就会把开头若干帧**静默丢弃**（丢帧本身是设计原则：
                # 宁可丢帧也不伪造位姿，但"能等就不该丢"）。
                # 此刻还没开始 spin，所以这里自己转几次回调是安全的。
                wait = getattr(self.pose_provider, "wait_ready", None)
                if callable(wait):
                    import rclpy  # noqa: PLC0415
                    ready_s = float(cfg.get("deploy.ros2.pose.ready_timeout_s", 5.0))
                    self.pose_ready = bool(wait(
                        ready_s,
                        spin_once=lambda d: rclpy.spin_once(self, timeout_sec=d),
                    ))
                    if not self.pose_ready:
                        self.get_logger().warn(
                            f"位姿来源 {self.pose_provider.name} 在 {ready_s:.1f}s 内未就绪；"
                            "行为：位姿查不到的帧会被**丢弃**（不会伪造位姿）")
            except Exception as exc:
                self.get_logger().warn(f"位姿来源初始化失败（{exc}），将使用恒等位姿")

            # ---- 时间同步的三路订阅 ----
            from message_filters import ApproximateTimeSynchronizer, Subscriber  # noqa: PLC0415

            qos = _sensor_qos()
            self._sub_color = Subscriber(
                self, Image, self.topics.get("color_image", "/camera/color/image_raw"), qos_profile=qos)
            self._sub_depth = Subscriber(
                self, Image, self.topics.get("depth_image", "/camera/depth/image_raw"), qos_profile=qos)
            self._sub_info = Subscriber(
                self, CameraInfo, self.topics.get("camera_info", "/camera/color/camera_info"), qos_profile=qos)

            self._sync = ApproximateTimeSynchronizer(
                [self._sub_color, self._sub_depth, self._sub_info],
                queue_size=10, slop=self.sync_slop,
            )
            self._sync.registerCallback(self._on_synced)

            # ★ 地图是**状态型**话题 → 必须用 `_state_qos()`（TRANSIENT_LOCAL），
            # 否则真机上后启动的规划/导航节点永远收不到地图。见 `_state_qos` 的说明。
            self.map_pub = self.create_publisher(
                String, self.topics.get("semantic_map", "/roboground/semantic_map"),
                _state_qos())

            self.get_logger().info(
                f"PerceptionNode 已启动：位姿来源={self.pose_provider.name}，"
                f"时间同步容差={self.sync_slop * 1000:.0f}ms"
            )

        # ---------- 时间同步回调（三路消息一起到）----------
        def _on_synced(self, color_msg, depth_msg, info_msg) -> None:
            """三路消息按时间戳配对成功后调用 —— 直接处理，无需再缓存。"""
            try:
                color = _image_to_numpy(color_msg)
                depth_raw = _image_to_numpy(depth_msg)
                intrinsics = camera_info_to_intrinsics(info_msg)
            except Exception as exc:
                self.get_logger().warn(f"解析同步消息失败：{exc}")
                return

            stamp = _stamp_to_seconds(color_msg.header.stamp)
            self.process_frame(color, depth_raw, intrinsics, stamp=stamp)

        # ---------- 核心处理 ----------
        def process_frame(self, color, depth_raw, intrinsics, *, stamp=None,
                          frame_id: Optional[str] = None) -> bool:
            """把一对同步好的 RGB-D + 内参 + 位姿变成一帧并并入地图。

            组装逻辑在 `bridge.frame_from_streams` 里（纯函数、可离线测试），
            这里只负责计数、告警与发布。
            """
            from roboground.deployment.ros2.bridge import frame_from_streams  # noqa: PLC0415

            fid = frame_id or f"ros_{self._frame_count}"
            frame = frame_from_streams(
                color, depth_raw, intrinsics, self.pose_provider,
                stamp=stamp,
                depth_scale=self.depth_scale,
                frame_id=fid,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                autosync=self.autosync_depth,
            )

            if frame is None:
                # ⚠️ **宁可丢帧，也不要用恒等位姿凑数**：
                # 错误位姿会污染整张地图（事后极难归因），丢帧只是少一点观测。
                self.pose_failures += 1
                if self.pose_failures % 10 == 1:
                    self.get_logger().warn(
                        f"位姿不可用，已丢弃 {self.pose_failures} 帧"
                        f"（来源={self.pose_provider.name}）")
                return False

            self.builder.add_frame(frame)
            self._frame_count += 1

            publish_every = self.publish_every
            if publish_every > 0 and self._frame_count % publish_every == 0:
                smap = self.builder.finalize()
                self.map_pub.publish(String(data=to_json(map_to_dict(smap))))
                self.get_logger().info(
                    f"已处理 {self._frame_count} 帧，发布地图（{smap.num_objects} 个物体，"
                    f"位姿失败 {self.pose_failures} 帧）"
                )
            return True

        def refresh_params(self) -> None:
            """把 `self.cfg` 里**热参数**的值重新读进运行期字段。

            这是"在线调参"的真正落点：`ros2 param set /perception sync_slop 0.42`
            → `params._make_callback` 把它写进 `cfg` → 回调调本方法
            → `self._sync.slop` 变成 0.42，**下一帧**就用新值。

            冷参数（感知后端、话题名、位姿来源、voxel_size…）不在 `HOT_PARAMS` 里，
            `params.declare_params` 注册的回调会**拒绝**这类设置并说明原因 ——
            刻意不做"静默忽略"，因为"配了不生效"是最难查的一类坑。
            """
            self.min_depth = float(self.cfg.get("geometry.min_depth", 0.1))
            self.max_depth = float(self.cfg.get("geometry.max_depth", 8.0))
            self.depth_scale = float(self.cfg.get("geometry.depth_scale", 1000.0))
            self.autosync_depth = bool(self.cfg.get("deploy.ros2.autosync_depth", False))
            self.publish_every = int(self.cfg.get("deploy.ros2.publish_every_n", 10))
            self.sync_slop = float(self.cfg.get("deploy.ros2.sync_slop", 0.05))
            sync = getattr(self, "_sync", None)
            if sync is not None:
                # `message_filters.ApproximateTimeSynchronizer` 的 slop 是普通属性，
                # 直接赋值即可对新来的消息生效（不需要重建订阅）。
                sync.slop = self.sync_slop

        def stats(self) -> Dict[str, Any]:
            return {
                "frames_processed": self._frame_count,
                "pose_failures": self.pose_failures,
                "pose_source": self.pose_provider.name,
                # 启动期位姿源是否等到了（False 表示开头若干帧会被丢）
                "pose_ready": bool(getattr(self, "pose_ready", False)),
                "sync_slop_s": float(self.sync_slop),
                **self.pose_provider.describe(),
            }

        @property
        def frames_processed(self) -> int:
            return self._frame_count

    class QueryNode(Node):
        """订阅自然语言问题 → 推理 → 发布结构化答案。"""

        def __init__(self, cfg=None, *, node_name: str = "roboground_query",
                     use_vlm: Optional[bool] = None) -> None:
            """构造问答节点。

            `use_vlm=None` 时声明一个 `use_vlm` ROS2 参数并以它为准
            （默认关：开箱即用不需要显存）。推理器是**惰性构造**的，
            所以这里改 `reasoning.vlm.enabled` 一定早于它被创建。
            """
            super().__init__(node_name)
            from roboground.reasoning.vlm import HybridReasoner  # noqa: PLC0415

            if cfg is None:
                from roboground.deployment.ros2.params import declare_params  # noqa: PLC0415

                cfg, _applied = declare_params(self)
            self.cfg = cfg

            if use_vlm is None:
                self.declare_parameter(
                    "use_vlm", bool(cfg.get("reasoning.vlm.enabled", False)))
                use_vlm = bool(self.get_parameter("use_vlm").value)
            else:
                # 显式传了值也要声明，否则 `ros2 param get /query use_vlm` 查不到
                self.declare_parameter("use_vlm", bool(use_vlm))
            cfg.set("reasoning.vlm.enabled", bool(use_vlm))
            self.use_vlm = bool(use_vlm)

            self.topics = dict(cfg.get("deploy.ros2.topics", {}) or {})
            # ★ 地图是**状态型**话题 → 必须用 `_state_qos()`（TRANSIENT_LOCAL），
            # 否则真机上后启动的规划/导航节点永远收不到地图。见 `_state_qos` 的说明。
            self.map_pub = self.create_publisher(
                String, self.topics.get("semantic_map", "/roboground/semantic_map"),
                _state_qos())
            # 答案也是**状态型**：查询者可能比推理完成晚加入，
            # 用 TRANSIENT_LOCAL 保证它一订阅就能拿到最近一条答案。
            self.answer_pub = self.create_publisher(
                String, self.topics.get("answer", "/roboground/answer"),
                _state_qos())
            # 查询请求是**事件型**（一问一答），丢一条就是丢一个请求 → RELIABLE；
            # 但不需要 latched（重放历史查询没有意义）。
            self.create_subscription(
                String, self.topics.get("query", "/roboground/query"),
                self._on_query, 10)

            self._map = None
            self._reasoner = None
            # 慢档：单独线程，避免阻塞 ROS 回调
            self._queue: list = []
            self._lock = threading.Lock()
            self._stop = threading.Event()
            self._worker = threading.Thread(target=self._loop, daemon=True)
            self._worker.start()
            self.get_logger().info("QueryNode 已启动")

        def update_map(self, semantic_map) -> None:
            with self._lock:
                self._map = semantic_map
                self._reasoner = None     # 地图换了，推理器要重建

        def _on_query(self, msg) -> None:
            with self._lock:
                self._queue.append(str(msg.data))
            if len(self._queue) > 8:          # 防止问题堆积
                self._queue = self._queue[-8:]

        def _loop(self) -> None:
            from roboground.reasoning.vlm import HybridReasoner  # noqa: PLC0415

            slow_hz = float(self.cfg.get("deploy.slow_tier_hz", 1.0))
            period = 1.0 / slow_hz if slow_hz > 0 else 1.0

            while not self._stop.is_set():
                with self._lock:
                    query = self._queue.pop(0) if self._queue else None
                    smap = self._map
                    if self._reasoner is None and smap is not None:
                        self._reasoner = HybridReasoner(smap, cfg=self.cfg)
                    reasoner = self._reasoner

                if query and reasoner is not None:
                    try:
                        result = reasoner.answer(query)
                        self.answer_pub.publish(String(data=to_json(answer_to_dict(result))))
                    except Exception as exc:
                        self.get_logger().warn(f"推理失败：{exc}")
                self._stop.wait(period)

        def destroy_node(self) -> bool:
            self._stop.set()
            return super().destroy_node()

else:  # pragma: no cover - 无 ROS2 环境
    class _Placeholder:
        """无 rclpy 时的占位类：实例化即给出可操作的报错。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_ros2()

    class PerceptionNode(_Placeholder):  # type: ignore[no-redef]
        """需要 ROS2 运行时。见 `roboground.deployment.ros2.bridge.require_ros2`。"""

    class QueryNode(_Placeholder):  # type: ignore[no-redef]
        """需要 ROS2 运行时。见 `roboground.deployment.ros2.bridge.require_ros2`。"""


# ==========================================================================
# 入口
# ==========================================================================
def spin_perception(cfg) -> None:  # pragma: no cover - 需要 ROS2
    """启动感知节点（阻塞）。

    处理流程现在完全由**时间同步回调**驱动（三路消息配对成功即处理一帧），
    所以这里只需要 `rclpy.spin`，不再需要手动轮询缓存。
    """
    require_ros2()
    node = PerceptionNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def spin_query(cfg) -> None:  # pragma: no cover - 需要 ROS2
    """启动问答节点（阻塞）。"""
    require_ros2()
    node = QueryNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
