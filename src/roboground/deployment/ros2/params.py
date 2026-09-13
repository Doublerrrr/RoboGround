# -*- coding: utf-8 -*-
"""ROS2 参数 ↔ `Config` 的接线表与声明逻辑（**唯一接线点**）。

为什么参数表放在 `roboground` 库里，而不是 `ros2_ws` 那个 ament 包里
=================================================================
因为"声明参数的节点"和"使用参数的代码"必须是同一个节点：

    `--params-file` 的顶层键就是**节点名**，必须与运行时节点名完全一致。

早期实现用另一个名字（`roboground_param_bootstrap`）的节点去读参数、算出
Config，再拿给真正的 `perception` 节点用。实测（`scripts/wsl/42_probe_param_matching.py`）：

    参数文件键 = perception，节点名 = perception                 → 取到 0.77 ✓
    参数文件键 = perception，节点名 = roboground_param_bootstrap → 取到 0.05 ✗
    参数文件键 = perception，节点名 = query                      → 取到 0.05 ✗

也就是说 bootstrap 读到的是**默认值**、真节点读到的是**文件里的值**，
两者静默不一致 —— 又一个"看起来配了其实没生效"。正确做法就是让真正干活的
节点自己声明参数（本模块的 `declare_params`），顺带还解决了：

  · `ros2 param get/list/dump /perception` 能看到**真实生效值**；
  · `ros2 param set /perception sync_slop 0.42` 能在线生效（热参数）；
  · 冷参数被显式**拒绝**并说明原因，而不是静默忽略。

离线可测性
=========
本模块顶层**不 import rclpy**（ROS2 相关的调用都在函数体内延迟 import）。
`declare_params` 只用到宿主对象的 4 个方法：
`get_name` / `declare_parameter` / `get_parameter` / `add_on_set_parameters_callback`，
所以测试里用一个 30 行的假节点就能把接线逻辑完整测掉
（见 `tests/test_ros2_params.py`）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # ROS2 已 source 时用真实的参数结果消息
    from rcl_interfaces.msg import SetParametersResult as _SetParametersResult
except ImportError:  # pragma: no cover - 离线兜底（Windows 上没装 ROS2）
    class _SetParametersResult:  # type: ignore[no-redef]
        """与 `rcl_interfaces/SetParametersResult` 字段一致的替身。

        有了它，参数回调的**判定逻辑**（放行/拒绝/原因文案）就能在没装 ROS2
        的机器上离线测试 —— 真机上最容易错的就是这段逻辑。
        真机上走的是 ROS2 的真消息，两者字段相同、语义相同。
        """

        __slots__ = ("successful", "reason")

        def __init__(self, *, successful: bool = True, reason: str = "") -> None:
            self.successful = successful
            self.reason = reason


def result(successful: bool, reason: str = "") -> Any:
    """构造参数回调结果（真 ROS2 消息或离线替身）。"""
    return _SetParametersResult(successful=successful, reason=reason)

#: 参数名 → `Config` 里的点号路径。`None` = 只给 ROS2 用，不进 Config。
#:
#: ⚠️ 改这张表必须同步改 `config/roboground.yaml`；
#:    `tests/test_config_paths.py` 锁死"每个路径在 DEFAULT_CONFIG 里真实存在"，
#:    `tests/test_ros2_package.py` 锁死"YAML 键集合与这张表一致"。
PARAM_MAP: Dict[str, Optional[str]] = {
    "config_file": None,                     # 只给 ROS2 用
    "pose_source": "deploy.ros2.pose.source",
    "pose_target_frame": "deploy.ros2.pose.target_frame",
    "pose_source_frame": "deploy.ros2.pose.source_frame",
    "optical_frame_correction": "deploy.ros2.pose.optical_frame_correction",
    "odom_topic": "deploy.ros2.pose.odom_topic",
    "sync_slop": "deploy.ros2.sync_slop",
    "publish_every_n": "deploy.ros2.publish_every_n",
    "depth_scale": "geometry.depth_scale",
    "voxel_size": "geometry.voxel_size",
    "min_depth": "geometry.min_depth",
    "max_depth": "geometry.max_depth",
    "autosync_depth": "deploy.ros2.autosync_depth",
    "detector": "perception.detector",
    "segmenter": "perception.segmenter",
    "encoder": "perception.encoder",
    # 话题名（各家机器人命名差别很大，必须可配）
    "color_image_topic": "deploy.ros2.topics.color_image",
    "depth_image_topic": "deploy.ros2.topics.depth_image",
    "camera_info_topic": "deploy.ros2.topics.camera_info",
    "semantic_map_topic": "deploy.ros2.topics.semantic_map",
    "query_topic": "deploy.ros2.topics.query",
    "answer_topic": "deploy.ros2.topics.answer",
}

#: **热参数**：运行期改一次、下一帧就生效。
#:
#: 判据是"这个值在每一帧的处理路径上被重新读取"，而不是"它看起来不重要"：
#:   · sync_slop        → `ApproximateTimeSynchronizer.slop`（可直接改属性）
#:   · publish_every_n  → 每帧发布判断
#:   · min/max_depth    → 每帧传给 `frame_from_streams`
#:   · depth_scale      → 每帧传给 `frame_from_streams`
#:   · autosync_depth   → 每帧传给 `frame_from_streams`
HOT_PARAMS: Tuple[str, ...] = (
    "sync_slop", "publish_every_n", "min_depth", "max_depth",
    "depth_scale", "autosync_depth",
)

#: **冷参数**：只在节点构造时读一次，运行期改不了 —— 回调会**明确拒绝**。
#:
#: 为什么不是"静默忽略"：静默忽略会让用户以为调参生效了（本项目反复强调
#: "配了不生效"是最贵的坑）。ROS2 的参数回调支持返回失败原因，
#: 那就把原因说清楚：`ros2 param set` 会直接打印拒绝原因。
COLD_PARAMS: Tuple[str, ...] = tuple(
    name for name, path in PARAM_MAP.items()
    if path is not None and name not in HOT_PARAMS
)


def plan_param_update(names: Sequence[str]) -> Dict[str, List[str]]:
    """把一批待设置的参数名分成"能热改的"与"需要重启的"（纯函数，可离线测试）。

    三个桶：
      · `apply`    —— 热参数，改了下一次回调就生效；
      · `restart`  —— 冷参数 + `config_file`（改配置文件本来就要重启，
                      而且 "接受但什么都不做" 正是要避免的静默失败）；
      · `unknown`  —— 不在接线表里的名字，也**不允许**悄悄放过。
    """
    apply, restart, unknown = [], [], []
    for name in names:
        if name not in PARAM_MAP:
            unknown.append(name)
        elif name in HOT_PARAMS:
            apply.append(name)
        else:
            # 冷参数，以及 config_file（它在 PARAM_MAP 里但 path=None）
            restart.append(name)
    return {"apply": apply, "restart": restart, "unknown": unknown}


def get_path(name: str) -> Optional[str]:
    """参数名 → Config 点号路径（未知参数返回 `None`）。"""
    return PARAM_MAP.get(name)


def declare_params(node: Any, cfg: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
    """在 `node` 上声明全部参数并注册在线调参回调；返回 `(生效的 Config, 被覆盖项)`。

    Parameters
    ----------
    node
        任何提供 `get_name` / `declare_parameter` / `get_parameter` /
        `add_on_set_parameters_callback` 的对象（真 rclpy 节点或测试假节点）。
    cfg
        已有配置。给 `None` 时按 `config_file` 参数自行加载。

    Notes
    -----
    **声明默认值 = 配置里的真实生效值**，这一点是刻意的：

      · 声明为 `None`（不设默认）→ `ros2 param get` 显示 "not set"、
        `ros2 param dump` 里看不到，用户无法知道当前生效值；
      · 声明为写死的默认值（例如 `sync_slop=0.1`）→ 会把 YAML 里配的
        `0.05` **静默盖掉**；
      · 声明为**配置里的真实值** → 两个问题都没有：既看得见，
        又不会覆盖 YAML，而且类型是具体的（launch 传字符串会当场报错，
        而不是靠下游 `float()` 侥幸兜住）。

    声明之后还有**关键的一步**：把参数的实际取值（可能已被 `-p` 或
    `--params-file` 覆盖）同步回 `cfg`，否则会出现
    "`ros2 param get` 显示 0.42，节点实际按 0.05 跑"。
    """
    from roboground import load_config  # noqa: PLC0415

    node.declare_parameter("config_file", "")
    cfg_file = str(node.get_parameter("config_file").value or "")
    if cfg is None:
        cfg = load_config(cfg_file) if cfg_file else load_config()

    for name, path in PARAM_MAP.items():
        if path is None:
            continue
        node.declare_parameter(name, cfg.get(path))

    # ★ 覆盖值回灌：`-p` / `--params-file` 的值在 declare 时就已生效，
    #   这里把它们同步进 cfg，保证"参数服务看到的"与"算法用的"是同一个值。
    applied = sync_config_from_params(node, cfg)

    node.add_on_set_parameters_callback(_make_callback(node, cfg))
    return cfg, applied


def _make_callback(node: Any, cfg: Any):
    """构造参数设置回调：热参数放行、冷参数拒绝并说明原因。"""
    def _callback(params):
        plan = plan_param_update([p.name for p in params])
        if plan["unknown"]:
            return result(False,
                          "未知参数（不在 PARAM_MAP 里）："
                          + ", ".join(sorted(plan["unknown"])))
        if plan["restart"]:
            return result(False,
                          "以下参数只在节点构造时读取，无法在线修改："
                          + ", ".join(sorted(plan["restart"]))
                          + "。请改 config/roboground.yaml 或重启节点"
                            "（ros2 run … --ros-args -p 名:=值）")
        for p in params:
            if p.name == "config_file":
                # 只记录来源，不改已加载的配置（换配置文件本来就要重启）
                continue
            cfg.set(get_path(p.name), _unwrap(p.value))
        # 让节点把新的值同步到自己的运行期字段（例如 self.min_depth）
        refresh = getattr(node, "refresh_params", None)
        if callable(refresh):
            refresh()
        return result(True)
    return _callback


def _unwrap(value: Any) -> Any:
    """rclpy 的数组参数是 `array.array`，转回普通 list 更利于序列化与比较。"""
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_unwrap(v) for v in value]
    return value


def _same(a: Any, b: Any) -> bool:
    """比较两个配置值是否等价（容忍 int/float 与 bool 的差异）。"""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-12
    return a == b


def sync_config_from_params(node: Any, cfg: Any) -> Dict[str, Any]:
    """把节点上参数的**实际取值**同步进 `cfg`，返回真正发生变化（=被覆盖）的项。

    只记录"与配置不同"的项，所以日志里的 `applied` 就是"这次有哪些参数
    被命令行/参数文件改掉了"，一眼能看出环境差异。
    """
    applied: Dict[str, Any] = {}
    for name, path in PARAM_MAP.items():
        if path is None:
            continue
        value = _unwrap(node.get_parameter(name).value)
        if value is None:
            continue
        if _same(cfg.get(path), value):
            continue
        cfg.set(path, value)
        applied[name] = value
    return applied


def config_from_params(node: Any) -> Tuple[Any, Dict[str, Any]]:
    """自行加载配置，再同步节点参数，返回 `(cfg, 被覆盖的项)`。

    保留给"参数已在别处声明好"的场景（测试、外部集成）。
    正常路径用 `declare_params`。
    """
    from roboground import load_config  # noqa: PLC0415

    cfg_file = str(node.get_parameter("config_file").value or "")
    cfg = load_config(cfg_file) if cfg_file else load_config()
    return cfg, sync_config_from_params(node, cfg)
