"""ROS2 消息转换层（纯 Python，**不依赖 rclpy，可单元测试**）。

把项目内部的类型（`RGBDFrame` / `SemanticMap` / `ReasoningResult`）
与 ROS 消息（或 JSON 字符串）互相转换。

设计取舍：为什么用 JSON 字符串而不是自定义 .msg
---------------------------------------------
- 自定义 msg 需要 `colcon build` 编译消息包，跨机器/跨 ROS 发行版容易出问题；
- 本项目的地图/答案是**结构化但 schema 会演进**的数据，JSON 更合适；
- 代价是失去类型安全与带宽效率 —— 生产环境建议定义正式 .msg，
  这一点在 docstring 里明确说明，而不是假装它是最优解。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("deployment.ros2.bridge")


def _probe_ros2() -> bool:
    """探测 rclpy 是否可用（不抛异常）。"""
    try:
        import rclpy  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


#: 本机是否具备 ROS2 运行时
ROS2_AVAILABLE: bool = _probe_ros2()

#: 话题 → 消息类型（供节点与文档共用，避免两处不一致）
TOPIC_TYPES: Dict[str, str] = {
    "/camera/color/image_raw": "sensor_msgs/Image",
    "/camera/depth/image_raw": "sensor_msgs/Image",
    "/camera/color/camera_info": "sensor_msgs/CameraInfo",
    "/roboground/semantic_map": "std_msgs/String",
    "/roboground/query": "std_msgs/String",
    "/roboground/answer": "std_msgs/String",
}


def require_ros2() -> None:
    """需要 ROS2 时调用；缺失则给出**可操作的**报错信息。"""
    if not ROS2_AVAILABLE:
        raise ImportError(
            "当前环境没有 ROS2（rclpy 不可用）。\n"
            "ROS2 不要用 pip 安装，请用官方安装包：\n"
            "  - Windows: 安装 ros2-humble 官方发行版，并 source 其 setup 脚本\n"
            "  - 或使用 WSL2 / Ubuntu 安装 ros2-humble\n"
            "提示：RoboGround 的核心功能（建图/查询/推理/评测）不需要 ROS，\n"
            "      只有 deployment.ros2.nodes 里的节点才需要。"
        )


# ==========================================================================
# RGB-D 帧
# ==========================================================================
def frame_to_dict(frame: RGBDFrame, *, include_images: bool = False) -> Dict[str, Any]:
    """`RGBDFrame` → 可 JSON 序列化的 dict。

    Parameters
    ----------
    include_images
        是否把图像数据也塞进去（会非常大，仅在调试/落盘时用）。
        默认 False，只传元数据 + 相机参数。
    """
    payload: Dict[str, Any] = {
        "frame_id": frame.frame_id,
        "timestamp": float(frame.timestamp),
        "width": int(frame.width),
        "height": int(frame.height),
        "intrinsics": {
            "fx": float(frame.intrinsics.fx), "fy": float(frame.intrinsics.fy),
            "cx": float(frame.intrinsics.cx), "cy": float(frame.intrinsics.cy),
            "width": frame.intrinsics.width, "height": frame.intrinsics.height,
        },
        "pose": {
            "R": np.asarray(frame.pose.R, dtype=float).tolist(),
            "t": np.asarray(frame.pose.t, dtype=float).tolist(),
        },
        "meta_keys": sorted(k for k in frame.meta.keys() if k not in {"boxes_3d", "labels"}),
    }
    boxes = frame.meta.get("boxes_3d")
    if boxes is not None:
        payload["boxes_3d"] = np.asarray(boxes, dtype=float).tolist()
    labels = frame.meta.get("labels")
    if labels is not None:
        payload["labels"] = [str(x) for x in labels]

    if include_images:
        payload["color_shape"] = list(np.asarray(frame.color).shape)
        payload["depth_shape"] = list(np.asarray(frame.depth_m).shape)
        payload["color_bytes"] = np.asarray(frame.color, dtype=np.uint8).tobytes().hex()
        payload["depth_m"] = np.asarray(frame.depth_m, dtype=np.float32).ravel().tolist()
    return payload


def dict_to_frame(data: Dict[str, Any]) -> Optional[RGBDFrame]:
    """dict → `RGBDFrame`（图像数据需通过 `color_bytes` / 外部提供）。

    ⚠️ 如果 dict 里没有图像数据，返回 None —— 明确失败好过造一个假帧
    让下游在几百行之后才崩。
    """
    if "color_shape" not in data or "color_bytes" not in data:
        logger.debug("dict 里没有图像数据（color_bytes/color_shape），无法还原 RGBDFrame")
        return None

    height, width = int(data["height"]), int(data["width"])
    color = np.frombuffer(bytes.fromhex(data["color_bytes"]), dtype=np.uint8)
    color = color.reshape(tuple(data["color_shape"]))

    if "depth_m" in data:
        depth = np.asarray(data["depth_m"], dtype=np.float32).reshape(height, width)
    else:
        depth = np.zeros((height, width), dtype=np.float32)

    ik = data["intrinsics"]
    intrinsics = CameraIntrinsics(
        fx=ik["fx"], fy=ik["fy"], cx=ik["cx"], cy=ik["cy"],
        width=ik.get("width", width), height=ik.get("height", height),
    )
    p = data["pose"]
    pose = CameraPose(np.asarray(p["R"], dtype=float), np.asarray(p["t"], dtype=float))

    meta: Dict[str, Any] = {}
    if "boxes_3d" in data:
        meta["boxes_3d"] = np.asarray(data["boxes_3d"], dtype=np.float32)
    if "labels" in data:
        meta["labels"] = [str(x) for x in data["labels"]]

    return RGBDFrame(
        color=color, depth_m=depth, intrinsics=intrinsics, pose=pose,
        frame_id=str(data.get("frame_id", "ros_frame")),
        timestamp=float(data.get("timestamp", 0.0)),
        meta=meta,
    )


def camera_info_to_intrinsics(msg: Any, *, width: Optional[int] = None,
                              height: Optional[int] = None) -> CameraIntrinsics:
    """`sensor_msgs/CameraInfo` → `CameraIntrinsics`。

    ⚠️ **字段名大小写是个坑**（真实 rclpy 才暴露得出来）：
    - ROS2 的 **Python** 消息类用小写：`msg.k`、`msg.d`、`msg.p`；
    - C++ 侧才是大写 `K`/`D`/`P`。

    我第一版按 C++ 约定写成 `msg.K`，离线单测全过（用的是假对象），
    真机上直接报 "CameraInfo 缺少 K 矩阵"。这里两种都收，
    避免不同实现/不同版本带来的差异。
    """
    K = getattr(msg, "k", None)
    if K is None:
        K = getattr(msg, "K", None)
    if K is None:
        raise ValueError("CameraInfo 缺少 k/K 矩阵")
    if len(K) < 9:
        raise ValueError(f"CameraInfo 的 k 矩阵长度应为 9，实际 {len(K)}")

    msg_w = getattr(msg, "width", None)
    msg_h = getattr(msg, "height", None)
    out_w = int(width if width is not None else (msg_w or 0)) or None
    out_h = int(height if height is not None else (msg_h or 0)) or None

    return CameraIntrinsics(
        fx=float(K[0]), fy=float(K[4]), cx=float(K[2]), cy=float(K[5]),
        width=out_w, height=out_h,
    )


# ==========================================================================
# 语义地图 / 回答
# ==========================================================================
def map_to_dict(semantic_map, *, max_objects: int = 200) -> Dict[str, Any]:
    """`SemanticMap` → 可 JSON 序列化的 dict（物体列表 + 摘要）。"""
    lo, hi = semantic_map.bounds
    return {
        "type": "SemanticMap",
        "num_objects": int(semantic_map.num_objects),
        "num_voxels": int(semantic_map.num_voxels),
        "feature_dim": int(semantic_map.feature_dim),
        "voxel_size": float(semantic_map.voxel_grid.voxel_size),
        "bounds": {"min": np.asarray(lo, float).tolist(), "max": np.asarray(hi, float).tolist()},
        "labels": list(semantic_map.labels[:50]),
        "robot_position": (semantic_map.meta or {}).get("robot_position"),
        "objects": [o.to_dict() for o in semantic_map.objects[:max_objects]],
        "summary": semantic_map.describe(max_objects=min(20, max_objects)),
    }


def answer_to_dict(result: Any) -> Dict[str, Any]:
    """`ReasoningResult` → 可 JSON 序列化的 dict。"""
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    else:
        payload = {"answer": str(result)}
    payload["type"] = "ReasoningResult"
    return payload


def to_json(payload: Dict[str, Any]) -> str:
    """安全地转 JSON 字符串（处理 numpy 类型）。"""
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    return json.dumps(payload, ensure_ascii=False, default=_default)


def from_json(text: str) -> Dict[str, Any]:
    """JSON 字符串 → dict（失败时返回带 error 的空 dict）。"""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"value": data}
    except Exception as exc:
        logger.warn(f"JSON 解析失败：{exc}")
        return {"error": str(exc), "raw": str(text)[:200]}


# ==========================================================================
# 传感器流 → RGBDFrame（节点与离线测试共用的核心逻辑）
# ==========================================================================
def frame_from_streams(
    color: np.ndarray,
    depth_raw: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose_provider: Any,
    *,
    stamp: Optional[float] = None,
    depth_scale: float = 1000.0,
    frame_id: str = "ros_frame",
    min_depth: float = 0.1,
    max_depth: float = 8.0,
    autosync: bool = False,
) -> Optional[RGBDFrame]:
    """把传感器数据 + 位姿组装成 `RGBDFrame`；**位姿不可用时返回 None**。

    为什么单独抽成纯函数
    --------------------
    它是"ROS 消息 → 内部类型"的咽喉，也是真机最容易出错的地方。
    抽出来后就能在没有 rclpy 的机器上完整测试，
    节点里只剩薄薄一层胶水（见 `tests/test_ros2_pose.py`）。

    ★ 关键设计：**位姿拿不到就返回 None，绝不退回恒等位姿。**
    用恒等位姿建图会污染整张地图 —— 不同时刻的帧被叠在同一位置，
    表现为重影，且物体坐标全错、事后极难归因。
    丢一帧只是少一点观测，代价可控得多。

    Parameters
    ----------
    depth_raw
        原始深度（通常 uint16 毫米）；`depth_scale` 是换算成米的除数。
    autosync
        深度与彩色分辨率不一致时是否自动缩放对齐（异构相机常见）。
    """
    if pose_provider is None:
        logger.warn("没有提供位姿来源，无法组装帧")
        return None

    try:
        pose = pose_provider.get_pose(stamp)
    except Exception as exc:
        logger.debug(f"位姿查询异常：{exc}")
        pose = None
    if pose is None:
        return None

    from roboground.geometry.projection import raw_depth_to_meters  # noqa: PLC0415

    depth_m = raw_depth_to_meters(np.asarray(depth_raw), depth_scale)
    color = np.asarray(color)

    if depth_m.shape != color.shape[:2]:
        if not autosync:
            logger.warn(
                f"深度图 {depth_m.shape} 与彩色图 {color.shape[:2]} 尺寸不一致，"
                "且未开启 autosync，无法组装帧"
            )
            return None
        from PIL import Image  # noqa: PLC0415

        depth_m = np.asarray(
            Image.fromarray(depth_m).resize((color.shape[1], color.shape[0]), Image.NEAREST),
            dtype=np.float32,
        )

    depth_m = np.where((depth_m > min_depth) & (depth_m < max_depth), depth_m, 0.0).astype(np.float32)

    return RGBDFrame(
        color=color,
        depth_m=depth_m,
        intrinsics=intrinsics,
        pose=pose,
        timestamp=float(stamp) if stamp is not None else 0.0,
        frame_id=frame_id,
        meta={"source": "ros2", "has_pose": True},
    )
