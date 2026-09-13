#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实的 ROS2 端到端测试（在 WSL/Ubuntu 里跑，需要 rclpy）。

它验证的是"离线测试覆盖不到"的那一层：
```
rclpy 消息 → 时间同步 → TF 查询 → 帧组装 → 建图 → 话题发布
```

测试设计（为什么不循环论证）
--------------------------
1. **TF 是硬编码常量发布的**，不用我们自己的辅助函数生成：
   - `map → camera_link`：平移 (0, 0, 1.2)，旋转单位四元数
   - `camera_link → camera_color_optical_frame`：标准 REP-103 光学系旋转
     四元数 `(-0.5, 0.5, -0.5, 0.5)`
   于是"相机在 map 的 (0,0,1.2)、朝 +x 看"这件事完全由 TF 决定，
   节点必须**自己**从 TF 重建出这个位姿。
2. **场景来自解析式渲染器**（`data/synthetic.py`），物体世界坐标是精确已知的。
3. 断言：节点建出的地图里，物体位置必须接近这些已知真值。
   —— 位姿约定只要错一点（哪怕只是轴搞反），物体就会跑到别处，测试必然失败。

用法::

    source /opt/ros/humble/setup.bash
    python3 scripts/wsl/ros2_e2e_test.py
    python3 scripts/wsl/ros2_e2e_test.py --frames 6 --verbose
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

# 项目 src 加入路径（WSL 里没pip install也能跑）
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---- 硬编码的 TF 常量（测试的"事实来源"）----
CAM_LINK_TRANSLATION = (0.0, 0.0, 1.2)                 # 相机架在 map 的 (0,0,1.2)
CAM_LINK_QUATERNION = (0.0, 0.0, 0.0, 1.0)             # x,y,z,w —— 单位旋转
OPTICAL_QUATERNION = (-0.5, 0.5, -0.5, 0.5)            # REP-103 → 光学系（标准值）
OPTICAL_TRANSLATION = (0.0, 0.0, 0.0)                  # 与 camera_link 同点

TARGET_FRAME = "map"
CAMERA_LINK = "camera_link"
OPTICAL_FRAME = "camera_color_optical_frame"


def build_render_pose():
    """为"渲染输入数据"构造位姿。

    注意：这里用常量 + 项目辅助函数算，是**为了生成输入数据**（把场景渲染成图像）。
    被测对象（PerceptionNode）拿到的是 TF 常量，必须自己重建位姿 —— 所以不循环论证。
    """
    from roboground.types import CameraPose
    from roboground.deployment.ros2.tf import quaternion_to_matrix

    R_link_optical = quaternion_to_matrix(*OPTICAL_QUATERNION)
    R_map_link = quaternion_to_matrix(*CAM_LINK_QUATERNION)
    R_map_optical = R_map_link @ R_link_optical
    C = np.asarray(CAM_LINK_TRANSLATION) + R_map_link @ np.asarray(OPTICAL_TRANSLATION)
    R_map_to_cam = R_map_optical.T
    return CameraPose(R_map_to_cam, -R_map_to_cam @ C)


def render_scene(width: int, height: int, fov_deg: float = 60.0):
    """渲染一个已知场景，返回 (color, depth_m, intrinsics, gt_objects)。"""
    from roboground.data.synthetic import SceneObject, SyntheticRoom, _look_forward_pose

    # 物体放在 map 坐标系里（相机在 (0,0,1.2) 朝 +x 看，所以物体放在 +x 方向）
    objects = [
        SceneObject("table", [2.0, -0.3, 0.4], [0.9, 0.9, 0.75], 0.0, (150, 110, 80)),
        SceneObject("chair", [2.4, 0.6, 0.45], [0.5, 0.5, 0.9], 0.0, (90, 90, 140)),
        SceneObject("cup", [1.7, 0.0, 0.85], [0.12, 0.12, 0.14], 0.0, (230, 230, 240)),
    ]
    room = SyntheticRoom(width=6.0, depth=6.0, height=2.6, objects=objects)

    # 用与 TF 等价的位姿渲染（相机在世界 (0,0,1.2)，朝 +x，略微俯视）
    pose = build_render_pose()
    K = _intrinsics_for(width, height, fov_deg)
    res = room.render(pose, K)
    gt = [(o.label, np.asarray(o.center, dtype=np.float64)) for o in objects]
    return res.color, res.depth_m, K, gt


def _intrinsics_for(width: int, height: int, fov_deg: float):
    from roboground.types import CameraIntrinsics

    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return CameraIntrinsics(fx=fx, fy=fx, cx=(width - 1) / 2.0,
                            cy=(height - 1) / 2.0, width=width, height=height)


# =============================================================================
# ROS2 部分
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=5, help="发布多少帧")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--rate-hz", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=90.0, help="总超时（秒）")
    ap.add_argument("--tolerance", type=float, default=0.35, help="物体定位容差（米）")
    ap.add_argument("--pose", choices=["auto","tf","odometry","static"], default="auto",
                    help="位姿注入方式；auto 会优先 tf，无 tf2_ros 时退回 odometry")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        print(f"[FAIL] 需要 ROS2 环境：{exc}")
        print("  请先： source /opt/ros/humble/setup.bash")
        return 2

    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    # tf2_ros 在 RoboStack 的 Windows 构建里**不存在**（只有 Linux/macOS）。
    # 所以做成可选：拿不到就走 odometry/static 位姿。
    tf2_available = True
    try:
        from tf2_ros import StaticTransformBroadcaster
    except ImportError:
        tf2_available = False

    if args.pose == "auto":
        args.pose = "tf" if tf2_available else "odometry"
    if args.pose == "tf" and not tf2_available:
        print("[WARN] 环境里没有 tf2_ros（RoboStack Windows 无此包），自动退回 odometry 位姿")
        args.pose = "odometry"

    print("=" * 72)
    print("RoboGround × ROS2 真实端到端测试")
    print(f"  ROS 实现 : rclpy（{'Linux/WSL' if tf2_available else 'RoboStack Windows'}）")
    print(f"  位姿方式 : {args.pose}")
    print("=" * 72)

    # ---- 准备场景 ----
    color, depth_m, K, gt = render_scene(args.width, args.height)
    valid = int((depth_m > 0.1).sum())
    print(f"[1] 场景就绪：{color.shape}  有效深度 {valid} px")
    for label, c in gt:
        print(f"      GT {label:<8} 世界坐标 ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})")
    if valid < 500:
        print("[FAIL] 渲染出的有效深度太少，无法测试（检查相机朝向/物体位置）")
        return 2
    print(f"[2] TF 常量：{TARGET_FRAME} → {CAMERA_LINK} 平移 {CAM_LINK_TRANSLATION}；"
          f"{CAMERA_LINK} → {OPTICAL_FRAME} 已给标准光学旋转")

    rclpy.init()
    pub_node = Node("roboground_e2e_publisher")

    # ---- 位姿注入：TF 或 里程计 ----
    # 无论哪种方式，都用**硬编码的标准常量**，不使用项目自己的辅助函数，
    # 这样节点必须自己把外部位姿重建出来 —— 测试才有意义。
    odom_pub = None
    if args.pose == "tf":
        tf_broadcaster = StaticTransformBroadcaster(pub_node)

        def make_tf(parent: str, child: str, translation, quaternion) -> TransformStamped:
            t = TransformStamped()
            t.header.stamp = pub_node.get_clock().now().to_msg()
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = translation
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quaternion
            return t

        tf_broadcaster.sendTransform([
            make_tf(TARGET_FRAME, CAMERA_LINK, CAM_LINK_TRANSLATION, CAM_LINK_QUATERNION),
            make_tf(CAMERA_LINK, OPTICAL_FRAME, OPTICAL_TRANSLATION, OPTICAL_QUATERNION),
        ])
        print(f"[3] 已发布静态 TF（{TARGET_FRAME} → {CAMERA_LINK} → {OPTICAL_FRAME}）")
    elif args.pose == "odometry":
        from nav_msgs.msg import Odometry

        odom_pub = pub_node.create_publisher(Odometry, "/odom", 10)
        print(f"[3] 将以 /odom 发布相机位姿：位置 {CAM_LINK_TRANSLATION}，"
              f"姿态用标准光学系四元数 {OPTICAL_QUATERNION}")
    else:
        print(f"[3] 位姿方式 static（由配置给定，不经 ROS）")

    # ---- 传感器发布器 ----
    sensor_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )
    color_pub = pub_node.create_publisher(Image, "/camera/color/image_raw", sensor_qos)
    depth_pub = pub_node.create_publisher(Image, "/camera/depth/image_raw", sensor_qos)
    info_pub = pub_node.create_publisher(CameraInfo, "/camera/color/camera_info", sensor_qos)

    def make_image(arr: np.ndarray, encoding: str) -> Image:
        msg = Image()
        msg.header.stamp = pub_node.get_clock().now().to_msg()
        msg.header.frame_id = OPTICAL_FRAME
        msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = int(arr.strides[0])
        msg.data = arr.tobytes()
        return msg

    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

    # ---- 订阅我们的地图输出 ----
    received_maps: list = []

    def on_map(msg: String):
        received_maps.append(msg.data)

    map_sub = pub_node.create_subscription(String, "/roboground/semantic_map", on_map, 10)

    # ---- 启动被测试的节点 ----
    from roboground import load_config
    from roboground.deployment.ros2.nodes import PerceptionNode

    cfg = load_config()
    # ★ 关键：位姿来源设为 TF，并指向标准 optical frame（所以不需要轴纠正）
    cfg.set("deploy.ros2.pose.source", args.pose)
    cfg.set("deploy.ros2.pose.target_frame", TARGET_FRAME)
    cfg.set("deploy.ros2.pose.source_frame", OPTICAL_FRAME)
    # ⚠️ 这里必须是 **False**，两种位姿模式都不需要轴纠正：
    #   - tf 模式：source_frame 已经是 `camera_color_optical_frame`，
    #     TF 查出来的就是光学系位姿；再套一次纠正 = **重复旋转**。
    #   - odometry 模式：我们直接发布的就是相机光学系位姿。
    # → 这两种都不需要轴纠正。
    #
    # ★ **但 `static` 模式需要！** 它的 `translation`/`rpy` 描述的是
    #   **`camera_link`** 的位姿（REP-103：x 前/y 左/z 上），不是光学系。
    #   所以必须先做 `R_OPTICAL_TO_LINK` 纠正才能得到光学系位姿。
    #
    # ⚠️ 这是一个**真实 bug 的修复**：之前这里写死成 `False`，
    #   于是 `--pose static` 拿到的是"把 camera_link 当成光学系"的错误位姿 ——
    #   实测建图 **0 个物体**（点全跑到 bounds 外面去了）。
    #   之所以一直没发现，是因为回归只跑 `auto`（在 WSL 上选 tf、
    #   在 RoboStack 上退回 odometry），**static 这条分支从没进过回归**。
    #   教训：**一个模式有 N 条分支，只测一条 = 其他 N-1 条都是未知状态。**
    cfg.set("deploy.ros2.pose.optical_frame_correction", args.pose == "static")
    cfg.set("deploy.ros2.pose.odom_topic", "/odom")
    cfg.set("deploy.ros2.pose.translation", list(CAM_LINK_TRANSLATION))
    cfg.set("deploy.ros2.pose.rpy", [0.0, 0.0, 0.0])
    cfg.set("deploy.ros2.sync_slop", 0.1)
    cfg.set("deploy.ros2.publish_every_n", 3)
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("perception.prompts", ["table", "chair", "cup"])
    cfg.set("geometry.depth_scale", 1000.0)
    cfg.set("project.verbose", False)

    # ---- 启动被测试的节点 ----
    try:
        node = PerceptionNode(cfg, node_name="roboground_perception_e2e")
    except Exception as exc:
        print(f"[FAIL] PerceptionNode 构造失败：{type(exc).__name__}: {exc}")
        rclpy.shutdown()
        return 2
    print(f"[4] PerceptionNode 已启动（位姿来源={node.pose_provider.name}）")

    # stub 检测器需要 GT 框 → 通过 frame.meta 注入。
    # 真实场景里这些框由检测器产生；离线后端下我们直接给真值，
    # 这样测试的焦点就落在"ROS2 链路 + 位姿"上，而不是检测精度。
    gt_sizes = {
        "table": [0.9, 0.9, 0.75],
        "chair": [0.5, 0.5, 0.9],
        "cup": [0.12, 0.12, 0.14],
    }
    default_size = [0.3, 0.3, 0.3]
    boxes = np.stack([
        np.concatenate([c, gt_sizes.get(label, default_size), [0.0]])
        for label, c in gt
    ]).astype(np.float32)
    labels = [g[0] for g in gt]

    original_add_frame = node.builder.add_frame

    def add_frame_with_gt(frame, **kw):
        frame.meta["boxes_3d"] = boxes
        frame.meta["labels"] = labels
        if args.verbose:
            # 探针：0 体素时用来区分"位姿错"还是"点云空"。
            # ⚠️ 字段名是 `depth_m`（不是 `depth`）—— 写错会让 AttributeError
            # 冒泡到 spin 线程，把整个测试变成"0 帧处理"，反而掩盖真正的问题。
            d = np.asarray(frame.depth_m)
            pose = getattr(frame, "pose", None)
            t = "None" if pose is None else np.round(np.asarray(pose.t), 3).tolist()
            r0 = None if pose is None else np.round(np.asarray(pose.R)[0], 3).tolist()
            print(f"    [dbg] depth={d.shape}/{d.dtype} nonzero={int((d > 0).sum())} "
                  f"range=[{d.min():.2f},{d.max():.2f}] pose_t={t} pose_R0={r0}")
        return original_add_frame(frame, **kw)

    node.builder.add_frame = add_frame_with_gt

    # ---- 独立线程 spin ----
    stop = threading.Event()
    errors: list = []

    def spin_loop():
        try:
            while not stop.is_set() and rclpy.ok():
                rclpy.spin_once(pub_node, timeout_sec=0.02)
                rclpy.spin_once(node, timeout_sec=0.02)
        except Exception as exc:      # pragma: no cover
            errors.append(f"{type(exc).__name__}: {exc}")

    spin_thread = threading.Thread(target=spin_loop, daemon=True)
    spin_thread.start()

    # 给 TF 静态广播一点时间传到 buffer
    time.sleep(1.0)

    # ---- 发布帧 ----
    period = 1.0 / max(args.rate_hz, 0.1)
    print(f"[5] 开始发布 {args.frames} 帧 RGB-D（{args.rate_hz}Hz）...")
    t0 = time.time()
    sent = 0
    while sent < args.frames and (time.time() - t0) < args.timeout:
        info = CameraInfo()
        info.header.stamp = pub_node.get_clock().now().to_msg()
        info.header.frame_id = OPTICAL_FRAME
        info.height, info.width = args.height, args.width
        info.k = [K.fx, 0.0, K.cx, 0.0, K.fy, K.cy, 0.0, 0.0, 1.0]
        info.p = [K.fx, 0.0, K.cx, 0.0, 0.0, K.fy, K.cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        if odom_pub is not None:
            odom = Odometry()
            odom.header.stamp = info.header.stamp
            odom.header.frame_id = TARGET_FRAME
            odom.child_frame_id = OPTICAL_FRAME
            odom.pose.pose.position.x = CAM_LINK_TRANSLATION[0]
            odom.pose.pose.position.y = CAM_LINK_TRANSLATION[1]
            odom.pose.pose.position.z = CAM_LINK_TRANSLATION[2]
            (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
             odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = OPTICAL_QUATERNION
            odom_pub.publish(odom)

        color_pub.publish(make_image(color, "rgb8"))
        depth_pub.publish(make_image(depth_mm, "16UC1"))
        info_pub.publish(info)
        sent += 1
        time.sleep(period)

    # 等节点把最后几帧处理完
    time.sleep(2.0)
    stop.set()
    spin_thread.join(timeout=3.0)

    # ---- 断言 ----
    print()
    print("=" * 72)
    print("结果")
    print("=" * 72)
    stats = node.stats()
    print(f"  已处理帧数     : {stats['frames_processed']}")
    print(f"  位姿失败帧数   : {stats['pose_failures']}")
    print(f"  位姿来源       : {stats['pose_source']}")
    print(f"  收到的地图消息 : {len(received_maps)}")
    if errors:
        print(f"  spin 异常      : {errors[:2]}")

    failures = []
    if stats["frames_processed"] == 0:
        failures.append("节点没有处理任何帧（TF/时间同步链路可能不通）")
    if stats["pose_failures"] > 0:
        failures.append(f"有 {stats['pose_failures']} 帧因位姿缺失被丢弃（TF 查找失败）")
    if not received_maps:
        failures.append("没有收到语义地图消息")

    # 解析地图，检查物体位置
    if received_maps:
        import json

        payload = json.loads(received_maps[-1])
        objs = payload.get("objects", [])
        print(f"  地图物体数     : {len(objs)}")
        print()
        print("  物体定位对比（地图 vs GT 真值）：")
        # ★ **必须按标签匹配，不能"找最近的 GT"**。
        #
        # 这是一个真实 bug：第一版对每个地图物体找**最近**的 GT，却不检查标签。
        # 后果是 table 的地图质心 (1.78, 0.10, 0.68) 离 **cup 的 GT**
        # (1.70, 0, 0.85) 只有 0.213 m，离它自己的 GT (2.0, −0.3, 0.4) 反而有 0.534 m
        # —— 于是 table 这一行**报了 cup 的误差**，看起来 0.209 m 很好，
        # 实际是把两个物体张冠李戴了。**这种错会让指标系统性虚高**。
        from collections import defaultdict
        gt_by_label = defaultdict(list)
        for gt_label, gt_c in gt:
            gt_by_label[gt_label].append(np.asarray(gt_c, dtype=np.float64))

        matched = 0
        unmatched_labels = []
        for obj in objs:
            center = np.asarray(obj["center"], dtype=np.float64)
            label = obj.get("label", "?")
            cands = gt_by_label.get(label)
            if not cands:
                unmatched_labels.append(label)
                continue
            d = min(float(np.linalg.norm(center - c)) for c in cands)
            nearest = min(cands, key=lambda c: float(np.linalg.norm(center - c)))
            ok = d <= args.tolerance
            matched += int(ok)
            flag = "✓" if ok else "✗"
            print(f"    {flag} {label:<8} 地图({center[0]:+.2f},{center[1]:+.2f},{center[2]:+.2f}) "
                  f"GT({nearest[0]:+.2f},{nearest[1]:+.2f},{nearest[2]:+.2f}) 误差={d:.3f}m")
        if unmatched_labels:
            print(f"    （地图里有 {len(unmatched_labels)} 个标签在 GT 中不存在："
                  f"{sorted(set(unmatched_labels))}）")
        if matched < min(2, len(gt)):
            failures.append(
                f"只有 {matched} 个物体**按标签**落在 {args.tolerance}m 容差内"
                f"（位姿约定可能错了）")

    print()
    if failures:
        print("[FAIL] 测试未通过：")
        for f in failures:
            print(f"       · {f}")
        rc = 1
    else:
        print("[PASS] ROS2 端到端测试通过：")
        print("       · rclpy 消息 → 时间同步 → TF 查询 → 帧组装 → 建图 → 话题发布 全链路打通")
        print(f"       · 物体定位误差均在 {args.tolerance}m 内")
        rc = 0

    try:
        node.destroy_node()
        pub_node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
