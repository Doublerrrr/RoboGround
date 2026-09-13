#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""13 · 隔离诊断：TF 位姿约定到底哪一步错了（在 WSL 内跑）。

背景
----
`04_verify_ros2.sh` 在 WSL（有真 tf2_ros）下跑出 **0 体素 → 0 物体**，
而同样的测试在 RoboStack Windows（无 tf2_ros、走 odometry）下是**通过**的。
换位姿来源就坏，说明问题在位姿约定，而不是链路本身。

本脚本把"位姿重建"这一步单独拎出来，用**同一棵静态 TF 树**做三组对照：

  A. source_frame = camera_color_optical_frame, correction=False   ← 文档说的正确用法
  B. source_frame = camera_link,              correction=True     ← 另一种正确用法
  C. source_frame = camera_color_optical_frame, correction=True   ← 疑似 bug 的用法

**A 和 B 必须给出同一个位姿**（同一个物理相机，两种描述方式）；
C 如果与 A 不同，就说明"查 optical frame 还叠加纠正"确实是重复旋转。

这个"两种约定必须一致"的判据，比只测一条路径强得多 ——
它同时验证了 R_OPTICAL_TO_LINK 这个矩阵本身是对的。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
for p in (str(_HERE), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ros2_e2e_test as T  # noqa: E402  复用测试里的硬编码 TF 常量


def fmt(R: np.ndarray, t: np.ndarray) -> str:
    body = "\n".join("      [" + "  ".join(f"{v:+.4f}" for v in row) + "]" for row in R)
    return f"    R =\n{body}\n    t = [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}]"


def main() -> int:
    import rclpy
    from rclpy.node import Node
    from tf2_ros import StaticTransformBroadcaster
    from geometry_msgs.msg import TransformStamped

    from roboground.deployment.ros2.tf import TfPoseProvider, quaternion_to_matrix

    print("=" * 78)
    print("TF 位姿约定隔离诊断")
    print("=" * 78)
    print(f"  TF 树 : {T.TARGET_FRAME} → {T.CAMERA_LINK} → {T.OPTICAL_FRAME}")
    print(f"  相机架 : {T.CAMERA_LINK} 在 map 的平移 {T.CAM_LINK_TRANSLATION}，"
          f"姿态 {T.CAM_LINK_QUATERNION}")
    print(f"  光学系 : {T.OPTICAL_FRAME} 相对相机架的四元数 {T.OPTICAL_QUATERNION}")

    # ---- 期望值：由测试自带的 build_render_pose 给出（生成输入数据时用的那个）----
    expected = T.build_render_pose()
    print()
    print("── 期望位姿（测试生成渲染数据时用的）──")
    print(fmt(np.asarray(expected.R), np.asarray(expected.t)))

    # ---- 起 rclpy，发同一棵静态 TF 树 ----
    rclpy.init()
    node = Node("roboground_tf_diag")
    bcast = StaticTransformBroadcaster(node)

    def make_tf(parent, child, translation, quaternion) -> TransformStamped:
        m = TransformStamped()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id = parent
        m.child_frame_id = child
        (m.transform.translation.x, m.transform.translation.y,
         m.transform.translation.z) = translation
        (m.transform.rotation.x, m.transform.rotation.y,
         m.transform.rotation.z, m.transform.rotation.w) = quaternion
        return m

    bcast.sendTransform([
        make_tf(T.TARGET_FRAME, T.CAMERA_LINK,
                T.CAM_LINK_TRANSLATION, T.CAM_LINK_QUATERNION),
        make_tf(T.CAMERA_LINK, T.OPTICAL_FRAME,
                T.OPTICAL_TRANSLATION, T.OPTICAL_QUATERNION),
    ])
    print()
    print("已发布静态 TF，等待 2s 让 buffer 建立 ...")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    stamp = node.get_clock().now().to_msg()

    cases = [
        ("A", T.OPTICAL_FRAME, False, "查 optical frame，不纠正（文档说的正确用法）"),
        ("B", T.CAMERA_LINK, True, "查 camera_link，做纠正（另一种正确用法）"),
        ("C", T.OPTICAL_FRAME, True, "查 optical frame 又纠正 ← 疑似 bug"),
        ("D", T.CAMERA_LINK, False, "查 camera_link 不纠正（错，留作对照）"),
    ]

    poses = {}
    for tag, src, corr, desc in cases:
        prov = TfPoseProvider(node, target_frame=T.TARGET_FRAME, source_frame=src,
                              optical_frame_correction=corr, timeout_s=2.0)
        prov.setup()
        # 给 listener 一点时间收静态变换
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
        pose = prov.get_pose(stamp)
        poses[tag] = pose
        print()
        print(f"── {tag}. {desc}")
        print(f"     source={src}  correction={corr}  failures={prov.failures}")
        if pose is None:
            print("     [FAIL] 拿不到位姿")
            continue
        R = np.asarray(pose.R)
        t = np.asarray(pose.t)
        print(fmt(R, t))
        # 与期望位姿比：世界点 → 相机点，用两者算同一个世界点应得到相同结果
        p_world = np.array([2.0, 0.0, 0.5])
        d_exp = np.asarray(expected.R) @ p_world + np.asarray(expected.t)
        d_got = R @ p_world + t
        print(f"     把世界点 (2.0, 0.0, 0.5) 变换到相机系："
              f"期望 {np.round(d_exp, 4).tolist()} 得到 {np.round(d_got, 4).tolist()}"
              f"  误差 {np.linalg.norm(d_exp - d_got):.4f} m")

    # ---- 判定 ----
    print()
    print("=" * 78)
    print("判定")
    print("=" * 78)

    def err(a, b):
        if poses.get(a) is None or poses.get(b) is None:
            return float("nan")
        return float(np.linalg.norm(np.asarray(poses[a].t) - np.asarray(poses[b].t)))

    def err_vs_expected(tag):
        if poses.get(tag) is None:
            return float("nan")
        return float(np.linalg.norm(np.asarray(poses[tag].t) - np.asarray(expected.t)))

    e_ab = err("A", "B")
    e_c = err_vs_expected("C")
    e_a = err_vs_expected("A")
    e_b = err_vs_expected("B")

    print(f"  A vs B 的平移差 = {e_ab:.6f} m  "
          f"→ {'✅ 两种约定一致（R_OPTICAL_TO_LINK 正确）' if e_ab < 1e-6 else '❌ 不一致'}")
    print(f"  A vs 期望       = {e_a:.6f} m  {'✅' if e_a < 1e-6 else '❌'}")
    print(f"  B vs 期望       = {e_b:.6f} m  {'✅' if e_b < 1e-6 else '❌'}")
    print(f"  C vs 期望       = {e_c:.6f} m  "
          f"→ {'✅ 无问题' if e_c < 1e-6 else '❌ C 的位姿是错的（重复旋转）'}")

    if e_c > 1e-6:
        print()
        print("  ★ 结论：`04_verify_ros2.sh` 里 `optical_frame_correction = (pose=='tf')`")
        print("     在 source_frame 已经是 `*_optical_frame` 的情况下**多套了一次轴纠正**，")
        print("     位置/朝向都被旋转两次 → 反投影出的点云整体跑偏 → 建图 0 体素。")
        print("     正确做法：查 optical frame 时 correction 必须为 False（与 tf.py 文档一致）。")

    # ---- stamp 类型对照：项目内部约定是 float 秒，tf2 要的是 Time 消息 ----
    print()
    print("=" * 78)
    print("stamp 类型对照（这是「0 体素」的真正根因）")
    print("=" * 78)
    print("  项目内部约定 `PoseProvider.get_pose(stamp)` 里的 stamp 是 **float 秒** ——")
    print("  因为 nodes.py 里写的是 `stamp = _stamp_to_seconds(color_msg.header.stamp)`，")
    print("  而 `TrajectoryPoseProvider` 也直接 `float(stamp)`。")
    print("  但 `TfPoseProvider` 把它**原样**喂给了 tf2 的 `lookup_transform()`，")
    print("  后者要的是 `rclpy.time.Time` 消息 —— 类型不匹配。")
    print()
    prov_tf = TfPoseProvider(node, target_frame=T.TARGET_FRAME,
                             source_frame=T.OPTICAL_FRAME,
                             optical_frame_correction=False, timeout_s=1.0)
    prov_tf.setup()
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)

    now_msg = node.get_clock().now().to_msg()
    now_float = float(now_msg.sec) + float(now_msg.nanosec) * 1e-9

    for label, st in (("ROS Time 消息（tf2 期望的类型）", now_msg),
                      (f"float 秒 {now_float:.3f}（项目内部约定）", now_float)):
        before = prov_tf.failures
        pose = prov_tf.get_pose(st)
        print(f"  [{label}]")
        print(f"     failures 增量 = {prov_tf.failures - before}")
        if pose is None:
            print("     → 返回 None（帧应被丢弃，但实测 pose_failures=0，可疑）")
            continue
        R = np.asarray(pose.R)
        t = np.asarray(pose.t)
        same = (np.allclose(R, np.asarray(expected.R))
                and np.allclose(t, np.asarray(expected.t)))
        print(f"     t = {np.round(t, 4).tolist()}")
        print(f"     R 对角 = {np.round(np.diag(R), 4).tolist()}")
        print(f"     → {'✅ 与期望位姿一致' if same else '❌ 与期望位姿不一致'}")
        if np.allclose(R, np.eye(3)) and np.allclose(t, 0.0):
            print("       ★★ 这是**单位位姿**！帧会拿恒等位姿去建图 → 点云全跑偏 → 0 体素")

    # ---- 数据面验证：反投影 GT 物体，看落点 ----
    print()
    print("=" * 78)
    print("数据面验证：用 A/C 两组位姿反投影深度，看落点离 GT 多远")
    print("=" * 78)
    color, depth_m, K, gt = T.render_scene(320, 240)
    v, u = np.meshgrid(np.arange(depth_m.shape[1]), np.arange(depth_m.shape[0]))
    valid = depth_m > 0.1
    # K 是 CameraIntrinsics **对象**（不是矩阵），所以按属性取 —— 踩过一次
    fx, fy, cx, cy = float(K.fx), float(K.fy), float(K.cx), float(K.cy)
    x = (u[valid] - cx) / fx * depth_m[valid]
    y = (v[valid] - cy) / fy * depth_m[valid]
    z = depth_m[valid]
    pts_cam = np.stack([x, y, z], axis=1)
    print(f"  有效深度点 {len(pts_cam)} 个")

    for tag in ("A", "C"):
        pose = poses.get(tag)
        if pose is None:
            continue
        R, t = np.asarray(pose.R), np.asarray(pose.t)
        pts_world = (pts_cam - t) @ R          # R 是 world→cam，所以反变换
        print(f"  [{tag}] 世界点范围 "
              f"x[{pts_world[:,0].min():+.2f},{pts_world[:,0].max():+.2f}] "
              f"y[{pts_world[:,1].min():+.2f},{pts_world[:,1].max():+.2f}] "
              f"z[{pts_world[:,2].min():+.2f},{pts_world[:,2].max():+.2f}]")
        for label, c in gt:
            d = np.linalg.norm(pts_world - np.asarray(c), axis=1).min()
            print(f"        GT {label:<7} {np.round(c,2).tolist()}  最近点距 {d:.3f} m")

    rclpy.shutdown()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
