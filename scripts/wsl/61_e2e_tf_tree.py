#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""61 · 完整 TF 树端到端验证（配合 `60_verify_tf_tree.sh` 使用）。

验证两件事
=========
**A. 链路正确性** —— 把 `ros2 launch roboground_ros tf_tree.launch.py` 真起的
   四条边查出来，与纯 Python 的解析解（`roboground_ros.tf_spec`）逐位对比。

   为什么必须"解析解 vs 实测"两路对答案：TF 的四元数方向错了**不报任何错**，
   只是相机位姿整体转了一个固定角度（实测 1.697 m 误差），
   只有拿一个独立算出来的期望值对比才能发现。

**B. 端到端一致性** —— 用 `pose_source:=tf` 跑真 `PerceptionNode`，
   走完整链条 `map→odom→base_link→camera_link→optical` 建图，
   物体定位结果必须与 `pose_source:=static` 的**逐位相同**。

   三个互相独立的位姿来源给出同一个地图，比"某一条路径能跑"强得多 ——
   它同时排除了"TF 查得对但被用错"和"位姿对了但建图错"。
"""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

_SRC = Path("/mnt/g/RoboGround/src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, "/mnt/g/RoboGround/scripts/wsl")

WIDTH, HEIGHT = 320, 240
N_FRAMES = 6
GT_TOLERANCE = 0.35
#: `pose_source:=static` 已实测的物体误差（米）；tf 模式必须给出同样的数
EXPECTED_STATIC_ERRORS = {"table": 0.534, "chair": 0.239, "cup": 0.053}
EXPECTED_TOL = 1e-3


def _frames_from_buffer(buf) -> list:
    """从 tf2 Buffer 里取出树上的全部帧名。

    ⚠️ `all_frames_as_yaml()` 在 Python 绑定里返回的是**字符串**（不是 dict），
    必须先 `yaml.safe_load`（这点文档没写清楚，实测才知道）。
    万一解析失败就退回按行扫 `key:` 的笨办法 —— 宁可粗糙也不要误报"没有帧"。
    """
    if not hasattr(buf, "all_frames_as_yaml"):
        return []
    raw = buf.all_frames_as_yaml()
    try:
        import yaml

        data = yaml.safe_load(raw) or {}
        if isinstance(data, dict):
            return [str(k).strip().rstrip(":") for k in data]
    except Exception:
        pass
    names = []
    for line in str(raw).splitlines():
        line = line.rstrip()
        if line and not line.startswith((" ", "\t")) and line.endswith(":"):
            names.append(line[:-1].strip())
    return names


def check_tf_tree() -> dict:
    """A. 查 TF 树：帧集合 + 与解析解的逐位对比。"""
    import rclpy
    from rclpy.node import Node
    from tf2_ros import Buffer, TransformListener

    from roboground_ros import tf_spec

    rclpy.init()
    node = Node("roboground_tf_tree_check")
    buf = Buffer()
    TransformListener(buf, node)

    frames = tf_spec.frame_names()
    target, source = frames[0], frames[-1]          # map → camera_color_optical_frame

    # 等 TF 树就位
    t0 = time.time()
    while time.time() - t0 < 20.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if buf.can_transform(target, source, rclpy.time.Time()):
            break
    ok_reachable = buf.can_transform(target, source, rclpy.time.Time())

    all_frames = _frames_from_buffer(buf)
    # `all_frames_as_yaml()` 只列出**有父节点**的帧，根帧（map）不在里面 ——
    # 所以"map 在不在"要用 can_transform 判断（见下面的逐帧可达性检查）。
    reachable = {}
    for f in frames:
        try:
            reachable[f] = bool(buf.can_transform(frames[0], f, rclpy.time.Time()))
        except Exception:
            reachable[f] = False

    measured = None
    if ok_reachable:
        tf_msg = buf.lookup_transform(target, source, rclpy.time.Time())
        tr = tf_msg.transform.translation
        ro = tf_msg.transform.rotation
        measured = ((float(tr.x), float(tr.y), float(tr.z)),
                    (float(ro.x), float(ro.y), float(ro.z), float(ro.w)))

    expected_t, expected_q = tf_spec.lookup_transform(target, source)

    node.destroy_node()
    rclpy.shutdown()

    result = {
        "target": target, "source": source,
        "reachable": ok_reachable,
        "expected_frames": frames,
        "actual_frames": sorted(set(all_frames)),
        "frame_reachable": reachable,
        "measured_t": measured[0] if measured else None,
        "measured_q": measured[1] if measured else None,
        "expected_t": tuple(round(v, 9) for v in expected_t),
        "expected_q": tuple(round(v, 9) for v in expected_q),
    }

    if measured:
        from roboground.deployment.ros2.tf import (
            quaternion_to_matrix,
            rotation_matrix_to_quaternion,
        )
        from roboground_ros import tf_spec as TS

        R_meas = quaternion_to_matrix(*measured[1])
        R_exp = np.asarray(TS.quaternion_to_matrix(expected_q))
        result["R_max_err"] = float(np.abs(R_meas - R_exp).max())
        result["t_max_err"] = float(
            np.abs(np.asarray(measured[0]) - np.asarray(expected_t)).max())
        # 交叉验证：把实测四元数转回矩阵再转成四元数，应当一致（单位性/归一化）
        q_roundtrip = rotation_matrix_to_quaternion(R_meas)
        result["q_roundtrip_err"] = max(
            abs(a - b) for a, b in zip(q_roundtrip, measured[1]))
    return result


def check_e2e_tf_pose() -> dict:
    """B. 用 pose_source=tf 跑真节点，对比 static 模式的物体定位。"""
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    import ros2_e2e_test as T
    from roboground import load_config
    from roboground.deployment.ros2.nodes import PerceptionNode
    from roboground_ros.tf_spec import resolve_frames

    frames = resolve_frames()
    color, depth_m, K, gt = T.render_scene(WIDTH, HEIGHT)

    cfg = load_config()
    cfg.set("deploy.ros2.pose.source", "tf")           # ★ 走 TF 树
    cfg.set("deploy.ros2.pose.target_frame", frames["map"])
    cfg.set("deploy.ros2.pose.source_frame", frames["optical"])
    # 查的是 *_optical_frame → 不做轴纠正（做了会重复旋转）
    cfg.set("deploy.ros2.pose.optical_frame_correction", False)
    cfg.set("deploy.ros2.sync_slop", 0.1)
    cfg.set("deploy.ros2.publish_every_n", 3)
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("perception.prompts", ["table", "chair", "cup"])
    cfg.set("geometry.depth_scale", 1000.0)
    cfg.set("project.verbose", False)

    rclpy.init()
    node = PerceptionNode(cfg, node_name="roboground_tf_tree_e2e")

    gt_list = [(str(l), np.asarray(c, dtype=float)) for l, c in gt]
    sizes = {"table": [0.9, 0.9, 0.75], "chair": [0.5, 0.5, 0.9], "cup": [0.12, 0.12, 0.14]}
    boxes = np.stack([np.concatenate([c, sizes.get(l, [0.3, 0.3, 0.3]), [0.0]])
                      for l, c in gt_list]).astype(np.float32)
    labels = [l for l, _ in gt_list]
    _orig = node.builder.add_frame

    def _add(frame, **kw):
        frame.meta["boxes_3d"] = boxes
        frame.meta["labels"] = labels
        return _orig(frame, **kw)

    node.builder.add_frame = _add

    maps = []
    state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL,
                           history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(String, "/roboground/semantic_map",
                             lambda m: maps.append(1), state_qos)

    stop = threading.Event()
    threading.Thread(target=lambda: [rclpy.spin_once(node, timeout_sec=0.05)
                                     for _ in iter(lambda: not stop.is_set(), False)],
                     daemon=True).start()

    sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE,
                            history=HistoryPolicy.KEEP_LAST, depth=10)
    pub_c = node.create_publisher(Image, cfg.get("deploy.ros2.topics.color_image"), sensor_qos)
    pub_d = node.create_publisher(Image, cfg.get("deploy.ros2.topics.depth_image"), sensor_qos)
    pub_i = node.create_publisher(CameraInfo, cfg.get("deploy.ros2.topics.camera_info"), sensor_qos)

    depth_raw = np.clip(np.asarray(depth_m) * 1000.0, 0, 65535).astype(np.uint16)

    def _img(arr, encoding, stamp_ns):
        m = Image()
        m.header.stamp.sec = stamp_ns // 1_000_000_000
        m.header.stamp.nanosec = stamp_ns % 1_000_000_000
        m.header.frame_id = frames["optical"]
        m.height, m.width = arr.shape[:2]
        m.encoding = encoding
        m.is_bigendian = 0
        m.step = m.width * (2 if encoding == "16UC1" else 3)
        m.data = np.ascontiguousarray(arr).tobytes()
        return m

    def _info(stamp_ns):
        m = CameraInfo()
        m.header.stamp.sec = stamp_ns // 1_000_000_000
        m.header.stamp.nanosec = stamp_ns % 1_000_000_000
        m.header.frame_id = frames["optical"]
        m.height, m.width = HEIGHT, WIDTH
        m.k = [float(K.fx), 0.0, float(K.cx), 0.0, float(K.fy), float(K.cy), 0.0, 0.0, 1.0]
        m.d = [0.0] * 5
        m.distortion_model = "plumb_bob"
        return m

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if all(p.get_subscription_count() > 0 for p in (pub_c, pub_d, pub_i)):
            break
        time.sleep(0.1)

    t0 = time.time()
    for i in range(N_FRAMES):
        stamp_ns = int(time.time() * 1e9)
        pub_c.publish(_img(np.asarray(color, dtype=np.uint8), "rgb8", stamp_ns))
        pub_d.publish(_img(depth_raw, "16UC1", stamp_ns))
        pub_i.publish(_info(stamp_ns))
        time.sleep(0.2)
    time.sleep(1.5)
    stop.set()
    time.sleep(0.3)

    st = node.stats()
    smap = node.builder.finalize()
    objects = [(o.label, np.asarray(o.center, dtype=float)) for o in smap.objects]
    errors = {}
    for lbl, c in gt_list:
        best, bd = None, 1e9
        for label, center in objects:
            if label != lbl:          # ★ 只认同名物体（历史 bug：按最近匹配，虚报误差）
                continue
            d = float(np.linalg.norm(center - c))
            if d < bd:
                best, bd = center, d
        if best is not None:
            errors[lbl] = (bd, best.tolist())
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass

    return {
        "elapsed_s": time.time() - t0,
        "frames_processed": int(st.get("frames_processed", 0)),
        "pose_failures": int(st.get("pose_failures", 0)),
        "pose_source": st.get("pose_source"),
        "pose_ready": bool(st.get("pose_ready", False)),
        "n_objects": smap.num_objects,
        "n_maps": len(maps),
        "errors": errors,
    }


def main() -> int:
    print("=" * 82)
    print("完整 TF 树端到端验证")
    print("=" * 82)

    print("\n[A] TF 树链路与解析解对比")
    a = check_tf_tree()
    print(f"  目标链：{a['target']} → {a['source']}")
    print(f"  树上的帧（期望 {len(a['expected_frames'])} 个）："
          f"{', '.join(a['expected_frames'])}")
    print(f"  实测 /tf_static 里的帧（不含根帧）：{', '.join(a['actual_frames']) or '(没读到)'}")
    print("  逐帧可达性：" + "  ".join(
        f"{k}={'✓' if v else '✗'}" for k, v in a["frame_reachable"].items()))
    print(f"  can_transform = {a['reachable']}")
    if a.get("measured_t") is not None:
        print(f"  实测平移 {tuple(round(v, 9) for v in a['measured_t'])}")
        print(f"  解析平移 {a['expected_t']}")
        print(f"  实测四元数 {tuple(round(v, 9) for v in a['measured_q'])}")
        print(f"  解析四元数 {a['expected_q']}")
        print(f"  ★ 旋转矩阵最大误差 {a['R_max_err']:.3e}   平移最大误差 {a['t_max_err']:.3e}")

    print("\n[B] pose_source:=tf 端到端建图（走完整 TF 链）")
    b = check_e2e_tf_pose()
    print(f"  处理帧数 {b['frames_processed']} / 位姿失败 {b['pose_failures']} / "
          f"位姿来源 {b['pose_source']} / 地图消息 {b['n_maps']} / 物体 {b['n_objects']}")
    for lbl, (err, center) in sorted(b["errors"].items()):
        exp = EXPECTED_STATIC_ERRORS.get(lbl)
        mark = "✓" if err <= GT_TOLERANCE else "✗"
        extra = ""
        if exp is not None:
            same = abs(err - exp) <= EXPECTED_TOL
            extra = (f"   [static 模式实测 {exp:.3f} m，"
                     f"{'逐位一致 ✓' if same else f'不一致 ✗（差 {abs(err - exp):.3f}）'}]")
        print(f"  {mark} {lbl:<6} 误差 {err:.3f} m  地图中心 {np.round(center, 3).tolist()}{extra}")

    print("\n" + "=" * 82)
    print("结果")
    print("=" * 82)
    checks = {
        "TF 树上 5 个标准帧都可达（含根帧 map）": all(a["frame_reachable"].values()),
        "map → optical 可查通": a["reachable"],
        "★ 实测 TF 与解析解一致（旋转）": a.get("R_max_err", 1) < 1e-9,
        "★ 实测 TF 与解析解一致（平移）": a.get("t_max_err", 1) < 1e-9,
        "★ 启动期等到了位姿源（pose_ready）": b["pose_ready"],
        "tf 模式无位姿失败": b["pose_failures"] == 0,
        "tf 模式处理了全部帧": b["frames_processed"] == N_FRAMES,
        "tf 模式建出 3 个物体": b["n_objects"] == len(EXPECTED_STATIC_ERRORS),
        "★ tf 模式物体误差与 static 模式逐位一致": all(
            abs(b["errors"].get(l, (9e9,))[0] - e) <= EXPECTED_TOL
            for l, e in EXPECTED_STATIC_ERRORS.items()),
        # ⚠️ 判据是"≥2/3 达标"而不是"全部达标"，理由与 `20_rosbag_e2e.py` 一致：
        # `table` 是场景里最大的物体，用 **bbox 当掩码**时它的 3D 包围盒会把
        # 地面点一起纳入 → 质心被拉偏（实测 0.534 m）。这是 bbox 掩码的固有性质，
        # 不是位姿或链路问题（证据：三种互相独立的位姿来源给出同一个 0.534）。
        "物体定位达标(≥2/3)": sum(
            1 for e, _ in b["errors"].values() if e <= GT_TOLERANCE
        ) >= 2 and len(b["errors"]) == len(EXPECTED_STATIC_ERRORS),
    }
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}")
    passed = all(checks.values())
    print()
    print("[PASS] 完整 TF 树验证通过" if passed else "[FAIL] 有检查项未通过")
    if passed:
        print("""
  这一段证明的是**位姿链路本身是对的**：
    · 四条边（map→odom→base_link→camera_link→optical）都真在 /tf_static 上；
    · 实测查询结果与独立算出的解析解逐位一致（四元数方向没有写反）；
    · 走 TF 建出来的地图与 static 模式**同一个数** —— 两条独立实现互相印证。
  边界：本 launch 的四条边都是**静态**的，所以它验证的是链路而非"移动建图"；
        真机移动场景由真机 TF / 里程计覆盖。
""")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
