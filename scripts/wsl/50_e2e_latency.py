#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""50 · 延迟与吞吐测量：把"一帧要多久、能扛多少 Hz"量清楚。

为什么不能只报一个"端到端延迟"
============================
第一版脚本按 10 Hz 发 640×480 的 RGB-D，然后统计"图像发布 → 地图收到"的时延，
结论是 4.1 s —— **这个数字是错的**（错在把测量工具的瓶颈算到了管线头上）。
真相是：管线每帧要 ~370 ms，10 Hz 的数据根本处理不过来，
消息在队列里越堆越多，于是"同步等待"变成了 1.2 s。

所以正确的做法是**分层测**，每层只问一个问题：

  A. 纯计算：一帧从 RGB-D 到并入地图要多久？（与 ROS 无关，可复现）
  B. 链路延迟：在**管线扛得住**的速率下，图像发布 → 地图收到要多久？
  C. 承载上限：加快速率到 10 Hz 会丢多少帧？（这才是"能不能上真机"的答案）

只在 A 上做优化、或只在 B 上报数字，都会得出误导性的结论。
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_SRC = Path("/mnt/g/RoboGround/src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, "/mnt/g/RoboGround/scripts/wsl")

RESOLUTIONS = [(320, 240), (640, 480)]
N_COMPUTE = 10
N_LINK = 12
LINK_HZ = 2.0            # 低于管线承载能力的速率，保证不丢帧
SATURATE_HZ = 10.0       # 快档目标速率（测丢帧率）
N_SATURATE = 40
#: 10 Hz 那一档重复几次。
#:
#: 为什么必须重复：实测同一台机器、同一份代码，**冷启动**那次 ROS 链路单帧
#: 从 42 ms 涨到 59.7 ms，10 Hz 丢帧率随之从 0% 变成 10%。
#: 21 Hz 的纯计算上限对 10 Hz 只有约 2 倍余量，机器状态一变就撑不住 ——
#: **单次测量得出的"0% 丢帧"是不可靠的结论，必须看分布。**
REPEATS = 3


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean_ms": float(a.mean() * 1000),
            "median_ms": float(np.median(a) * 1000),
            "p95_ms": float(np.percentile(a, 95) * 1000),
            "max_ms": float(a.max() * 1000), "min_ms": float(a.min() * 1000)}


def _cfg_for(width: int, height: int, *, publish_every_n: int = 10,
             pose_source: str = "static"):
    from roboground import load_config

    cfg = load_config()
    cfg.set("deploy.ros2.pose.source", pose_source)
    cfg.set("deploy.ros2.pose.translation", [0.0, 0.0, 1.2])
    cfg.set("deploy.ros2.pose.rpy", [0.0, 0.0, 0.0])
    cfg.set("deploy.ros2.pose.optical_frame_correction", True)
    cfg.set("deploy.ros2.sync_slop", 0.1)
    cfg.set("deploy.ros2.publish_every_n", publish_every_n)
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("perception.prompts", ["table", "chair", "cup"])
    cfg.set("geometry.depth_scale", 1000.0)
    cfg.set("project.verbose", False)
    return cfg


def _frames(width: int, height: int, n: int, cfg):
    """造 n 帧 RGBDFrame（含 GT 框注入，stub 检测器需要）。"""
    import ros2_e2e_test as T
    from roboground.types import CameraPose, RGBDFrame

    color, depth_m, K, gt = T.render_scene(width, height)
    pose = T.build_render_pose()
    sizes = {"table": [0.9, 0.9, 0.75], "chair": [0.5, 0.5, 0.9], "cup": [0.12, 0.12, 0.14]}
    boxes = np.stack([np.concatenate([np.asarray(c, float), sizes.get(l, [0.3] * 3), [0.0]])
                      for l, c in gt]).astype(np.float32)
    labels = [str(l) for l, _ in gt]
    out = []
    for i in range(n):
        f = RGBDFrame(color=np.asarray(color, dtype=np.uint8),
                      depth_m=np.asarray(depth_m, dtype=np.float32),
                      intrinsics=K, pose=CameraPose(pose.R.copy(), pose.t.copy()),
                      frame_id=f"f{i}")
        f.meta["boxes_3d"] = boxes
        f.meta["labels"] = labels
        out.append(f)
    return out


# =============================================================================
# A. 纯计算：一帧多久（无 ROS）
# =============================================================================
def part_a_scaling() -> Dict:
    """[A4] 吞吐随**目标数量**的伸缩：能带多少个物体跑到 10 Hz？

    为什么这一项最关键
    ================
    每帧耗时几乎正比于"检测到的目标数"（每个目标都要做掩码取点 + 体素融合）。
    所以"单帧 42 ms"这种数字只有在**明确目标数**的前提下才有意义 ——
    本项目就出现过：漏注入 GT 框，stub 检测器自己造了 15 个框，
    于是同一段代码测出 400+ ms 并得出"跑不到 10 Hz"的**错误结论**。
    """
    from roboground.data.synthetic import SceneObject, SyntheticRoom
    from roboground.mapping import MapBuilder
    from roboground.types import CameraPose

    import ros2_e2e_test as T

    print("\n  [A4] 吞吐 vs 目标数（320×240，只改目标个数，其余不变）")
    out = {}
    pose = T.build_render_pose()
    # ★ 复用 `render_scene` 里**已验证能被渲染出来**的三个物体（0.9/0.5/0.12 m），
    #   沿 +x / +y 平铺成多组，而不是自己另造一套摆放方式。
    #   教训：上一版把 0.5 m 的小方块摆在 14×14 m 的大房间里，
    #   渲染器（射线步进）根本采不到它们 → 地图物体 0、
    #   耗时 0.1 ms、算出"6784 Hz"这种荒谬数字。曲线必须建立在
    #   "目标真的被渲染出来且被建进地图"之上（下面的 check 会守这一点）。
    base = [
        ("table", (0.0, -0.3, 0.4), (0.9, 0.9, 0.75), (150, 110, 80)),
        ("chair", (0.0, 0.6, 0.45), (0.5, 0.5, 0.9), (90, 90, 140)),
        ("cup", (0.0, 0.0, 0.85), (0.12, 0.12, 0.14), (230, 230, 240)),
    ]
    layout = [(2.0, 0.0), (3.2, 0.0), (4.4, 0.0), (2.0, 1.2), (3.2, 1.2), (4.4, 1.2)]
    for n_obj in (3, 6, 9, 12, 18):
        objects = []
        for (dx, dy) in layout:
            for (lbl, c, sz, col) in base:
                if len(objects) >= n_obj:
                    break
                objects.append(SceneObject(
                    lbl, [c[0] + dx, c[1] + dy, c[2]], list(sz), 0.0, col))
            if len(objects) >= n_obj:
                break
        room = SyntheticRoom(width=6.0, depth=6.0, height=2.6, objects=objects)
        res = room.render(pose, T._intrinsics_for(320, 240, 60.0))
        valid_px = int((np.asarray(res.depth_m) > 0.1).sum())

        cfg = _cfg_for(320, 240)
        builder = MapBuilder(cfg)
        boxes = np.stack([np.concatenate([np.asarray(o.center, float),
                                          np.asarray(o.size, float), [0.0]])
                          for o in objects]).astype(np.float32)
        labels = [o.label for o in objects]
        per = []
        for i in range(N_COMPUTE):
            from roboground.types import RGBDFrame

            f = RGBDFrame(color=np.asarray(res.color, dtype=np.uint8),
                          depth_m=np.asarray(res.depth_m, dtype=np.float32),
                          intrinsics=res.intrinsics,
                          pose=CameraPose(pose.R.copy(), pose.t.copy()), frame_id=f"f{i}")
            f.meta["boxes_3d"] = boxes
            f.meta["labels"] = labels
            t = time.perf_counter()
            builder.add_frame(f)
            per.append(time.perf_counter() - t)
        smap = builder.finalize()
        s = _stats(per)
        hz = 1000.0 / max(s["mean_ms"], 1e-9)
        out[n_obj] = {"per_frame_ms": s["mean_ms"], "hz": hz,
                      "n_objects_map": smap.num_objects, "valid_px": valid_px,
                      "merged": smap.num_objects != n_obj,
                      "no_object": smap.num_objects == 0}
        flag = "✓" if hz >= 10.0 else "✗"
        # 地图物体数少于目标数是**正常现象**（互相遮挡 / 同名同区域被关联合并），
        # 真正决定耗时的是"每帧送进去多少个检测"（= n_obj），所以只把它作为信息打印；
        # 但"一个物体都没建出来"就必须报警（那说明这一档是空转，曲线不可信）。
        note = ""
        if out[n_obj]["no_object"]:
            note = "  ⚠ 一个物体都没建出来 → 这一档是空转，曲线不可信"
        elif out[n_obj]["merged"]:
            note = (f"（地图 {smap.num_objects} 个：部分目标互相遮挡"
                    "或被关联合并，属正常）")
        print(f"    {n_obj:>3} 个检测/帧（有效深度 {valid_px:>6} px） → "
              f"{s['mean_ms']:7.1f} ms/帧 ⇒ {hz:5.2f} Hz  {flag}  {note}")
    out["_note"] = ("耗时随**目标数**与**掩码像素面积**增长；本表刻意复用已验证的"
                    "物体尺寸，避免'目标没被渲染出来'这种假曲线")
    return out


def part_a_compute() -> Dict:
    """[A] 纯计算吞吐（不经过 ROS，无队列干扰）。"""
    from roboground.mapping import MapBuilder

    out = {}
    print("\n[A] 纯计算吞吐（不经过 ROS，无队列干扰）")
    for (w, h) in RESOLUTIONS:
        cfg = _cfg_for(w, h)
        frames = _frames(w, h, N_COMPUTE, cfg)
        builder = MapBuilder(cfg)
        per_frame = []
        for f in frames:
            t = time.perf_counter()
            builder.add_frame(f)
            per_frame.append(time.perf_counter() - t)
        t = time.perf_counter()
        smap = builder.finalize()
        finalize_s = time.perf_counter() - t
        n_points = int((np.asarray(frames[0].depth_m) > 0.1).sum())
        s = _stats(per_frame)
        out[f"{w}x{h}"] = {"per_frame": s, "finalize_ms": finalize_s * 1000,
                           "n_objects": smap.num_objects, "valid_depth_px": n_points,
                           "sustain_hz": 1.0 / max(s["mean_ms"] / 1000, 1e-9)}
        print(f"  {w}×{h}（有效深度 {n_points} px，{smap.num_objects} 物体）")
        print(f"    add_frame   mean {s['mean_ms']:7.1f} ms   median {s['median_ms']:7.1f} ms"
              f"   p95 {s['p95_ms']:7.1f} ms   max {s['max_ms']:7.1f} ms")
        print(f"    finalize()  {finalize_s * 1000:7.1f} ms")
        print(f"    → 单帧预算 ≈ {s['mean_ms']:.0f} ms ⇒ 稳态上限 ≈ {out[f'{w}x{h}']['sustain_hz']:.2f} Hz")

    # 640×480 的分阶段剖析（回答"时间花在哪"）
    print("\n  [A2] 640×480 单帧剖析（cProfile，累计耗时前 8）")
    cfg = _cfg_for(640, 480)
    frames = _frames(640, 480, 5, cfg)
    builder = MapBuilder(cfg)
    pr = cProfile.Profile()
    pr.enable()
    for f in frames:
        builder.add_frame(f)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(8)
    lines = [ln for ln in buf.getvalue().splitlines()
             if "cumtime" in ln or ("/" in ln and "{" not in ln)]
    for ln in lines[:12]:
        print("    " + ln.strip()[:150])
    out["profile_top"] = lines[:12]

    # [A3] `frame_from_streams`（深度图 → 世界点云）单独计时。
    #
    # 为什么必须单独量：A 是直接喂 `RGBDFrame` 给 `add_frame`，**跳过了这一步**；
    # 而真机上这一步每帧都要做。不量它就会出现"A 说 39 ms、ROS 链路上说 104 ms"
    # 这种对不上的数字（本次测量就出现过，差的就是这 60 多毫秒）。
    print("\n  [A3] frame_from_streams（深度图 → 世界点云）单独计时")
    from roboground.deployment.ros2.bridge import frame_from_streams
    from roboground.types import CameraIntrinsics

    class _FixedPose:
        """最小的位姿提供者（不需要 ROS2），用于离线计时。"""
        name = "fixed"

        def __init__(self, pose):
            self._pose = pose

        def get_pose(self, stamp=None, frame_id="map"):
            return self._pose

        def describe(self):
            return {"name": self.name}

    out["bridge"] = {}
    for (w, h) in RESOLUTIONS:
        cfg = _cfg_for(w, h)
        color, depth_m, K, gt = __import__("ros2_e2e_test").render_scene(w, h)
        pose = __import__("ros2_e2e_test").build_render_pose()
        depth_raw = np.clip(np.asarray(depth_m) * 1000.0, 0, 65535).astype(np.uint16)
        provider = _FixedPose(pose)
        times = []
        for i in range(N_COMPUTE):
            t = time.perf_counter()
            fr = frame_from_streams(np.asarray(color, dtype=np.uint8), depth_raw, K,
                                    provider, stamp=float(i) * 0.1,
                                    depth_scale=1000.0, frame_id=f"f{i}",
                                    min_depth=0.1, max_depth=8.0, autosync=False)
            times.append(time.perf_counter() - t)
            assert fr is not None, "frame_from_streams 返回 None（位姿/深度有问题）"
        s = _stats(times)
        n_pts = int(fr.points_world.shape[0]) if hasattr(fr, "points_world") else -1
        out["bridge"][f"{w}x{h}"] = {"time": s, "n_points": n_pts}
        n_det = int(fr.meta.get("boxes_3d").shape[0]) if fr.meta.get("boxes_3d") is not None else 0
        print(f"    {w}×{h}  mean {s['mean_ms']:7.1f} ms   p95 {s['p95_ms']:7.1f} ms"
              f"   （帧对象点数属性 {n_pts}）")
        total = out[f"{w}x{h}"]["per_frame"]["mean_ms"] + s["mean_ms"]
        out[f"{w}x{h}"]["total_ms"] = total
        out[f"{w}x{h}"]["total_hz"] = 1000.0 / max(total, 1e-9)
        print(f"    → 端到端单帧预算 ≈ {s['mean_ms']:.0f}（桥接）+ "
              f"{out[f'{w}x{h}']['per_frame']['mean_ms']:.0f}（管线）= {total:.0f} ms"
              f" ⇒ {out[f'{w}x{h}']['total_hz']:.2f} Hz")
    return out


# =============================================================================
# B/C. ROS 链路：低速率测延迟、高速率测丢帧
# =============================================================================
def _ros_run(hz: float, n_frames: int, width: int, height: int, *,
             publish_every_n: int = 10) -> Dict:
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    import ros2_e2e_test as T
    from roboground.deployment.ros2.nodes import PerceptionNode

    color, depth_m, K, gt = T.render_scene(width, height)
    cfg = _cfg_for(width, height, publish_every_n=publish_every_n)

    rclpy.init()
    node = PerceptionNode(cfg, node_name="roboground_latency")
    gt_list = [(str(l), np.asarray(c, dtype=float)) for l, c in gt]
    sizes = {"table": [0.9, 0.9, 0.75], "chair": [0.5, 0.5, 0.9], "cup": [0.12, 0.12, 0.14]}
    boxes = np.stack([np.concatenate([c, sizes.get(l, [0.3] * 3), [0.0]])
                      for l, c in gt_list]).astype(np.float32)
    labels = [l for l, _ in gt_list]
    _orig_add = node.builder.add_frame
    perceive_ms: List[float] = []

    def _timed_add(frame, **kw):
        # ★ 必须注入 GT 框：`stub` 检测器靠 `frame.meta["boxes_3d"]` 拿目标。
        # 不注入的话它会按 prompts 自己造一堆框（实测 15 个/帧 vs 3 个），
        # 于是"纯计算 A"和"ROS 链路 B"测的根本不是同一个工作量 ——
        # 第一版就因此得到 42 ms vs 99 ms 的对不上的结果。
        frame.meta["boxes_3d"] = boxes
        frame.meta["labels"] = labels
        t = time.perf_counter()
        out = _orig_add(frame, **kw)
        perceive_ms.append(time.perf_counter() - t)
        return out

    node.builder.add_frame = _timed_add

    recv_stamps: List[float] = []
    sync_wait: List[float] = []

    # ★ 打点方式：包装 `process_frame`（**公开方法**），不要重新注册同步回调！
    #
    # 为什么不能 `node._sync.registerCallback(wrapper)`：
    #   `message_filters.SimpleFilter.registerCallback` 是**追加**语义
    #   （内部是 `self.callbacks[len(self.callbacks)] = (cb, args)`，不是替换）。
    #   于是原来注册的 `_on_synced` **仍在**，wrapper 里又调用了一次 `_orig_synced`
    #   ⇒ 每个同步帧被处理**两遍**，单帧耗时直接翻倍、`frames_processed` 翻倍，
    #   而所有数字看起来都"挺合理"，极难察觉。
    #   （实测：12 次同步 → 24 帧被处理。见 `scripts/wsl/44_diag_register_callback.sh`。）
    #
    # 包 `process_frame` 则没有这个问题：`_on_synced` 内部是
    # `self.process_frame(...)`，实例属性会遮蔽类方法，所以只会调用一次；
    # 而且图像时间戳就是它的关键字参数 `stamp`，正好用来算"同步等待"。
    _orig_pf = node.process_frame

    def _wrap_pf(color, depth_raw, intrinsics, *, stamp=None, frame_id=None):
        t_enter = time.time()
        if stamp is not None:
            sync_wait.append(t_enter - float(stamp))
        return _orig_pf(color, depth_raw, intrinsics, stamp=stamp, frame_id=frame_id)

    node.process_frame = _wrap_pf

    maps: List[float] = []
    state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL,
                           history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(String, "/roboground/semantic_map",
                             lambda m: maps.append(time.time()), state_qos)

    stop = threading.Event()

    def _spin():
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.02)

    th = threading.Thread(target=_spin, daemon=True)
    th.start()

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
        m.header.frame_id = "camera_color_optical_frame"
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
        m.header.frame_id = "camera_color_optical_frame"
        m.height, m.width = height, width
        m.k = [float(K.fx), 0.0, float(K.cx), 0.0, float(K.fy), float(K.cy), 0.0, 0.0, 1.0]
        m.d = [0.0] * 5
        m.distortion_model = "plumb_bob"
        return m

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if all(p.get_subscription_count() > 0 for p in (pub_c, pub_d, pub_i)):
            break
        time.sleep(0.1)

    sent: List[float] = []
    period = 1.0 / hz
    t0 = time.time()
    for i in range(n_frames):
        ts = time.time()
        stamp_ns = int(ts * 1e9)
        pub_c.publish(_img(np.asarray(color, dtype=np.uint8), "rgb8", stamp_ns))
        pub_d.publish(_img(depth_raw, "16UC1", stamp_ns))
        pub_i.publish(_info(stamp_ns))
        sent.append(ts)
        sleep = (t0 + (i + 1) * period) - time.time()
        if sleep > 0:
            time.sleep(sleep)

    # 等到"发完 + 管线把队列吃完"，最多等 5 s
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if len(sync_wait) >= n_frames:
            break
        time.sleep(0.1)
    time.sleep(1.0)
    stop.set()
    th.join(timeout=3.0)

    st = node.stats()
    smap = node.builder.finalize()
    n_objects = smap.num_objects
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass

    n_synced = len(sync_wait)
    frames_processed = int(st.get("frames_processed", 0))
    # ★ 打点自检：每帧只能被处理一次。
    # 有了这条断言，"registerCallback 追加导致每帧处理两遍"那类 bug
    # 会当场暴露，而不是悄悄把耗时放大一倍。
    double_processed = len(perceive_ms) != frames_processed

    return {
        "hz": hz, "n_sent": n_frames, "n_synced": n_synced,
        "drop_rate": 1.0 - n_synced / max(n_frames, 1),
        "frames_processed": frames_processed,
        "double_processed": bool(double_processed),
        "pose_failures": int(st.get("pose_failures", 0)),
        "n_objects": n_objects, "n_maps": len(maps),
        "sync_wait": _stats(sync_wait),
        "perceive": _stats(perceive_ms),
        "total": _stats(sync_wait),
        "size": f"{width}x{height}",
    }


def _print_link(r: Dict, label: str) -> None:
    print(f"  {label}")
    print(f"    发送 {r['n_sent']} 帧 → 同步成功 {r['n_synced']} 帧 "
          f"（丢帧率 {r['drop_rate'] * 100:.1f}%）/ 处理 {r['frames_processed']} 帧 / "
          f"位姿失败 {r['pose_failures']} / 地图 {r['n_maps']} 条 / 物体 {r['n_objects']}")
    if r["double_processed"]:
        print("    ✗ 打点自检失败：add_frame 调用次数 ≠ frames_processed"
              "（可能又被处理了两遍，数字不可信）")
    for key, name in (("sync_wait", "同步等待（图像时间戳→进回调）"),
                      ("perceive", "感知+融合（单帧）")):
        s = r[key]
        if not s.get("n"):
            print(f"    {name:<28} 无样本")
            continue
        print(f"    {name:<28} n={s['n']:<3} mean {s['mean_ms']:8.1f} ms   "
              f"median {s['median_ms']:8.1f} ms   p95 {s['p95_ms']:8.1f} ms   "
              f"max {s['max_ms']:8.1f} ms")


def main() -> int:
    print("=" * 84)
    print("RoboGround × ROS2 延迟与吞吐测量")
    print("=" * 84)

    a = part_a_compute()
    a4 = part_a_scaling()

    print(f"\n[B] ROS 链路延迟（{LINK_HZ:.0f} Hz，低于承载能力 → 应当不丢帧）")
    b = _ros_run(LINK_HZ, N_LINK, 320, 240)
    _print_link(b, "320×240 @ 2 Hz")

    print(f"\n[C] 承载上限（{SATURATE_HZ:.0f} Hz，快档目标速率 → 看丢帧率）")
    print(f"    重复 {REPEATS} 次：丢帧率受**机器状态**影响（冷启动/缓存/竞争），"
          "单次结果不能当结论")
    c_runs = []
    for i in range(REPEATS):
        r = _ros_run(SATURATE_HZ, N_SATURATE, 320, 240)
        c_runs.append(r)
        _print_link(r, f"320×240 @ 10 Hz（第 {i + 1}/{REPEATS} 次）")
    c = c_runs[0]
    drops = [r["drop_rate"] for r in c_runs]
    perceives = [r["perceive"]["mean_ms"] for r in c_runs]
    drop_worst, drop_best = max(drops), min(drops)
    perceive_worst = max(perceives)

    # ---- 判定：只对"该做到的事"设阈值 ----
    print("\n" + "=" * 84)
    print("结果")
    print("=" * 84)
    hz10_ok = [n for n, d in a4.items() if isinstance(n, int) and d["hz"] >= SATURATE_HZ]
    a4_nums = {n: d for n, d in a4.items() if isinstance(n, int)}
    checks = {
        "A 纯计算两档分辨率都有结果": all(
            a[f"{w}x{h}"]["n_objects"] > 0 for w, h in RESOLUTIONS),
        "A 320×240 单帧（桥接+管线）< 200 ms（能跑 5 Hz 以上）":
            a["320x240"]["total_ms"] < 200,
        "A 分辨率越高越慢（点数 ∝ 像素数）":
            a["640x480"]["total_ms"] > a["320x240"]["total_ms"],
        "A4 目标越多越慢（单调不降）":
            all(a4_nums[p]["per_frame_ms"] <= a4_nums[q]["per_frame_ms"] + 1e-6
                for p, q in zip(sorted(a4_nums), sorted(a4_nums)[1:])),
        "A4 每一档都真的建出了物体（不是空转）":
            not any(d["no_object"] for d in a4_nums.values()),
        "B 低速率下不丢帧（链路本身没问题）": b["drop_rate"] < 0.05,
        "B 无位姿失败": b["pose_failures"] == 0,
        "★ 打点自检：没有一帧被处理两遍":
            not any(r["double_processed"] for r in (b, *c_runs)),
        "B 同步等待 < 300 ms（速率低于承载能力时）": b["sync_wait"]["mean_ms"] < 300,
        # ⚠️ 这里**不**判定"10 Hz 必须 0% 丢帧"。
        #
        # 原因是实测过：同一台机器、同一份代码，冷启动那次 ROS 链路单帧
        # 从 42 ms 涨到 59.7 ms，10 Hz 丢帧率就从 0% 变成 10%（21 Hz 上限
        # 对 10 Hz 只有约 2 倍余量，撑不住竞争）。**单次测量的结论不成立。**
        # 所以只做一条"没有整体崩掉"的宽松下界，真实能力靠**重复测量的分布**陈述。
        "C 10 Hz 重复测量未崩溃（丢帧率最差 < 50%）": drop_worst < 0.50,
    }
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}")

    print("\n  结论（按实测数据写，不写想当然的话）：")
    for (w, h) in RESOLUTIONS:
        d = a[f"{w}x{h}"]
        br = a["bridge"][f"{w}x{h}"]
        print(f"    {w}×{h}: 桥接(深度→点云) {br['time']['mean_ms']:.1f} ms + "
              f"管线(感知+融合) {d['per_frame']['mean_ms']:.0f} ms "
              f"= {d['total_ms']:.0f} ms/帧 ⇒ 稳态上限 {d['total_hz']:.2f} Hz"
              f"（地图发布另加 {d['finalize_ms']:.1f} ms，每 N 帧一次）")
    print(f"    目标数伸缩（320×240，每帧送入 N 个检测）："
          + "，".join(f"N={n} → {d['hz']:.1f} Hz" for n, d in sorted(a4_nums.items())))
    if hz10_ok:
        print(f"    ⇒ 在 320×240 下：目标数 ≤ {max(hz10_ok)} 时能跑到 "
              f"{SATURATE_HZ:.0f} Hz；超过就掉帧")
    else:
        print(f"    ⇒ 320×240 下连 {min(a4_nums)} 个目标都跑不到 {SATURATE_HZ:.0f} Hz")
    print(f"    B 档（管线扛得住的速率）延迟：同步等待 mean {b['sync_wait']['mean_ms']:.1f} ms，"
          f"单帧感知 {b['perceive']['mean_ms']:.1f} ms")
    print(f"    ★ {SATURATE_HZ:.0f} Hz 重复 {REPEATS} 次：丢帧率 "
          + " / ".join(f"{d * 100:.1f}%" for d in drops)
          + f"（最差 {drop_worst * 100:.1f}%，最好 {drop_best * 100:.1f}%）")
    print(f"      对应的单帧感知耗时：" + " / ".join(f"{p:.1f} ms" for p in perceives))
    print(f"    ⇒ 结论要这么说：**{SATURATE_HZ:.0f} Hz 处于能力边缘** —— "
          f"纯计算上限 {a['320x240']['total_hz']:.0f} Hz 只有约 "
          f"{a['320x240']['total_hz'] / SATURATE_HZ:.1f} 倍余量，"
          "机器状态一变（冷启动 / 竞争）就掉帧。"
          "要稳定达标得把单帧预算压到 ~20 ms 以内，而不是指望它刚好够。")

    passed = all(checks.values())
    print()
    print("[PASS] 延迟与吞吐测量完成" if passed else "[FAIL] 有指标不达标")
    print("""
  诚实边界：
    · 发布者与订阅者在**同一进程**、DDS 走回环 → 链路延迟量到的是**下界**；
    · 每帧耗时**正比于目标个数**，所以任何"单帧 xx ms"都必须连目标数一起说；
    · 640×480 比 320×240 慢约 3 倍（点数 ∝ 像素数）→ 高分辨率下 10 Hz 不达标；
    · ★ **10 Hz 丢帧率必须看重复测量的分布，不能看单次**：
      实测冷启动那次单帧 59.7 ms → 丢帧 10%；机器热起来后 40~47 ms → 丢帧 0%。
      写文档时只能写"32×240 单帧 43~47 ms（21~23 Hz），10 Hz 处于能力边缘"，
      **不能写"10 Hz 丢帧率 0%"**（那是挑了一次好结果）。
    · 测量工具本身也会骗人：本脚本第一版把 `builder.add_frame` 包了一层却忘了
      注入 GT 框（stub 检测器于是自造 15 个框），又用 `registerCallback` 打点
      （它是**追加**语义 → 每帧被处理两遍），两个错误叠加后得出
      "单帧 553 ms、跑不到 10 Hz"的**完全错误**结论。现在两处都有自检断言。
""")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
