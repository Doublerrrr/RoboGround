# -*- coding: utf-8 -*-
"""TF 树的**纯数据**描述（不 import rclpy / launch，因此可以离线测试）。

为什么把 TF 树单独抽成一个模块
============================
1. **可以离线验证**：`launch/*.launch.py` 需要 ROS2 才能 import，而"四元数方向对不对"
   这件事必须能在没装 ROS2 的机器上跑测试（本项目 386 个测试里绝大多数不依赖 ROS2）。
   把数字放进纯 Python 模块后，`tests/test_ros2_package.py` 能把链路**算出来**
   和 `euler_to_camera_pose(..., optical_frame_correction=True)` 对比。
2. **单一事实来源**：launch 文件、WSL 验证脚本、文档都引用这里的数字，
   不会出现"launch 里改了一个数、脚本里还是旧的"。

TF 的方向约定（这是最容易错、且错了完全静默的一步）
=================================================
tf2 里 `frame_id=parent, child_frame_id=child` 的一条边，语义是

    p_parent = R · p_child + t          （存的是"子系在父系中的位姿"）

于是 `camera_link → camera_color_optical_frame` 这条边要填的 R，
是 **optical → camera_link** 的旋转，即 `R_OPTICAL_TO_LINK`：

    右(x_cam) → −y_robot      下(y_cam) → −z_robot      前(z_cam) → +x_robot

它的四元数是 `(x, y, z, w) = (−0.5, 0.5, −0.5, 0.5)`。
**写成转置就错了**（`(0.5,−0.5,0.5,0.5)` 是另一个旋转），
实测会让相机位姿误差 1.697 m，而且不报任何错 —— 见 `docs/实现笔记.md`。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

#: 逻辑帧名 → 默认实际帧名。真机改名时传 `frames=` 覆盖即可。
DEFAULT_FRAMES: Dict[str, str] = {
    "map": "map",
    "odom": "odom",
    "base_link": "base_link",
    "camera_link": "camera_link",
    "optical": "camera_color_optical_frame",
}

#: 相机安装高度（米）：base_link → camera_link 的平移
CAMERA_HEIGHT_M = 1.2

#: `camera_link → optical` 的旋转四元数 (x, y, z, w) = R_OPTICAL_TO_LINK 的四元数
#: ⚠️ 方向是 **optical → camera_link**，不是它的转置。改之前先看模块 docstring。
OPTICAL_QUATERNION_XYZW: Tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5)

#: TF 树的边。每项： (父逻辑名, 子逻辑名, 平移, 四元数, 真机归属, 真机上是否动态)
#:
#: `owner` / `dynamic` 不是装饰性的：真机上只有最后两条边是我们能控制的，
#: 前两条由定位与里程计模块发布 —— 这一点必须说清楚，否则会被误读成
#: "RoboGround 就是这么发布 TF 的"。
EDGES: List[Tuple[str, str, Tuple[float, float, float],
                  Tuple[float, float, float, float], str, bool]] = [
    ("map", "odom", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
     "定位 / SLAM（AMCL、cartographer…）", True),
    ("odom", "base_link", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
     "轮式 / 视觉里程计", True),
    ("base_link", "camera_link", (0.0, 0.0, CAMERA_HEIGHT_M), (0.0, 0.0, 0.0, 1.0),
     "相机外参标定结果（通常写进 URDF）", False),
    ("camera_link", "optical", (0.0, 0.0, 0.0), OPTICAL_QUATERNION_XYZW,
     "相机驱动（realsense2_camera 等）", False),
]


# --------------------------------------------------------------------------
# 四元数 / 矩阵小工具（不依赖 numpy，保持本模块零依赖）
# --------------------------------------------------------------------------
def quaternion_to_matrix(q: Sequence[float]) -> List[List[float]]:
    """(x, y, z, w) → 3×3 旋转矩阵（与 `deployment/ros2/tf.py` 同约定）。"""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def matrix_to_quaternion(R: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """3×3 旋转矩阵 → (x, y, z, w)。"""
    r = [[float(v) for v in row] for row in R]
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2][1] - r[1][2]) / s
        y = (r[0][2] - r[2][0]) / s
        z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        w = (r[2][1] - r[1][2]) / s
        x = 0.25 * s
        y = (r[0][1] + r[1][0]) / s
        z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        w = (r[0][2] - r[2][0]) / s
        x = (r[0][1] + r[1][0]) / s
        y = 0.25 * s
        z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        w = (r[1][0] - r[0][1]) / s
        x = (r[0][2] + r[2][0]) / s
        y = (r[1][2] + r[2][1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mat_vec(A, v):
    return [sum(A[i][k] * float(v[k]) for k in range(3)) for i in range(3)]


def _transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def _compose(t1, q1, t2, q2):
    """先施加 (t2,q2) 再施加 (t1,q1)：p = R1(R2 p + t2) + t1。"""
    R1, R2 = quaternion_to_matrix(q1), quaternion_to_matrix(q2)
    R = _mat_mul(R1, R2)
    t = [a + b for a, b in zip(_mat_vec(R1, t2), [float(v) for v in t1])]
    return tuple(t), matrix_to_quaternion(R)


def _invert(t, q):
    """(parent←child) 的逆：(child←parent)。"""
    R = quaternion_to_matrix(q)
    Rt = _transpose(R)
    t = [float(v) for v in t]
    t_inv = [-v for v in _mat_vec(Rt, t)]
    return tuple(t_inv), matrix_to_quaternion(Rt)


def resolve_frames(frames: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    out = dict(DEFAULT_FRAMES)
    if frames:
        out.update({k: v for k, v in frames.items() if k in out})
    return out


def lookup_transform(target: str, source: str, *,
                     frames: Optional[Dict[str, str]] = None,
                     edges=None):
    """模拟 `tf2_ros.Buffer.lookup_transform(target, source)`。

    返回 `(translation, quaternion_xyzw)`，语义与 tf2 一致：
    **p_target = R · p_source + t**。

    参数用**实际帧名**（例如 `camera_color_optical_frame`），与 tf2 接口对齐。
    树不连通时抛 `LookupError`（真机 tf2 也是抛异常，不静默返回单位阵）。
    """
    fm = resolve_frames(frames)
    name_of = {logical: real for logical, real in fm.items()}
    logical_of = {real: logical for logical, real in name_of.items()}
    if target not in logical_of:
        raise LookupError(f"target frame {target!r} 不在 TF 树里（已知：{sorted(logical_of)}）")
    if source not in logical_of:
        raise LookupError(f"source frame {source!r} 不在 TF 树里（已知：{sorted(logical_of)}）")
    t_logical, s_logical = logical_of[target], logical_of[source]
    if t_logical == s_logical:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

    # 邻接表：child → parent 用原变换；parent → child 用逆
    adj: Dict[str, List[Tuple[str, tuple, tuple]]] = {}
    for parent, child, t, q, _owner, _dyn in (edges if edges is not None else EDGES):
        adj.setdefault(child, []).append((parent, tuple(t), tuple(q)))
        ti, qi = _invert(t, q)
        adj.setdefault(parent, []).append((child, ti, qi))

    # BFS：从 source 走到 target，沿途复合
    from collections import deque

    seen = {s_logical}
    dq = deque([(s_logical, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))])
    while dq:
        cur, acc_t, acc_q = dq.popleft()
        if cur == t_logical:
            return acc_t, acc_q
        for nxt, t, q in adj.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            # 走一条边：p_nxt = R·p_cur + t，再叠加上已有的 acc
            nt, nq = _compose(t, q, acc_t, acc_q)
            dq.append((nxt, nt, nq))
    raise LookupError(f"TF 树里 {source!r} 与 {target!r} 不连通")


def frame_names(frames: Optional[Dict[str, str]] = None) -> List[str]:
    """树上所有实际帧名（用于校验 `view_frames` 的结果）。"""
    fm = resolve_frames(frames)
    out: List[str] = []
    for _p, _c, _t, _q, _o, _d in EDGES:
        for logical in (_p, _c):
            real = fm[logical]
            if real not in out:
                out.append(real)
    return out
