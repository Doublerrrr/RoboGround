"""pytest 共享夹具与工具。

设计原则
--------
1. **测试不依赖网络与大模型**：默认全走离线后端（stub/box/color_hist）；
2. **测试不依赖数据集**：几何/建图/查询用合成场景（`data.synthetic`），
   完全确定、可复现；
3. **真实数据测试标记为 `slow`**：有 SUN RGB-D 时才跑，用 `-m slow` 启用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 让测试在未 pip install -e 的情况下也能 import（与 pyproject 的 pythonpath 一致）
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from roboground import load_config  # noqa: E402
from roboground.data.synthetic import (  # noqa: E402
    SyntheticRoom,
    default_intrinsics,
    make_synthetic_frame,
    make_synthetic_sequence,
)
from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame  # noqa: E402


# ==========================================================================
# 配置
# ==========================================================================
@pytest.fixture
def cfg():
    """默认配置 + 强制离线后端（保证测试无网络、无 GPU 也能跑）。"""
    c = load_config()
    c.set("perception.detector", "stub")
    c.set("perception.segmenter", "box")
    c.set("perception.encoder", "color_hist")
    c.set("project.verbose", False)
    c.set("perception.prompts", ["cup", "table", "chair", "box", "bottle"])
    return c


@pytest.fixture
def quiet():
    """让测试期间的日志静默（需要排查时注释掉这一行）。"""
    from roboground.utils.logging import set_verbosity

    set_verbosity(0)
    yield
    set_verbosity(1)


# ==========================================================================
# 几何与合成数据
# ==========================================================================
@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=120.0, fy=120.0, cx=79.5, cy=59.5, width=160, height=120)


@pytest.fixture
def flat_frame(intrinsics) -> RGBDFrame:
    """一帧"2 米处平面"的合成 RGB-D（深度恒为 2m）。"""
    height, width = 120, 160
    depth = np.full((height, width), 2.0, dtype=np.float32)
    color = np.full((height, width, 3), 120, dtype=np.uint8)
    return RGBDFrame(
        color=color, depth_m=depth, intrinsics=intrinsics,
        pose=CameraPose.identity(), frame_id="flat",
    )


@pytest.fixture
def room() -> SyntheticRoom:
    """确定性的合成房间（固定 seed，测试可复现）。"""
    return SyntheticRoom.random(seed=7, num_objects=5)


@pytest.fixture
def room_frames() -> list:
    """多视角合成序列（同一房间、相机沿 x 平移）。"""
    return make_synthetic_sequence(seed=7, num_frames=3, width=160, height=120, num_objects=5)


@pytest.fixture
def single_frame() -> RGBDFrame:
    return make_synthetic_frame(seed=11, width=160, height=120, num_objects=5)


# ==========================================================================
# 工具
# ==========================================================================
def make_observation(label: str, center, size, feature=None, frame_id: str = "f0"):
    """快速构造一个 `Observation`（多用于地图/关系测试）。"""
    from roboground.types import Detection2D, Observation

    center = np.asarray(center, dtype=np.float64).reshape(3)
    size = np.asarray(size, dtype=np.float64).reshape(3)
    half = np.clip(size, 1e-3, None) / 2.0
    # 生成 6 个面心附近的点，保证 bbox 约等于给定尺寸
    offsets = np.array([
        [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
    ], dtype=np.float64) * half
    points = center[None, :] + offsets

    det = Detection2D(
        label=label, score=0.9,
        bbox=np.array([10.0, 10.0, 20.0, 20.0]),
        feature=(None if feature is None else np.asarray(feature, dtype=np.float32)),
    )
    return Observation(
        detection=det, points_world=points, frame_id=frame_id,
    )


@pytest.fixture
def obs_factory():
    """返回 `make_observation` 工厂（避免测试里跨模块 import conftest）。"""
    return make_observation
