"""数据层：真实数据集加载、虚拟视角重渲染、合成数据、自动化标注。"""

from roboground.data.sunrgbd import (
    SUNRGBDScene,
    build_scene_index,
    load_scene_index,
    load_sunrgbd_scene,
    scene_to_frame,
)
from roboground.data.synthetic import (
    make_synthetic_frame,
    make_synthetic_sequence,
    SyntheticRoom,
)
from roboground.data.virtual_camera import (
    PointCloudRenderer,
    orbit_poses,
    scene_sequence_from_cloud,
)

__all__ = [
    # SUN RGB-D
    "SUNRGBDScene",
    "build_scene_index",
    "load_scene_index",
    "load_sunrgbd_scene",
    "scene_to_frame",
    # 合成
    "SyntheticRoom",
    "make_synthetic_frame",
    "make_synthetic_sequence",
    # 虚拟视角
    "PointCloudRenderer",
    "orbit_poses",
    "scene_sequence_from_cloud",
]
