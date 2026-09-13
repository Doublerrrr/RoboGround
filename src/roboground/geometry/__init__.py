"""几何层：相机模型、2D-3D 投影/反投影、体素化与特征聚合。

这是整个 RoboGround 的地基 —— 所有"2D 语义升维到 3D"的魔法都在这里。

坐标约定（与 `roboground.types` 一致）
------------------------------------
- 相机系：OpenCV 约定，x 右 / y 下 / z 前（深度）
- 世界系：z 上
- 外参 `CameraPose`：world → camera，即 `p_cam = R @ p_world + t`
"""

from roboground.geometry.camera import (
    camera_ray_directions,
    pixel_grid,
    transform_points,
)
from roboground.geometry.projection import (
    backproject_detection,
    depth_to_points_camera,
    depth_to_points_world,
    project_points_to_image,
    raw_depth_to_meters,
    sample_intrinsics_like,
)
from roboground.geometry.voxel import (
    VoxelGrid,
    aggregate_features_by_voxel,
    cluster_points_dbscan,
    voxel_keys,
)

__all__ = [
    # camera
    "pixel_grid",
    "camera_ray_directions",
    "transform_points",
    # projection
    "raw_depth_to_meters",
    "depth_to_points_camera",
    "depth_to_points_world",
    "project_points_to_image",
    "backproject_detection",
    "sample_intrinsics_like",
    # voxel
    "VoxelGrid",
    "voxel_keys",
    "aggregate_features_by_voxel",
    "cluster_points_dbscan",
]
