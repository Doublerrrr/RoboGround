"""数据层：真实数据集加载、**多视角全景融合**、合成数据、自动化标注。

⚠️ 这里**不再有"虚拟视角"**（点云 splatting 重渲染）。
上一版有 `virtual_camera.py`：拿**一个**真实视角造出 N 个**假的**视点。
那是同一份观测的重采样，**信息量不增加** —— 实测无效深度像素从 42.8%
涨到 77.0%（见 `docs/多视角数据核查报告.md`）。

现在换成方向相反的做法：`panorama.py` 把 **N 个真实视角**融合成
**一张全景**，每个像素都来自真实采集，信息量随视角数**真的增加**
（实测 office_6：6 视角 → 19.7%、48 视角 → 51.6%，见
`runs/36_pano_dataset_stats.json`）。入口在 `pano_scene.py`。

2D-3D-S 的子模块**故意不在这里 re-export**：它们依赖 PIL 且要指定数据根目录，
按需 `from roboground.data.pano_scene import load_scene` 更清楚。
"""

from roboground.data.panorama import (
    Panorama,
    frame_for_panorama,
    fuse_to_equirect,
    panorama_rays_world,
)
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

__all__ = [
    # 多视角 → 全景融合（真实数据主链路）
    "Panorama",
    "fuse_to_equirect",
    "frame_for_panorama",
    "panorama_rays_world",
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
]
