"""投影与反投影：深度 → 3D 点云 → 世界系；世界点 → 像素。

这是"2D 语义升维到 3D"的核心算子。所有函数都是纯 numpy、全向量化。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from roboground.geometry.camera import pixel_grid
from roboground.types import (
    CameraIntrinsics,
    CameraPose,
    Detection2D,
    Observation,
    RGBDFrame,
)


# ==========================================================================
# 深度预处理
# ==========================================================================
def raw_depth_to_meters(
    raw_depth: np.ndarray,
    scale: float = 1000.0,
    *,
    invalid_value: float = 0.0,
) -> np.ndarray:
    """原始 uint16 深度 → 米。

    SUN RGB-D 的约定是 `raw / 1000`（mm）；部分数据集是 `/10000`，
    所以 scale 必须由调用方显式给出（config 里是 `geometry.depth_scale`）。

    Parameters
    ----------
    raw_depth : np.ndarray
        原始深度图（uint16 或 float）。
    scale : float
        除数。`depth_m = raw / scale`。
    invalid_value : float
        原始图中代表"无效"的取值（通常是 0），转换后仍为 0。
    """
    raw = np.asarray(raw_depth).astype(np.float32)
    depth_m = raw / float(scale)
    if invalid_value is not None:
        depth_m = np.where(raw == invalid_value, 0.0, depth_m)
    return depth_m.astype(np.float32)


def filter_depth_ghosts(
    depth_m: np.ndarray,
    *,
    percentile: float = 99.0,
    min_depth: float = 0.1,
    max_depth: float = 8.0,
) -> np.ndarray:
    """剔除"幽灵点"：把超出高分位数的深度视为穿透产生的异常值。

    这是三维家项目踩出来的经验 —— 玻璃/透明材质会让深度信号穿透，
    在室外/后方产生虚远的点。分位数截断是成本最低、最有效的过滤手段。

    Returns
    -------
    np.ndarray
        过滤后的深度图（被剔除的位置置 0）。
    """
    d = np.asarray(depth_m, dtype=np.float32).copy()
    valid = np.isfinite(d) & (d > min_depth) & (d < max_depth)
    if not np.any(valid):
        return np.zeros_like(d)

    values = d[valid]
    if 0.0 < percentile < 100.0:
        cutoff = float(np.percentile(values, percentile))
        # 留一点余量，避免把正常远处物体误杀
        cutoff = max(cutoff, min_depth)
        valid &= d <= cutoff

    out = np.where(valid, d, 0.0).astype(np.float32)
    return out


# ==========================================================================
# 深度 → 3D 点
# ==========================================================================
def depth_to_points_camera(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    min_depth: float = 0.1,
    max_depth: float = 8.0,
    return_mask: bool = False,
):
    """深度图 → **相机系** 3D 点云。

    $$x = (u - c_x) \\cdot z / f_x,\\quad y = (v - c_y) \\cdot z / f_y,\\quad z = d$$

    Parameters
    ----------
    depth_m : (H, W) float
        深度，单位米。
    intrinsics : CameraIntrinsics
    min_depth, max_depth : float
        有效深度范围（米）。
    return_mask : bool
        是否额外返回 (H*W,) 的有效性掩码。

    Returns
    -------
    points : (N, 3) float64
        相机系点云（只包含有效像素）。
    mask : (H*W,) bool, optional
        仅当 `return_mask=True`。
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth 应为 (H,W)，收到 {depth.shape}")
    height, width = depth.shape

    # 若内参自带尺寸且不一致，说明调用方忘了同步内参 —— 早失败
    if intrinsics.width is not None and intrinsics.width != width:
        raise ValueError(
            f"内参 width={intrinsics.width} 与深度图 width={width} 不一致；"
            "resize 后必须同步缩放内参（见 sample_intrinsics_like）"
        )

    flat = depth.reshape(-1)
    valid = np.isfinite(flat) & (flat > min_depth) & (flat < max_depth)

    if not np.any(valid):
        pts = np.zeros((0, 3), dtype=np.float64)
        return (pts, valid) if return_mask else pts

    uv = pixel_grid(width, height, flatten=True)[valid]
    z = flat[valid].astype(np.float64)

    x = (uv[:, 0] - intrinsics.cx) * z / intrinsics.fx
    y = (uv[:, 1] - intrinsics.cy) * z / intrinsics.fy
    points = np.stack([x, y, z], axis=1)

    return (points, valid) if return_mask else points


def depth_to_points_world(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose: Optional[CameraPose] = None,
    *,
    min_depth: float = 0.1,
    max_depth: float = 8.0,
    return_mask: bool = False,
):
    """深度图 → **世界系** 3D 点云。"""
    points_cam, mask = depth_to_points_camera(
        depth_m, intrinsics,
        min_depth=min_depth, max_depth=max_depth, return_mask=True,
    )
    if pose is not None and points_cam.shape[0] > 0:
        points_world = pose.cam_to_world(points_cam)
    else:
        points_world = points_cam
    return (points_world, mask) if return_mask else points_world


def frame_to_pointcloud(
    frame: RGBDFrame,
    *,
    min_depth: float = 0.1,
    max_depth: float = 8.0,
    max_points: Optional[int] = 200_000,
    colors: bool = True,
    seed: int = 0,
):
    """一帧 RGB-D → 世界系彩色点云（供可视化与地图构建使用）。

    Returns
    -------
    points : (N, 3) float64
    colors_rgb : (N, 3) uint8, optional
        仅当 `colors=True`。
    """
    pts, mask = depth_to_points_world(
        frame.depth_m, frame.intrinsics, frame.pose,
        min_depth=min_depth, max_depth=max_depth, return_mask=True,
    )
    if max_points is not None and pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        idx.sort()
        pts = pts[idx]
        mask = np.flatnonzero(mask)[idx]
        full_mask = np.zeros(frame.height * frame.width, dtype=bool)
        full_mask[mask] = True
    else:
        full_mask = mask

    if not colors:
        return pts

    flat_rgb = frame.color.reshape(-1, 3)
    return pts, flat_rgb[full_mask]


# ==========================================================================
# 3D → 2D 投影
# ==========================================================================
def project_points_to_image(
    points_world: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """世界系点 → 像素坐标。

    Returns
    -------
    uv : (N, 2) float64
        像素坐标；视野外/在相机后方的点为 NaN。
    depth : (N,) float64
        相机系 z（米）。
    valid : (N,) bool
        是否投影到图像范围内。
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=bool)

    if width is None:
        width = intrinsics.width
    if height is None:
        height = intrinsics.height

    cam = pose.world_to_cam(pts)
    z = cam[:, 2]
    front = z > 1e-6

    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    safe_z = np.where(front, z, 1.0)
    uv[:, 0] = cam[:, 0] * intrinsics.fx / safe_z + intrinsics.cx
    uv[:, 1] = cam[:, 1] * intrinsics.fy / safe_z + intrinsics.cy
    # ⚠️ 必须显式把相机后方的点重新置为 NaN：
    # 上面的向量化赋值会覆盖掉初始化时的 NaN，若不重置，
    # 相机后方的点会得到一个"看起来合法"的像素坐标（镜像位置），
    # 下游据此画框会得到完全错误的检测结果。
    uv[~front] = np.nan

    if width is None or height is None:
        valid = front
    else:
        valid = (
            front
            & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
        )
    return uv, z, valid


# ==========================================================================
# 检测 → 3D 观测（感知层与映射层的桥）
# ==========================================================================
def _mask_from_bbox(bbox: np.ndarray, height: int, width: int) -> np.ndarray:
    """把 bbox 转成稠密 bool 掩码（无分割后端时的降级方案）。"""
    x1, y1, x2, y2 = np.round(bbox).astype(int)
    x1 = int(np.clip(x1, 0, width))
    x2 = int(np.clip(x2, 0, width))
    y1 = int(np.clip(y1, 0, height))
    y2 = int(np.clip(y2, 0, height))
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def unproject_pixels(frame: RGBDFrame, uv: np.ndarray,
                     depth_values: np.ndarray) -> np.ndarray:
    """像素 + 对应深度值 → **世界系** 3D 点（自动识别投影模型与深度语义）。

    · 针孔：深度是沿光轴的 **z 分量**，所以 `p_cam = z · k`（`k_z = 1`）；
    · 等距柱状全景：深度是从光心起算的**斜距 r**，
      所以 `p_world = C + r · d_world`。

    这两种语义**不能混用**：把斜距当 z 深度用（或反过来）会让远离光轴的
    像素沿视线方向被拉伸或压缩。
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(depth_values, dtype=np.float64).reshape(-1)
    K = frame.intrinsics

    if frame.meta.get("projection") == "equirect":
        W, H = float(K.width), float(K.height)
        az = (uv[:, 0] + 0.5) / W * 2.0 * np.pi - np.pi
        el = np.pi / 2.0 - (uv[:, 1] + 0.5) / H * np.pi
        cos_el = np.cos(el)
        d_cam = np.stack([cos_el * np.cos(az), cos_el * np.sin(az),
                          np.sin(el)], axis=1)
        R = np.asarray(frame.pose.R, dtype=np.float64).reshape(3, 3)
        # 相机系 → 世界系（旋转部分是 Rᵀ）
        d_world = d_cam @ R
        origin = np.asarray(frame.pose.camera_center(), dtype=np.float64).reshape(3)
        return origin[None, :] + z[:, None] * d_world

    # ★ 针孔通路刻意写成**与重构前逐字节等价**的形式：
    #   `x = (u−cx)·z/fx` → `stack([x, y, z])` → `cam_to_world`。
    #   等价但"更函数式"的写法（先归一化再乘回 z、或先算 k 再 `z·k`）
    #   实测要慢 29~60%（2000 点 36.6→47.4 µs；20000 点 540→866 µs），
    #   因为多了一次 `np.ones` 分配和一次 (n,3) 广播乘法。
    #   这条路径是 ROS2 实时链路和所有历史基准数字经过的地方，不能白白变慢。
    x = (uv[:, 0] - float(K.cx)) * z / float(K.fx)
    y = (uv[:, 1] - float(K.cy)) * z / float(K.fy)
    return frame.pose.cam_to_world(np.stack([x, y, z], axis=1))


def backproject_detection(
    detection: Detection2D,
    frame: RGBDFrame,
    *,
    min_depth: float = 0.1,
    max_depth: float = 8.0,
    ghost_percentile: Optional[float] = None,
    max_points: int = 20_000,
    seed: int = 0,
) -> Observation:
    """把一个 2D 检测/分割反投影成 3D 观测（世界系点集）。

    这是系统里最关键的一步：把"图上的语义"变成"空间里的语义"。

    Parameters
    ----------
    detection
        2D 检测（含 mask 或 bbox）。
    frame
        对应的 RGB-D 帧（提供深度与位姿）。
    ghost_percentile
        若给出，先做深度分位数截断过滤幽灵点（见 `filter_depth_ghosts`）。
    max_points
        每个实例最多保留多少点（防止单个大物体撑爆内存）。

    Returns
    -------
    Observation
        含 `points_world`（世界系 3D 点）的观测对象。
    """
    height, width = frame.shape

    # 1) 取掩码：优先用分割掩码，否则用 bbox 填充
    if detection.mask is not None:
        mask = np.asarray(detection.mask, dtype=bool)
        if mask.shape != (height, width):
            # 掩码尺寸与图像不一致：以 bbox 为准并给出提示
            mask = _mask_from_bbox(detection.bbox, height, width)
    else:
        mask = _mask_from_bbox(detection.bbox, height, width)

    if not np.any(mask):
        return Observation(
            detection=detection,
            points_world=np.zeros((0, 3)),
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )

    # 2) 深度预处理（可选幽灵点过滤）
    depth = frame.depth_m
    if ghost_percentile is not None:
        depth = filter_depth_ghosts(
            depth, percentile=ghost_percentile,
            min_depth=min_depth, max_depth=max_depth,
        )

    # 3) 掩码内的有效深度像素
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth) & mask
    if not np.any(valid):
        return Observation(
            detection=detection,
            points_world=np.zeros((0, 3)),
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )

    flat_depth = depth.reshape(-1)
    idx = np.flatnonzero(valid.reshape(-1))
    z = flat_depth[idx].astype(np.float64)

    uv = pixel_grid(width, height, flatten=True)[idx]
    # 4) 反投影到世界系（自动识别针孔 / 等距柱状全景，见 unproject_pixels）
    points_world = unproject_pixels(frame, uv, z)

    # 5) 点数上限保护
    if points_world.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(points_world.shape[0], size=int(max_points), replace=False)
        points_world = points_world[sel]

    return Observation(
        detection=detection,
        points_world=points_world,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
    )


def backproject_detections(
    detections,
    frame: RGBDFrame,
    **kwargs,
):
    """批量反投影（`backproject_detection` 的向量化外壳）。"""
    return [backproject_detection(det, frame, **kwargs) for det in detections]


# ==========================================================================
# 内参缩放
# ==========================================================================
def sample_intrinsics_like(
    intrinsics: CameraIntrinsics,
    width: int,
    height: int,
) -> CameraIntrinsics:
    """把内参重采样到目标宽高（resize 后必须调用）。

    要求宽高比一致（本项目只做等比例缩放）；若不一致会按各自轴分别缩放，
    此时图像会被拉伸，调用方需自行确保这与图像的实际 resize 方式一致。
    """
    if intrinsics.width is None or intrinsics.height is None:
        raise ValueError("原内参必须带 width/height 才能重采样")
    sx = float(width) / float(intrinsics.width)
    sy = float(height) / float(intrinsics.height)
    return CameraIntrinsics(
        fx=intrinsics.fx * sx,
        fy=intrinsics.fy * sy,
        cx=intrinsics.cx * sx,
        cy=intrinsics.cy * sy,
        width=int(width),
        height=int(height),
    )


def crop_intrinsics(
    intrinsics: CameraIntrinsics,
    x0: int,
    y0: int,
) -> CameraIntrinsics:
    """裁剪图像后更新主点（左上角裁掉 (x0, y0)）。"""
    return CameraIntrinsics(
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx - float(x0),
        cy=intrinsics.cy - float(y0),
        width=None if intrinsics.width is None else int(intrinsics.width - x0),
        height=None if intrinsics.height is None else int(intrinsics.height - y0),
    )
