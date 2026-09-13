"""相机模型的底层几何运算：像素网格、射线方向、刚体变换。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from roboground.types import CameraIntrinsics, CameraPose


def pixel_grid(width: int, height: int, *, flatten: bool = True) -> np.ndarray:
    """生成像素坐标网格。

    Parameters
    ----------
    width, height : int
    flatten : bool
        True 返回 (H*W, 2)（行优先，与 `depth.reshape(-1)` 对齐）；
        False 返回 (H, W, 2)。

    Returns
    -------
    np.ndarray
        每行是 `(u, v)` = `(列, 行)`。
    """
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)          # (H, W)
    if flatten:
        return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)
    return np.stack([uu, vv], axis=-1)


def camera_ray_directions(
    intrinsics: CameraIntrinsics,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> np.ndarray:
    """每个像素在**相机系**下的单位射线方向（z 轴向前）。

    Returns
    -------
    np.ndarray, shape (H*W, 3)
        归一化方向向量。乘上深度即得到 3D 点（见 `depth_to_points_camera`）。
    """
    if width is None:
        width = intrinsics.width
    if height is None:
        height = intrinsics.height
    if width is None or height is None:
        raise ValueError("必须提供 width/height，或让 intrinsics 携带它们")

    uv = pixel_grid(int(width), int(height), flatten=True)
    x = (uv[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (uv[:, 1] - intrinsics.cy) / intrinsics.fy
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=1)
    norm = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / np.clip(norm, 1e-12, None)


def transform_points(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """对 (N,3) 点集施加刚体变换 `p' = R @ p + t`。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return pts @ R.T + t


def compose_poses(outer: CameraPose, inner: CameraPose) -> CameraPose:
    """位姿复合：先施加 inner 再施加 outer。

    `p_out = R_o (R_i p + t_i) + t_o`
    """
    return CameraPose(
        R=outer.R @ inner.R,
        t=outer.R @ inner.t + outer.t,
    )


def invert_pose(pose: CameraPose) -> CameraPose:
    """位姿求逆。"""
    return pose.inverse()


def points_in_camera_frustum(
    points_world: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    width: int,
    height: int,
    *,
    margin: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """判断世界系点是否落在某相机视野内。

    Returns
    -------
    (mask, uv)
        `mask` 是 (N,) bool；`uv` 是 (N,2) 像素坐标（视野外的点也会算出，
        但调用方应只使用 mask 为 True 的行）。
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool), np.zeros((0, 2), dtype=np.float64)
    cam = pose.world_to_cam(pts)
    z = cam[:, 2]
    valid = z > 1e-6
    uv = np.full((pts.shape[0], 2), np.nan)
    safe_z = np.where(valid, z, 1.0)
    uv[:, 0] = cam[:, 0] * intrinsics.fx / safe_z + intrinsics.cx
    uv[:, 1] = cam[:, 1] * intrinsics.fy / safe_z + intrinsics.cy
    inside = (
        valid
        & (uv[:, 0] >= -margin)
        & (uv[:, 0] <= width - 1 + margin)
        & (uv[:, 1] >= -margin)
        & (uv[:, 1] <= height - 1 + margin)
    )
    return inside, uv


def look_at_pose(
    eye,
    target,
    *,
    world_up=(0.0, 0.0, 1.0),
) -> CameraPose:
    """构造「站在 `eye`、看向 `target`」的相机位姿（**world → camera**）。

    约定与本项目其余部分一致：`CameraPose` 存 world→camera，
    所以返回的 `R` 满足 `p_cam = R @ p_world + t`，光心 `C = −Rᵀt = eye`。

    这里用的是 OpenCV 相机系（x 右、y 下、z 前）：先定 `z_cam = target − eye`，
    再用 `world_up` 叉乘得到 `x_cam`，最后 `y_cam = z_cam × x_cam`。

    ⚠️ 这个函数原先住在 `data/virtual_camera.py`（那个模块用来"造虚拟视角"，
    已删除）。它的用途是**构造测试与验证用的已知位姿**，属于相机几何，
    所以搬到了 `geometry/` 下 —— 与"虚拟视角"这个概念无关。
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(world_up, dtype=np.float64).reshape(3)

    z_cam = target - eye
    norm = np.linalg.norm(z_cam)
    if norm < 1e-9:
        return CameraPose.identity()
    z_cam = z_cam / norm

    x_cam = np.cross(up, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:          # 视线与 up 平行，换一个参考轴
        x_cam = np.cross(np.array([1.0, 0.0, 0.0]), z_cam)
    x_cam = x_cam / max(np.linalg.norm(x_cam), 1e-9)
    y_cam = np.cross(z_cam, x_cam)

    R_cam_to_world = np.stack([x_cam, y_cam, z_cam], axis=1)
    R = R_cam_to_world.T
    t = -R @ eye
    return CameraPose(R, t)
