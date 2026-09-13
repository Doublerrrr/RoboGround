"""虚拟相机：从点云在**新视角**重渲染 RGB-D（splatting + z-buffer）。

为什么需要它
------------
SUN RGB-D 每个场景只有**一张** RGB-D 图。但 Stage 2 的价值在于
**多视角融合**（同一物体被多帧看到 → 特征平均 → 抗噪），所以我们需要
同一个场景的多视角观测。做法：

1. 用原始单帧的深度 + 内参反投影出世界系点云（带颜色）；
2. 在点云周围架设若干**虚拟相机**（内外参由我们自己定义，因此精确已知）；
3. 用 splatting + z-buffer 渲染出新视角的 RGB-D 图。

这带来两个好处：
- **可以造出任意长度的多视角序列**，且 GT（3D 框）保持不变；
- 内外参**精确已知**（绕开了真实环绕拍摄缺少位姿的问题 —— 这正是
  三维家项目里"虚拟相机"方案的同一个思路）。

⚠️ 诚实的局限（面试要主动说）
新视角渲染会有**遮挡空洞**（原视角看不到的区域没有点云）。
所以它适合验证"多视角融合能否收敛到同一个物体"，**不能**替代真实多视角数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.data.synthetic import RenderResult
from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("data.virtual_camera")


# ==========================================================================
# 位姿生成
# ==========================================================================
def look_at_pose(
    eye: Sequence[float],
    target: Sequence[float],
    *,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> CameraPose:
    """构造"站在 eye、看向 target"的相机位姿（world → camera）。"""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(world_up, dtype=np.float64).reshape(3)

    z_cam = target - eye
    norm = np.linalg.norm(z_cam)
    if norm < 1e-9:
        return CameraPose.identity()
    z_cam = z_cam / norm

    x_cam = np.cross(up, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:      # 视线与 up 平行，换一个参考轴
        x_cam = np.cross(np.array([1.0, 0.0, 0.0]), z_cam)
    x_cam = x_cam / max(np.linalg.norm(x_cam), 1e-9)
    y_cam = np.cross(z_cam, x_cam)

    R_cam_to_world = np.stack([x_cam, y_cam, z_cam], axis=1)
    R = R_cam_to_world.T
    t = -R @ eye
    return CameraPose(R, t)


def orbit_poses(
    center: Sequence[float],
    radius: float = 2.0,
    num: int = 6,
    *,
    height: Optional[float] = None,
    start_deg: float = -60.0,
    end_deg: float = 60.0,
    look_at_center: bool = True,
) -> List[CameraPose]:
    """在一个水平圆弧上生成若干朝向中心的相机位姿。

    Parameters
    ----------
    center
        环绕中心（通常取场景 GT 框的中心）。
    radius
        环绕半径（米）。
    num
        视角数量。
    height
        相机高度（None 时用 center 的 z + 0.2）。
    start_deg, end_deg
        圆弧起止角度（0° 指向世界 +x）。
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    if height is None:
        height = float(c[2]) + 0.2

    angles = np.linspace(np.deg2rad(start_deg), np.deg2rad(end_deg), int(num))
    poses: List[CameraPose] = []
    for ang in angles:
        eye = np.array([c[0] + radius * np.cos(ang), c[1] + radius * np.sin(ang), height])
        target = c if look_at_center else np.array([c[0], c[1], height], dtype=np.float64)
        poses.append(look_at_pose(eye, target))
    return poses


def forward_motion_poses(
    center: Sequence[float],
    distance: float = 1.0,
    num: int = 5,
    *,
    height: Optional[float] = None,
    lateral: float = 0.0,
) -> List[CameraPose]:
    """生成一段"朝向场景前进"的相机序列（模拟机器人靠近目标）。"""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    if height is None:
        height = float(c[2]) + 0.2
    offsets = np.linspace(distance, max(distance - 1.5, 0.4), int(num))
    poses: List[CameraPose] = []
    for off, lat in zip(offsets, np.linspace(-lateral, lateral, int(num))):
        eye = np.array([c[0] + lat, c[1] - off, height])
        poses.append(look_at_pose(eye, c))
    return poses


# ==========================================================================
# 点云渲染器
# ==========================================================================
@dataclass
class PointCloudRenderer:
    """把带颜色的点云用 splatting + z-buffer 渲染成图像。

    Attributes
    ----------
    points : (N,3) float64
        世界系点。
    colors : (N,3) uint8, optional
    labels : (N,) str, optional
        每点的语义标签（用于生成 GT 掩码）。
    """

    points: np.ndarray
    colors: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    min_depth: float = 0.1
    max_depth: float = 12.0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)
            if self.colors.shape[0] != self.points.shape[0]:
                raise ValueError(
                    f"颜色数({self.colors.shape[0]})与点数({self.points.shape[0]})不一致"
                )
        if self.labels is not None:
            self.labels = list(self.labels)
            if len(self.labels) != self.points.shape[0]:
                raise ValueError(
                    f"标签数({len(self.labels)})与点数({self.points.shape[0]})不一致"
                )

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    # ---------------- 渲染 ----------------
    def render(
        self,
        pose: CameraPose,
        intrinsics: CameraIntrinsics,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        splat: int = 1,
        background: int = 0,
        hole_fill: bool = True,
    ) -> RenderResult:
        """从给定位姿渲染一帧。

        Parameters
        ----------
        splat
            每个点的泼溅半径（像素）。1 表示 3×3，能明显减少空洞。
        hole_fill
            是否做一次形态学闭运算填补小空洞（纯 numpy 实现）。
        """
        if width is None:
            width = intrinsics.width
        if height is None:
            height = intrinsics.height
        if width is None or height is None:
            raise ValueError("需要 width/height（或让 intrinsics 携带）")
        width, height = int(width), int(height)

        depth_img = np.zeros((height, width), dtype=np.float32)
        color_img = np.full((height, width, 3), int(background), dtype=np.uint8)
        if self.labels is not None:
            label_img = np.full((height, width), -1, dtype=np.int32)
        else:
            label_img = None

        if self.num_points == 0:
            return RenderResult(
                color=color_img, depth_m=depth_img,
                instance_id=(label_img if label_img is not None else np.full((height, width), -1, np.int32)),
                labels=[], boxes_3d=np.zeros((0, 7), np.float32),
                intrinsics=intrinsics, pose=pose,
            )

        cam = pose.world_to_cam(self.points)
        z = cam[:, 2]
        valid = np.isfinite(z) & (z > self.min_depth) & (z < self.max_depth)
        if not np.any(valid):
            return RenderResult(
                color=color_img, depth_m=depth_img,
                instance_id=np.full((height, width), -1, np.int32),
                labels=[], boxes_3d=np.zeros((0, 7), np.float32),
                intrinsics=intrinsics, pose=pose,
            )

        idx = np.flatnonzero(valid)
        z_v = z[idx]
        u = cam[idx, 0] * intrinsics.fx / z_v + intrinsics.cx
        v = cam[idx, 1] * intrinsics.fy / z_v + intrinsics.cy

        inside = (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
        idx = idx[inside]
        u, v, z_v = u[inside], v[inside], z_v[inside]

        # 由远及近写入 → 近处最后覆盖（天然 z-buffer）
        order = np.argsort(-z_v)
        idx, u, v, z_v = idx[order], u[order], v[order], z_v[order]

        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)

        r = int(max(splat, 0))
        for dv in range(-r, r + 1):
            vv = np.clip(vi + dv, 0, height - 1)
            for du in range(-r, r + 1):
                uu = np.clip(ui + du, 0, width - 1)
                depth_img[vv, uu] = z_v
                if self.colors is not None:
                    color_img[vv, uu] = self.colors[idx]
                if label_img is not None:
                    label_img[vv, uu] = idx

        if hole_fill and r >= 1:
            depth_img = _fill_holes(depth_img)

        # 从渲染出的事实中重建标签列表与 bbox（保证与图像一致）
        labels_out: List[str] = []
        boxes_out = np.zeros((0, 7), dtype=np.float32)
        if label_img is not None and np.any(label_img >= 0):
            present = np.unique(label_img[label_img >= 0])
            labels_out = [self.labels[int(i)] for i in present if int(i) < len(self.labels)]
            boxes_out = _boxes_from_label_image(label_img, self.points, present)

        instance_id = label_img if label_img is not None else np.full((height, width), -1, np.int32)
        return RenderResult(
            color=color_img.astype(np.uint8),
            depth_m=depth_img,
            instance_id=instance_id,
            labels=labels_out,
            boxes_3d=boxes_out,
            intrinsics=intrinsics,
            pose=pose,
        )


def _fill_holes(depth: np.ndarray, *, iterations: int = 1) -> np.ndarray:
    """用"邻域有效深度的中位数"填补零值空洞（纯 numpy 形态学修复）。

    比 cv2.inpaint 简单得多，但对 splatting 产生的小空洞足够有效。
    """
    out = depth.copy()
    for _ in range(int(iterations)):
        zero = out <= 0
        if not np.any(zero):
            break
        padded = np.pad(out, 1, mode="edge")
        neigh = np.stack([
            padded[:-2, 1:-1], padded[2:, 1:-1],
            padded[1:-1, :-2], padded[1:-1, 2:],
        ], axis=0)                                    # (4,H,W)
        valid_neigh = np.where(neigh > 0, neigh, np.nan)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(valid_neigh, axis=0)
        fill = zero & np.isfinite(med)
        out[fill] = med[fill].astype(out.dtype)
    return out


def _boxes_from_label_image(
    label_img: np.ndarray,
    points: np.ndarray,
    present: np.ndarray,
) -> np.ndarray:
    """从"像素 → 点索引"的映射还原每个实例的 3D 包围盒。"""
    boxes: List[np.ndarray] = []
    for inst in present:
        sel = label_img == inst
        if not np.any(sel):
            continue
        point_idx = np.unique(label_img[sel])
        point_idx = point_idx[point_idx >= 0]
        pts = points[point_idx]
        if pts.shape[0] == 0:
            continue
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = (lo + hi) / 2.0
        size = np.clip(hi - lo, 1e-3, None)
        boxes.append(np.concatenate([center, size, [0.0]]).astype(np.float32))
    if not boxes:
        return np.zeros((0, 7), dtype=np.float32)
    return np.stack(boxes, axis=0)


# ==========================================================================
# 从 SUN RGB-D 场景造多视角序列
# ==========================================================================
def scene_sequence_from_cloud(
    scene,
    *,
    num_frames: int = 6,
    radius: float = 1.6,
    span_deg: float = 50.0,
    target: Optional[Sequence[float]] = None,
    max_points: int = 120_000,
    splat: int = 2,
    frame_prefix: str = "virt",
    seed: int = 0,
) -> List[RGBDFrame]:
    """把一个 SUN RGB-D 场景变成多视角序列（虚拟相机重渲染）。

    帧的 `meta` 里仍然携带**原始的 GT 3D 框与标签**（因为场景没变，
    GT 与视角无关），所以下游可以用同一份 GT 做评测。

    Parameters
    ----------
    target
        环绕中心（默认取 GT 框中心；没有 GT 时取点云中位数）。
    """
    from roboground.geometry.projection import frame_to_pointcloud  # noqa: PLC0415

    frame0 = scene.to_frame()
    points, colors = frame_to_pointcloud(
        frame0,
        min_depth=0.2, max_depth=8.0, max_points=int(max_points), colors=True, seed=seed,
    )
    if points.shape[0] == 0:
        logger.warn(f"场景 {scene.sequence} 点云为空，无法重渲染")
        return [frame0]

    # 每个点带上它所属实例的标签（用 GT 框包含关系判断）
    labels_per_point: Optional[List[str]] = None
    if scene.boxes_3d.shape[0] > 0:
        labels_per_point = _assign_labels_to_points(points, scene.boxes_3d, scene.labels)

    if target is None:
        if scene.boxes_3d.shape[0] > 0:
            target = scene.boxes_3d[:, :3].mean(axis=0)
        else:
            target = np.median(points, axis=0)
    target = np.asarray(target, dtype=np.float64).reshape(3)

    poses = orbit_poses(target, radius=radius, num=int(num_frames),
                        start_deg=-span_deg / 2.0, end_deg=span_deg / 2.0)

    renderer = PointCloudRenderer(points, colors, labels_per_point)
    width, height = scene.shape[1], scene.shape[0]

    frames: List[RGBDFrame] = []
    for i, pose in enumerate(poses):
        res = renderer.render(pose, scene.intrinsics, width=width, height=height, splat=splat)
        frame = res.to_frame(frame_id=f"{frame_prefix}_{i:03d}")
        # 保留原始 GT（视角无关），但把渲染出的实例框也带上供参考
        frame.meta["boxes_3d"] = scene.boxes_3d
        frame.meta["labels"] = list(scene.labels)
        frame.meta["rendered_boxes_3d"] = res.boxes_3d
        frame.meta["source"] = "virtual_camera"
        frame.meta["sequence"] = scene.sequence
        frames.append(frame)

    logger.debug(
        f"虚拟相机生成 {len(frames)} 帧（源场景 {scene.sequence}，"
        f"点云 {points.shape[0]} 点，标签覆盖 {len(set(labels_per_point)) if labels_per_point else 0} 类）"
    )
    return frames


def _assign_labels_to_points(
    points: np.ndarray,
    boxes_3d: np.ndarray,
    labels: Sequence[str],
) -> List[str]:
    """按 GT 框把标签分配给点（点在框内 → 该框的标签）。"""
    out = ["object"] * points.shape[0]
    pts = np.asarray(points, dtype=np.float64)
    assigned = np.zeros(points.shape[0], dtype=bool)

    for k in range(boxes_3d.shape[0]):
        cx, cy, cz, dx, dy, dz = boxes_3d[k, :6]
        heading = float(boxes_3d[k, 6]) if boxes_3d.shape[1] > 6 else 0.0
        c, s = np.cos(-heading), np.sin(-heading)
        rel = pts - np.array([cx, cy, cz], dtype=np.float64)
        # 转到框的局部系（绕 z 反向旋转）
        lx = rel[:, 0] * c - rel[:, 1] * s
        ly = rel[:, 0] * s + rel[:, 1] * c
        lz = rel[:, 2]
        half = np.abs(np.array([dx, dy, dz], dtype=np.float64)) / 2.0
        inside = (np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1]) & (np.abs(lz) <= half[2])
        # 先到先得（后面的框不覆盖已分配的，避免大框吞掉小框内的点）
        take = inside & (~assigned)
        out_arr = np.asarray(out, dtype=object)
        out_arr[take] = str(labels[k]) if k < len(labels) else "object"
        out = out_arr.tolist()
        assigned |= take

    return out
