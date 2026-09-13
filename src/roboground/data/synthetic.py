"""合成室内场景：解析式射线-盒求交渲染 RGB-D（含完美 GT）。

用途
----
1. **单元测试**：完全确定性、无需数据、无网络，却能提供精确到像素的
   GT（深度、实例掩码、3D 框），是验证几何正确性的最佳工具；
2. **消融实验**：可以精确控制"物体数量 / 距离 / 遮挡关系 / 尺寸"，
   比真实数据集更适合做受控实验。

渲染原理（为什么不用图形库）
--------------------------
场景全由**轴对齐（或绕 z 旋转）的盒体**组成，所以可以用**解析式射线-盒求交**
（slab 方法）逐像素求最近交点 —— 纯 numpy、无依赖、可微性无关但速度足够，
而且天然给出"每个像素属于哪个实例"的完美 GT。

关键细节：射线的参数就是**相机 z 深度**
设相机系下方向 `d_cam = [(u-cx)/fx, (v-cy)/fy, 1]`（**不归一化**），
则 `p_cam = s · d_cam` 中的 `s` 恰好等于该点的相机 z 坐标（深度）。
于是"取最小的正 s" = "取深度最近"，与 z-buffer 语义完全一致，
反投影回 3D 时也不需要在距离和深度之间来回换算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.geometry.camera import pixel_grid
from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame


# ==========================================================================
# 几何工具
# ==========================================================================
def rotz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def ray_box_intersection(
    origins: np.ndarray,
    dirs: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> np.ndarray:
    """slab 法求射线与轴对齐盒的最近正向交点参数。

    Parameters
    ----------
    origins : (N,3)
    dirs : (N,3)
        不需要归一化 —— 返回的参数与 `dirs` 同尺度。
    box_min, box_max : (3,)

    Returns
    -------
    np.ndarray, shape (N,)
        交点参数 `s`（`point = origin + s * dir`）；不相交为 `inf`。
    """
    parallel_eps = 1e-12
    safe = np.where(np.abs(dirs) < parallel_eps, parallel_eps, dirs)
    inv = 1.0 / safe

    t1 = (box_min[None, :] - origins) * inv
    t2 = (box_max[None, :] - origins) * inv

    t_near = np.minimum(t1, t2).max(axis=1)
    t_far = np.maximum(t1, t2).min(axis=1)

    hit = (t_far >= np.maximum(t_near, 0.0)) & (t_far > 0.0)
    # 起点在盒内时 t_near < 0，此时应取 t_far 的入口（=0 时的深度）
    t = np.where(hit, np.maximum(t_near, 0.0), np.inf)
    return t


def ray_rotated_box_intersection(
    origins: np.ndarray,
    dirs: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    heading: float = 0.0,
) -> np.ndarray:
    """射线与绕 z 旋转的有向盒求交（把射线转到盒的局部系再做 slab）。"""
    R = rotz(-heading)
    o_local = (origins - center[None, :]) @ R.T
    d_local = dirs @ R.T
    half = np.clip(size, 1e-6, None) / 2.0
    return ray_box_intersection(o_local, d_local, -half, half)


# ==========================================================================
# 场景对象
# ==========================================================================
@dataclass
class SceneObject:
    """合成场景中的一个物体（有向盒 + 颜色 + 标签）。"""

    label: str
    center: np.ndarray
    size: np.ndarray
    heading: float = 0.0
    color: Tuple[int, int, int] = (180, 180, 180)
    is_static_surface: bool = False       # 地面/墙面（不计入实例 GT）

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.size = np.asarray(self.size, dtype=np.float64).reshape(3)

    def corners(self) -> np.ndarray:
        """8 个角点（世界系）。"""
        half = self.size / 2.0
        signs = np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
        ], dtype=np.float64)
        return (signs * half) @ rotz(self.heading).T + self.center

    def to_box7(self) -> np.ndarray:
        """转成 `[cx,cy,cz,dx,dy,dz,heading]`（与 SUN RGB-D 的 GT 格式一致）。"""
        return np.concatenate([self.center, self.size, [self.heading]]).astype(np.float32)


@dataclass
class RenderResult:
    """渲染结果。"""

    color: np.ndarray                 # (H,W,3) uint8
    depth_m: np.ndarray               # (H,W) float32
    instance_id: np.ndarray           # (H,W) int32，-1 = 背景/未命中
    labels: List[str]                 # 实例 id → 标签
    boxes_3d: np.ndarray              # (K,7) float32
    intrinsics: CameraIntrinsics
    pose: CameraPose

    def to_frame(self, *, frame_id: str = "synth") -> RGBDFrame:
        return RGBDFrame(
            color=self.color,
            depth_m=self.depth_m,
            intrinsics=self.intrinsics,
            pose=self.pose,
            frame_id=frame_id,
            meta={
                "boxes_3d": self.boxes_3d,
                "labels": list(self.labels),
                "instance_id": self.instance_id,
                "source": "synthetic",
            },
        )


# ==========================================================================
# 合成房间
# ==========================================================================
@dataclass
class SyntheticRoom:
    """一个合成室内场景（地面 + 四面墙 + 若干家具盒）。

    Examples
    --------
    >>> room = SyntheticRoom.random(seed=0)
    >>> K = CameraIntrinsics(fx=120, fy=120, cx=80, cy=60, width=160, height=120)
    >>> res = room.render(CameraPose.identity(), K)
    >>> res.depth_m.shape
    (120, 160)
    """

    width: float = 4.0          # 房间尺寸（x）
    depth: float = 5.0          # 房间尺寸（y）
    height: float = 2.6         # 房间高度（z）
    objects: List[SceneObject] = field(default_factory=list)
    floor_color: Tuple[int, int, int] = (120, 110, 100)
    wall_color: Tuple[int, int, int] = (210, 205, 195)
    ceiling_color: Tuple[int, int, int] = (235, 235, 235)

    # ---------------- 构造 ----------------
    @classmethod
    def random(
        cls,
        seed: int = 0,
        *,
        num_objects: int = 6,
        width: float = 4.0,
        depth: float = 5.0,
        height: float = 2.6,
        allow_overlap: bool = False,
    ) -> "SyntheticRoom":
        """随机生成一个房间（物体不超出房间、可选不重叠）。"""
        rng = np.random.default_rng(seed)
        catalog = [
            ("table", (0.9, 0.9, 0.75), (150, 110, 80)),
            ("chair", (0.45, 0.45, 0.9), (90, 90, 140)),
            ("cup", (0.10, 0.10, 0.12), (230, 230, 240)),
            ("box", (0.35, 0.30, 0.30), (170, 140, 90)),
            ("bottle", (0.09, 0.09, 0.26), (90, 190, 120)),
            ("sofa", (1.8, 0.85, 0.8), (110, 130, 150)),
            ("shelf", (0.9, 0.35, 1.7), (160, 130, 100)),
            ("monitor", (0.55, 0.08, 0.35), (60, 60, 70)),
            ("trash can", (0.28, 0.28, 0.5), (130, 130, 130)),
            ("lamp", (0.25, 0.25, 0.5), (240, 220, 120)),
        ]
        picked = [catalog[int(i) % len(catalog)] for i in rng.permutation(len(catalog))[:num_objects]]

        objects: List[SceneObject] = []
        placed: List[Tuple[np.ndarray, np.ndarray]] = []
        for attempt in range(num_objects * 40):
            if len(objects) >= num_objects:
                break
            label, size, color = picked[len(objects)]
            size_arr = np.asarray(size, dtype=np.float64)
            margin = 0.25
            cx = rng.uniform(-width / 2 + margin + size_arr[0] / 2, width / 2 - margin - size_arr[0] / 2)
            cy = rng.uniform(0.6 + margin, depth - margin - size_arr[1] / 2)
            # 小物体放在桌面高度，大物体落地
            if size_arr[2] < 0.4 and label in {"cup", "bottle", "monitor", "lamp"}:
                cz = 0.75 + size_arr[2] / 2
            else:
                cz = size_arr[2] / 2
            center = np.array([cx, cy, cz])

            if not allow_overlap:
                bad = False
                for (oc, os_) in placed:
                    if np.all(np.abs(center - oc) < (size_arr + os_) / 2 + 0.05):
                        bad = True
                        break
                if bad:
                    continue

            heading = float(rng.uniform(0, np.pi)) if rng.random() < 0.5 else 0.0
            objects.append(SceneObject(label=label, center=center, size=size_arr,
                                       heading=heading, color=color))
            placed.append((center, size_arr))

        return cls(width=width, depth=depth, height=height, objects=objects)

    # ---------------- 表面（地面/墙/天花板）----------------
    def static_surfaces(self) -> List[SceneObject]:
        """返回静态表面盒（渲染用，不作为实例 GT）。"""
        w, d, h = self.width, self.depth, self.height
        t = 0.02
        return [
            SceneObject("floor", [0.0, d / 2.0, -t / 2], [w, d, t],
                        color=self.floor_color, is_static_surface=True),
            SceneObject("ceiling", [0.0, d / 2.0, h + t / 2], [w, d, t],
                        color=self.ceiling_color, is_static_surface=True),
            SceneObject("wall_left", [-w / 2 - t / 2, d / 2.0, h / 2], [t, d, h],
                        color=self.wall_color, is_static_surface=True),
            SceneObject("wall_right", [w / 2 + t / 2, d / 2.0, h / 2], [t, d, h],
                        color=self.wall_color, is_static_surface=True),
            SceneObject("wall_back", [0.0, -t / 2, h / 2], [w, t, h],
                        color=self.wall_color, is_static_surface=True),
            SceneObject("wall_front", [0.0, d + t / 2, h / 2], [w, t, h],
                        color=self.wall_color, is_static_surface=True),
        ]

    def all_boxes(self) -> List[SceneObject]:
        return self.static_surfaces() + list(self.objects)

    def gt_boxes_3d(self) -> np.ndarray:
        """实例 GT 的 3D 框（不含静态表面）。"""
        if not self.objects:
            return np.zeros((0, 7), dtype=np.float32)
        return np.stack([o.to_box7() for o in self.objects], axis=0)

    def gt_labels(self) -> List[str]:
        return [o.label for o in self.objects]

    # ---------------- 渲染 ----------------
    def render(
        self,
        pose: CameraPose,
        intrinsics: CameraIntrinsics,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_depth: float = 12.0,
        with_static: bool = True,
    ) -> RenderResult:
        """从给定相机位姿渲染一帧 RGB-D。"""
        if width is None:
            width = intrinsics.width
        if height is None:
            height = intrinsics.height
        if width is None or height is None:
            raise ValueError("需要 width/height（或让 intrinsics 携带）")

        width, height = int(width), int(height)
        uv = pixel_grid(width, height, flatten=True)                # (N,2)

        # 相机系方向（不归一化！这样射线参数 = 相机 z 深度）
        d_cam = np.stack([
            (uv[:, 0] - intrinsics.cx) / intrinsics.fx,
            (uv[:, 1] - intrinsics.cy) / intrinsics.fy,
            np.ones(uv.shape[0], dtype=np.float64),
        ], axis=1)

        # 转到世界系：方向 d_world = Rᵀ d_cam，起点 = 相机光心
        d_world = d_cam @ pose.R            # 等价于 (Rᵀ dᵀ)ᵀ
        origin_world = pose.camera_center()  # (3,)
        origins = np.broadcast_to(origin_world, d_world.shape)

        candidates = self.all_boxes() if with_static else list(self.objects)
        n_pix = d_world.shape[0]

        best_t = np.full(n_pix, np.inf, dtype=np.float64)
        best_id = np.full(n_pix, -1, dtype=np.int32)
        best_color = np.zeros((n_pix, 3), dtype=np.float64)

        instance_boxes: List[SceneObject] = []
        instance_index: Dict[int, int] = {}

        for obj in candidates:
            t = ray_rotated_box_intersection(origins, d_world, obj.center, obj.size, obj.heading)
            closer = t < best_t
            if not np.any(closer):
                continue
            best_t = np.where(closer, t, best_t)

            if obj.is_static_surface:
                best_id = np.where(closer, -1, best_id)
            else:
                if id(obj) not in instance_index:
                    instance_index[id(obj)] = len(instance_boxes)
                    instance_boxes.append(obj)
                inst = instance_index[id(obj)]
                best_id = np.where(closer, inst, best_id)

            # 简单朗伯着色：法线朝上/朝前更亮，制造一点光照变化
            base = np.asarray(obj.color, dtype=np.float64)
            shading = 0.85 + 0.15 * float(np.clip(np.cos(obj.heading), -1, 1))
            best_color = np.where(closer[:, None], base[None, :] * shading, best_color)

        # 深度：超过 max_depth 视为无效（0）
        depth = best_t.reshape(height, width).astype(np.float32)
        depth[(~np.isfinite(depth)) | (depth <= 0) | (depth > max_depth)] = 0.0

        color = np.clip(best_color.reshape(height, width, 3), 0, 255).astype(np.uint8)
        color[depth <= 0] = 0

        instance_id = best_id.reshape(height, width).astype(np.int32)
        boxes_3d = (np.stack([o.to_box7() for o in instance_boxes], axis=0)
                    if instance_boxes else np.zeros((0, 7), dtype=np.float32))
        labels = [o.label for o in instance_boxes]

        return RenderResult(
            color=color,
            depth_m=depth,
            instance_id=instance_id,
            labels=labels,
            boxes_3d=boxes_3d,
            intrinsics=intrinsics,
            pose=pose,
        )


# ==========================================================================
# 便捷函数
# ==========================================================================
def default_intrinsics(width: int = 160, height: int = 120, fov_deg: float = 60.0) -> CameraIntrinsics:
    """按视场角生成默认内参。"""
    fx = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    fy = fx
    return CameraIntrinsics(fx=fx, fy=fy, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
                            width=width, height=height)


def make_synthetic_frame(
    seed: int = 0,
    *,
    width: int = 160,
    height: int = 120,
    num_objects: int = 6,
    camera_height: float = 1.2,
    frame_id: str = "synth",
) -> RGBDFrame:
    """一步生成一帧合成 RGB-D（供快速测试）。"""
    room = SyntheticRoom.random(seed=seed, num_objects=num_objects)
    K = default_intrinsics(width, height)
    # 相机放在房间前侧、朝 +y 看（世界系 y 为前方）
    pose = _look_forward_pose(np.array([0.0, 0.3, camera_height]))
    res = room.render(pose, K)
    return res.to_frame(frame_id=frame_id)


def make_synthetic_sequence(
    seed: int = 0,
    *,
    num_frames: int = 4,
    width: int = 160,
    height: int = 120,
    num_objects: int = 6,
    camera_height: float = 1.2,
    stride: float = 0.35,
) -> List[RGBDFrame]:
    """生成一段"机器人沿 x 平移"的多视角序列（同一房间、同一批物体）。"""
    room = SyntheticRoom.random(seed=seed, num_objects=num_objects)
    K = default_intrinsics(width, height)
    frames: List[RGBDFrame] = []
    for i in range(int(num_frames)):
        offset = (i - (num_frames - 1) / 2.0) * stride
        pose = _look_forward_pose(np.array([offset, 0.3, camera_height]))
        res = room.render(pose, K)
        frames.append(res.to_frame(frame_id=f"synth_{i:03d}"))
    return frames


def _look_forward_pose(center_world: np.ndarray, *, pitch_deg: float = 10.0) -> CameraPose:
    """构造一个"站在某处、朝世界 +y 略向下看"的相机位姿（world → camera）。

    相机系 = OpenCV（x 右, y 下, z 前）；世界系 = (x 右, y 前, z 上)。
    朝 +y 看 → 相机 z 轴对齐世界 +y；向下俯仰 pitch 度。
    """
    pitch = np.deg2rad(pitch_deg)
    # 基向量（世界系表达）：z_cam 朝前下方、x_cam 朝右、y_cam 朝下
    z_cam = np.array([0.0, np.cos(pitch), -np.sin(pitch)])
    x_cam = np.array([1.0, 0.0, 0.0])
    y_cam = np.cross(z_cam, x_cam)          # 保证右手系
    R_cam_to_world = np.stack([x_cam, y_cam, z_cam], axis=1)   # 列为相机基
    R = R_cam_to_world.T                     # world → camera
    t = -R @ np.asarray(center_world, dtype=np.float64).reshape(3)
    return CameraPose(R, t)
