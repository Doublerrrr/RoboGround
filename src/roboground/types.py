"""跨模块共享的数据类型。

这里的 dataclass 是整个项目的"接口契约"：感知层产出 `Detection2D` / `Observation`，
映射层消费它们构建 `SemanticMap`，推理层消费地图产出 `SpatialRelation`。

约定（非常重要，全项目统一）
---------------------------
1. **相机坐标系**：OpenCV 约定 —— x 右、y 下、z 前（深度方向），单位米。
2. **世界坐标系**：z 轴向上（机器人/室内地图惯例），单位米。
3. **外参**：`CameraPose.R` / `CameraPose.t` 表示 **world → camera**：
   `p_cam = R @ p_world + t`。
4. **深度**：对外一律用**米**；`RGBDFrame.depth_m` 已是米，
   原始 uint16 毫米值请先用 `geometry.projection.raw_depth_to_meters` 转换。
5. **像素坐标**：`(u, v)`，u 是列（x），v 是行（y），浮点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


#: 固定的 90° 轴置换：OpenCV 相机系 `(x右, y下, z前)` → 世界系 `(x右, y前, z上)`
#: 即 `(x, y, z)_cam → (x, z, -y)_world`。
#: 它来自 SUN RGB-D 的标注工具约定，已被 `scripts/calibrate_sunrgbd_geometry.py` 标定确认。
_CAM_TO_WORLD_PERMUTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])


# ==========================================================================
# 相机 / 帧
# ==========================================================================
@dataclass
class CameraPose:
    """相机位姿（world → camera）。

    Attributes
    ----------
    R : (3, 3) float64
        旋转矩阵，正交且 det=+1。
    t : (3,) float64
        平移向量。
    """

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    # ---------------- 构造 ----------------
    @classmethod
    def identity(cls) -> "CameraPose":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_camera_to_world(
        cls,
        R_cam_to_world: np.ndarray,
        t_cam_in_world: Optional[np.ndarray] = None,
    ) -> "CameraPose":
        """从"相机→世界"的旋转构造位姿（本类内部存的是反向）。"""
        R_cw = np.asarray(R_cam_to_world, dtype=np.float64).reshape(3, 3)
        if t_cam_in_world is None:
            return cls(R_cw.T, np.zeros(3))
        # 世界→相机：cam = R_cwᵀ (p_world - C)
        return cls(R_cw.T, -R_cw.T @ np.asarray(t_cam_in_world, dtype=np.float64).reshape(3))

    @classmethod
    def from_sunrgbd(cls, Rtilt: np.ndarray) -> "CameraPose":
        """构造 SUN RGB-D 的相机位姿（**已由数据标定**，见下）。

        标定结论（`scripts/calibrate_sunrgbd_geometry.py` 用角点闭环一致性选出）::

            相机 → 世界:  R_cw = P @ Rtilt
            世界 → 相机:  R_wc = Rtiltᵀ @ Pᵀ

        其中 `P = [[1,0,0],[0,0,1],[0,-1,0]]` 是固定的 90° 轴置换，
        把 OpenCV 相机系 `(x右, y下, z前)` 变成标注工具用的
        `(x右, y前/深度, z上)` 世界系。

        为什么不是朴素的 `R_wc = Rtiltᵀ`？
        --------------------------------
        1. SUN RGB-D 的 GT 3D 框**不在 OpenCV 相机系里**（直接在相机系解释时，
           投影后 z 全为负，valid_ratio = 0），它在"y 为深度、z 为竖直"的
           重力对齐世界系里；
        2. `Rtilt` 本身接近单位阵（实测约 12° 的小倾斜），说明它是**传感器
           残余倾斜的精修**，作用在**相机原生轴**上 —— 因此顺序必须是
           "先 Rtilt，再做轴置换"（`P @ Rtilt`），而不是反过来；
        3. 数据支持这一点：`P @ Rtilt` 的闭环误差中位数 0.78 m，
           而 `Rtilt @ P`（顺序反了）是 2.21 m，差 2.8 倍。
        """
        Rtilt = np.asarray(Rtilt, dtype=np.float64).reshape(3, 3)
        P = _CAM_TO_WORLD_PERMUTATION
        R_cw = P @ Rtilt                     # 相机 → 世界
        return cls(R_cw.T, np.zeros(3))      # 世界 → 相机

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "CameraPose":
        """从 4×4 或 3×4 的 [R|t] 矩阵构造。"""
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape == (4, 4):
            m = m[:3, :]
        if m.shape != (3, 4):
            raise ValueError(f"期望 (3,4) 或 (4,4)，收到 {m.shape}")
        return cls(m[:, :3], m[:, 3])

    @classmethod
    def from_rt(cls, R: np.ndarray, t: np.ndarray) -> "CameraPose":
        return cls(R, t)

    # ---------------- 变换 ----------------
    def world_to_cam(self, points_world: np.ndarray) -> np.ndarray:
        """(N,3) 世界系 → 相机系。"""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        return pts @ self.R.T + self.t

    def cam_to_world(self, points_cam: np.ndarray) -> np.ndarray:
        """(N,3) 相机系 → 世界系。"""
        pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        return (pts - self.t) @ self.R

    def camera_center(self) -> np.ndarray:
        """相机光心在世界系中的位置：C = -Rᵀ t。"""
        return -self.R.T @ self.t

    def inverse(self) -> "CameraPose":
        """逆位姿（camera → world 的 [R|t]）。"""
        Rt = self.R.T
        return CameraPose(Rt, -Rt @ self.t)

    def as_matrix(self, homogeneous: bool = False) -> np.ndarray:
        if homogeneous:
            m = np.eye(4)
            m[:3, :3] = self.R
            m[:3, 3] = self.t
            return m
        return np.hstack([self.R, self.t.reshape(3, 1)])


@dataclass
class CameraIntrinsics:
    """针孔相机内参。

    支持两种构造方式：直接给 fx/fy/cx/cy，或给 3×3 的 K 矩阵。
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: Optional[int] = None
    height: Optional[int] = None

    def __post_init__(self) -> None:
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        self.cx = float(self.cx)
        self.cy = float(self.cy)

    @classmethod
    def from_matrix(cls, K: np.ndarray, width: Optional[int] = None,
                    height: Optional[int] = None) -> "CameraIntrinsics":
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K 应为 (3,3)，收到 {K.shape}")
        return cls(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                   width=width, height=height)

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    # 兼容常见别名，避免调用方记不住命名
    @property
    def K(self) -> np.ndarray:
        return self.to_matrix()

    def scaled(self, scale: float) -> "CameraIntrinsics":
        """按比例缩放内参（图像 resize 后必须调用，否则投影全错）。"""
        return CameraIntrinsics(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            width=None if self.width is None else int(round(self.width * scale)),
            height=None if self.height is None else int(round(self.height * scale)),
        )


@dataclass
class RGBDFrame:
    """一帧 RGB-D 观测。

    Attributes
    ----------
    color : (H, W, 3) uint8
        RGB 图像（注意是 RGB 不是 BGR）。
    depth_m : (H, W) float32
        深度图，**单位米**。无效深度置 0 或 NaN。
    intrinsics : CameraIntrinsics
    pose : CameraPose
        该帧相机在世界系中的位姿（world → camera）。
    timestamp : float
        秒。
    frame_id : str
    """

    color: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    pose: CameraPose = field(default_factory=CameraPose.identity)
    timestamp: float = 0.0
    frame_id: str = "frame"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.color = np.asarray(self.color)
        self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
        if self.color.ndim != 3 or self.color.shape[2] != 3:
            raise ValueError(f"color 应为 (H,W,3)，收到 {self.color.shape}")
        if self.depth_m.shape != self.color.shape[:2]:
            raise ValueError(
                f"depth 与 color 尺寸不一致：{self.depth_m.shape} vs {self.color.shape[:2]}"
            )

    @property
    def height(self) -> int:
        return int(self.color.shape[0])

    @property
    def width(self) -> int:
        return int(self.color.shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def valid_depth_mask(self, min_depth: float = 0.1,
                         max_depth: float = 8.0) -> np.ndarray:
        """返回 (H,W) bool：深度落在 [min, max] 且非 NaN 的像素。"""
        d = self.depth_m
        return np.isfinite(d) & (d > min_depth) & (d < max_depth)


# ==========================================================================
# 感知输出
# ==========================================================================
@dataclass
class Detection2D:
    """开放词汇 2D 检测结果。

    Attributes
    ----------
    label : str
        检测到的类别文本（开放词汇下是自然语言 prompt 或模型吐出的短语）。
    score : float
        置信度 ∈ [0,1]。
    bbox : (4,) float
        `(x1, y1, x2, y2)` 像素坐标。
    mask : (H, W) bool, optional
        实例掩码。没有分割时可用 bbox 填充（见 `segmenters.BoxSegmenter`）。
    feature : (D,) float32, optional
        区域语义特征（CLIP/DINOv2 或降级后的颜色直方图）。
    prompt : str, optional
        触发该检测的原始 prompt（便于回溯"哪句话检出了它"）。
    """

    label: str
    score: float
    bbox: np.ndarray
    mask: Optional[np.ndarray] = None
    feature: Optional[np.ndarray] = None
    prompt: Optional[str] = None

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float64).reshape(4)
        self.score = float(self.score)
        if self.mask is not None:
            self.mask = np.asarray(self.mask).astype(bool)
        if self.feature is not None:
            self.feature = np.asarray(self.feature, dtype=np.float32).reshape(-1)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "bbox": self.bbox.tolist(),
            "has_mask": self.mask is not None,
            "has_feature": self.feature is not None,
            "prompt": self.prompt,
        }


@dataclass
class Observation:
    """一次"检测 → 升维"的完整观测：2D 结果 + 它在 3D 世界中的落点。

    这是感知层与映射层之间的核心载体。
    """

    detection: Detection2D
    points_world: np.ndarray                 # (M,3) 该实例在世界系的 3D 点
    frame_id: str = "frame"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.points_world = np.asarray(self.points_world, dtype=np.float64).reshape(-1, 3)

    @property
    def label(self) -> str:
        return self.detection.label

    @property
    def feature(self) -> Optional[np.ndarray]:
        return self.detection.feature

    @property
    def num_points(self) -> int:
        return int(self.points_world.shape[0])

    @property
    def centroid(self) -> Optional[np.ndarray]:
        if self.num_points == 0:
            return None
        return self.points_world.mean(axis=0)


# ==========================================================================
# 空间关系 / 推理输出
# ==========================================================================
@dataclass
class SpatialRelation:
    """两个物体之间的空间关系（带米制证据）。

    `relation` 取值示例：above / below / left_of / right_of / in_front_of /
    behind / inside / near / far。
    """

    subject: str
    relation: str
    object: str
    distance: float                     # 米，两个物体中心距离
    evidence: Dict[str, float] = field(default_factory=dict)

    def to_text(self) -> str:
        return f"{self.subject} 在 {self.object} 的 {self.relation}（{self.distance:.2f} m）"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "distance": round(float(self.distance), 4),
            "evidence": {k: round(float(v), 4) for k, v in self.evidence.items()},
        }


@dataclass
class ReasoningResult:
    """推理层最终输出 —— 可直接被机器人规划模块消费的结构化结果。"""

    query: str
    answer: str
    targets: List["Any"] = field(default_factory=list)     # List[SemanticObject]
    relations: List[SpatialRelation] = field(default_factory=list)
    distances: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    backend: str = "rules"
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "targets": [
                t.to_dict() if hasattr(t, "to_dict") else str(t) for t in self.targets
            ],
            "relations": [r.to_dict() for r in self.relations],
            "distances": {k: round(float(v), 4) for k, v in self.distances.items()},
            "confidence": round(float(self.confidence), 4),
            "backend": self.backend,
        }


def as_points(array: Any) -> np.ndarray:
    """把任意输入规整成 (N,3) float64 点集（工具函数）。"""
    pts = np.asarray(array, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return pts.reshape(-1, 3)
