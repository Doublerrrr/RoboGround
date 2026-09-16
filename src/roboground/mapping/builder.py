"""地图构建器：RGB-D 帧序列 → 开放词汇 3D 语义地图（Stage 2 主体）。

完整链路
--------
```
RGB-D 帧
  │
  ├─ 1. 感知（PerceptionPipeline）       → 2D 检测 + 掩码 + 特征
  │
  ├─ 2. 反投影（backproject_detection）  → 每个实例的 3D 世界系点集
  │
  ├─ 3. 实例关联（ObjectTracker）        → 跨帧把"同一个物体"串成一条轨迹
  │
  ├─ 4. 体素融合（VoxelGrid）            → 多视角特征累加，得到 3D 特征场
  │
  └─ 5. 产出 SemanticMap                 → 物体层（规划用）+ 体素层（开放词汇用）
```

两个关键设计决策
----------------
1. **物体用"实例关联"而不是"空间聚类"**
   DBSCAN 之类的空间聚类有个硬伤：两个挨得近的物体会被粘成一个
   （实测：桌子边上的杯子只要中心距离 < eps 就会合并）。而我们在感知阶段
   本来就知道"每个检测是一个实例"，所以正确做法是**按身份关联**：
   标签语义兼容 + 中心距离/IoU 达标 → 同一物体；否则新建。
   这与 ConceptGraphs 等工作的做法一致，也是真实机器人系统该有的形态。

   消融（`scripts/42_ablate_association.py`，12 个真实采集点）证明这块**是承重的**：
   换掉关联改走聚类，命中率从 95.1% 掉到 66.1%、幽灵物体从 1.8% 涨到 28.6%。

2. **体素层与物体层并存**
   体素层承载"开放词汇"（特征向量 + 任意文本可比），
   物体层承载"机器人可用性"（有名字、有中心、有 bbox）。

   注意：体素化是**全局**做的，与物体关联解耦；关联完成后，再把体素
   按最近邻挂到各物体上（填 `SemanticObject.voxel_ids`）。

关联的两条实现（`mapping.assoc_strategy`）
----------------------------------------
· `"greedy"`：**逐个观测**贪心 —— 每个观测独立挑当前得分最高的轨迹。
  因为轨迹容量不限，这个"分配问题"是可分的，所以逐观测贪心在
  **该模型下就是最优的**。
· `"hungarian"`（默认）：**逐帧**一对一 —— 一帧内的观测与已有轨迹做
  全局最优分配（`scipy.optimize.linear_sum_assignment`），
  即"同一条轨迹在一帧里最多认领一个观测"。

两者不是同一个模型：`hungarian` 额外要求"一帧一物体一次观测"，
这正好堵住了 `greedy` 最大的失效模式 —— 把两个真的不同的物体
并进同一条轨迹（实测碎裂率 0.79 → 1.04、中心误差中位 0.502 → 0.425 m）。
它的前提是"一个物体在一帧里最多产生一个检测"，全景臂里跨 ±180°
被拆成两段的物体是例外（那一段会另起一条轨迹），
所以 `assoc_strategy` 保留了 `"greedy"` 可切回去。
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roboground.geometry.projection import backproject_detection
from roboground.geometry.voxel import VoxelGrid, cluster_points_dbscan
from roboground.mapping.semantic_map import SemanticMap, SemanticObject
from roboground.perception.base import PerceptionPipeline
from roboground.types import Detection2D, Observation, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("mapping.builder")


# ==========================================================================
# 内部：物体轨迹（跨帧聚合的临时状态）
# ==========================================================================
class _ObjectTrack:
    """一条物体轨迹：把多帧里属于同一物体的观测累加起来。

    为什么单独开一个类而不直接建 `SemanticObject`？
    因为 `SemanticObject` 的字段（中心、bbox、特征）都是**最终值**，
    而轨迹需要的是**可累加的中间量**（求和、计数、投票）。
    """

    __slots__ = (
        "track_id", "label_votes", "feature_sum", "weight_sum",
        "points_sum", "points_count", "bbox_min", "bbox_max",
        "sample_points", "sample_count", "num_observations",
        "frame_ids", "scores", "feature_dim", "_max_points",
    )

    def __init__(self, track_id: int, feature_dim: int, max_points: int = 10_000) -> None:
        self.track_id = int(track_id)
        self.feature_dim = int(feature_dim)
        self.label_votes: Counter = Counter()
        self.feature_sum = np.zeros(self.feature_dim, dtype=np.float64)
        self.weight_sum = 0.0
        self.points_sum = np.zeros(3, dtype=np.float64)
        self.points_count = 0
        self.bbox_min = np.full(3, np.inf, dtype=np.float64)
        self.bbox_max = np.full(3, -np.inf, dtype=np.float64)
        self.sample_points = np.zeros((0, 3), dtype=np.float64)
        self.sample_count = 0
        self.num_observations = 0
        self.frame_ids: set = set()
        self.scores: List[float] = []
        self._max_points = int(max_points)

    # ---------------- 写入 ----------------
    def add(self, obs: Observation) -> None:
        pts = obs.points_world
        w = float(obs.detection.score)

        self.num_observations += 1
        self.frame_ids.add(obs.frame_id)
        self.scores.append(w)

        if pts.shape[0] > 0:
            self.points_sum += pts.sum(axis=0)
            self.points_count += int(pts.shape[0])
            self.bbox_min = np.minimum(self.bbox_min, pts.min(axis=0))
            self.bbox_max = np.maximum(self.bbox_max, pts.max(axis=0))

            # 有界采样：保留最多 max_points 个点用于分位数 bbox
            room = self._max_points - self.sample_count
            if room > 0:
                take = pts[:room]
                self.sample_points = np.concatenate([self.sample_points, take], axis=0)
                self.sample_count += take.shape[0]

        feat = obs.feature
        if feat is not None:
            f = np.asarray(feat, dtype=np.float64).reshape(-1)
            if f.shape[0] == self.feature_dim:
                self.feature_sum += f * max(w, 1e-3)
                self.weight_sum += max(w, 1e-3)

        label = str(obs.label or "").strip()
        if label:
            # 票数按点数加权：大面积观测更可信
            self.label_votes[label] += max(int(pts.shape[0]), 1)

    # ---------------- 读出 ----------------
    @property
    def center(self) -> np.ndarray:
        if self.points_count == 0:
            return np.zeros(3, dtype=np.float64)
        return self.points_sum / float(self.points_count)

    @property
    def label(self) -> str:
        if not self.label_votes:
            return "object"
        return self.label_votes.most_common(1)[0][0]

    @property
    def feature(self) -> np.ndarray:
        if self.weight_sum <= 1e-9:
            return np.zeros(self.feature_dim, dtype=np.float32)
        f = (self.feature_sum / self.weight_sum).astype(np.float32)
        norm = float(np.linalg.norm(f))
        return f / norm if norm > 1e-8 else f

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    def bbox(self, percentile: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (min, max)。用采样点做分位数裁剪，抗离群点。"""
        pts = self.sample_points
        if percentile > 0 and pts.shape[0] >= 8:
            lo = np.percentile(pts, percentile, axis=0)
            hi = np.percentile(pts, 100.0 - percentile, axis=0)
            # 分位数结果不该超出真实范围
            lo = np.maximum(lo, self.bbox_min if np.all(np.isfinite(self.bbox_min)) else lo)
            hi = np.minimum(hi, self.bbox_max if np.all(np.isfinite(self.bbox_max)) else hi)
            return lo, hi
        if np.all(np.isfinite(self.bbox_min)):
            return self.bbox_min.copy(), self.bbox_max.copy()
        return pts.min(axis=0), pts.max(axis=0)

    def to_object(self, obj_id: int, *, percentile: float = 2.0,
                  n_voxels: int = 0, voxel_ids: Optional[np.ndarray] = None) -> SemanticObject:
        lo, hi = self.bbox(percentile)
        votes = dict(sorted(self.label_votes.items(), key=lambda kv: -kv[1])[:5])
        total = sum(self.label_votes.values()) or 1
        purity = self.label_votes.get(self.label, 0) / total if self.label_votes else 0.0

        support = min(1.0, max(n_voxels, self.points_count // 50) / 10.0)
        obs_score = min(1.0, self.num_observations / 3.0)
        confidence = float(
            (max(support, 1e-3) * max(obs_score, 1e-3) * max(purity, 1e-3)) ** (1.0 / 3.0)
        )

        return SemanticObject(
            obj_id=int(obj_id),
            label=self.label,
            center=self.center,
            bbox_min=lo,
            bbox_max=hi,
            feature=self.feature,
            num_voxels=int(n_voxels),
            num_points=int(self.points_count),
            confidence=confidence,
            label_scores=votes,
            voxel_ids=(np.zeros(0, dtype=np.int64) if voxel_ids is None else voxel_ids),
            frame_ids=sorted(self.frame_ids),
        )

    def __repr__(self) -> str:
        return (
            f"_ObjectTrack(id={self.track_id}, label={self.label!r}, "
            f"obs={self.num_observations}, pts={self.points_count})"
        )


# ==========================================================================
# 地图构建器
# ==========================================================================
class MapBuilder:
    """把 RGB-D 帧序列建成开放词汇 3D 语义地图。

    Examples
    --------
    >>> from roboground import load_config
    >>> builder = MapBuilder(load_config())        # doctest: +SKIP
    >>> smap = builder.build_from_frames(frames)   # doctest: +SKIP
    """

    def __init__(
        self,
        cfg,
        *,
        pipeline: Optional[PerceptionPipeline] = None,
        prompts: Optional[Sequence[str]] = None,
        text_encoder: Any = None,
    ) -> None:
        self.cfg = cfg
        self.prompts = (
            list(prompts) if prompts is not None
            else list(cfg.get("perception.prompts", []) or [])
        )

        if pipeline is None:
            from roboground.perception import build_pipeline  # 延迟导入

            pipeline = build_pipeline(cfg, prompts=self.prompts)
        self.pipeline = pipeline
        self.text_encoder = (
            text_encoder if text_encoder is not None else getattr(pipeline, "encoder", None)
        )

        # 几何参数
        self.depth_scale = float(cfg.get("geometry.depth_scale", 1000.0))
        self.min_depth = float(cfg.get("geometry.min_depth", 0.1))
        self.max_depth = float(cfg.get("geometry.max_depth", 8.0))
        self.ghost_percentile = cfg.get("geometry.depth_trunc_percentile", None)
        self.voxel_size = float(cfg.get("geometry.voxel_size", 0.05))

        # 关联参数
        self.assoc_radius = float(cfg.get("mapping.assoc_radius", 0.6))
        # `assoc_iou > 1` = 关闭 IoU 兜底（IoU 恒 ≤ 1）
        _iou = cfg.get("mapping.assoc_iou", 0.1)
        self.assoc_iou = None if _iou is None else float(_iou)
        self.assoc_require_label = bool(cfg.get("mapping.assoc_require_label", True))
        # 关联的**分配方式**：`hungarian`（逐帧全局最优，默认）| `greedy`（逐观测贪心）
        self.assoc_strategy = str(cfg.get("mapping.assoc_strategy", "hungarian")).lower()
        # 同帧内"其实是同一个物体"的重复检测合并门限（3D IoU）；>1 = 关闭
        _mg = cfg.get("mapping.assoc_merge_iou", 0.5)
        self.assoc_merge_iou = None if _mg is None else float(_mg)
        self.object_mode = str(cfg.get("mapping.object_mode", "association")).lower()
        self.object_max_points = int(cfg.get("mapping.object_max_points", 10_000))
        self.min_observations = int(cfg.get("mapping.min_observations", 1))

        # 状态
        self.grid: Optional[VoxelGrid] = None
        self.observations: List[Observation] = []
        self.tracks: List[_ObjectTrack] = []
        self.frames_processed: int = 0
        self.timings: Dict[str, float] = {}
        self._matcher = None       # 惰性构建的 LexicalMatcher（用于标签兼容判断）
        #: 各帧相机光心（世界系）。最后一帧的位置作为"机器人在哪"，
        #: 供规则引擎回答"离我最近的 X"这类问题。
        self.camera_centers: List[np.ndarray] = []
        #: 丢弃原因计数（检测总数 / 无有效深度 / 低置信度）
        self.drop_stats: Dict[str, int] = {
            "detections": 0, "no_valid_depth": 0, "low_confidence": 0,
        }

    # ---------------- 属性 ----------------
    @property
    def feature_dim(self) -> int:
        enc = getattr(self.pipeline, "encoder", None)
        if enc is not None:
            dim = int(getattr(enc, "feature_dim", 0))
            if dim > 0:
                return dim
        return int(self.cfg.get("perception.encoder_kwargs.feature_dim", 256))

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    # ---------------- 生命周期 ----------------
    def reset(self) -> None:
        """清空状态（可用同一 builder 建多张地图）。"""
        self.grid = None
        self.observations = []
        self.tracks = []
        self.frames_processed = 0
        self.timings = {}
        self.camera_centers = []
        self.drop_stats = {"detections": 0, "no_valid_depth": 0, "low_confidence": 0}

    # ---------------- 每帧处理 ----------------
    def add_frame(
        self,
        frame: RGBDFrame,
        *,
        prompts: Optional[Sequence[str]] = None,
    ) -> List[Observation]:
        """处理一帧：感知 → 反投影 → 关联 → 并入体素网格。"""
        t0 = time.perf_counter()

        # 1) 2D 开放词汇感知
        detections: List[Detection2D] = self.pipeline.run(frame, prompts=prompts)
        t1 = time.perf_counter()

        # 2) 反投影到 3D 世界系
        observations: List[Observation] = []
        for det in detections:
            obs = backproject_detection(
                det, frame,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                ghost_percentile=self.ghost_percentile,
            )
            if obs.num_points > 0:
                observations.append(obs)
        t2 = time.perf_counter()

        # 3) 实例关联 + 4) 体素融合
        #
        # ★ 关联按**帧**批量做（`_associate_frame`）：`hungarian` 策略需要
        #   "这一帧的所有观测一起和已有轨迹做全局最优匹配"。
        #   先筛掉低置信度，再批量关联，最后按原顺序并入体素网格 ——
        #   对 `greedy` 而言这与原来的"逐个关联"**完全等价**（顺序都没变）。
        self._ensure_grid()
        conf_thr = float(self.cfg.get("mapping.conf_threshold", 0.0))
        kept: List[Observation] = []
        for obs in observations:
            if obs.detection.score < conf_thr:
                self.drop_stats["low_confidence"] += 1
                continue
            kept.append(obs)
        self._associate_frame(kept, image_width=int(frame.color.shape[1]))
        for obs in kept:
            self.grid.add_observation(obs)
        t3 = time.perf_counter()

        # ---- 丢弃原因统计（**不要静默丢数据**）----
        # 实测在真实 SUN RGB-D 上，常有一半检测因为"掩码区域内没有有效深度"
        # 而无法反投影（门/窗/玻璃/太远/太近）。如果静默丢弃，
        # 使用者会以为"检测器没检出"，而实际是几何阶段筛掉了 —— 归因完全错。
        n_before = len(detections)
        n_after = len(observations)
        self.drop_stats["detections"] += n_before
        self.drop_stats["no_valid_depth"] += max(0, n_before - n_after)

        self.observations.extend(observations)
        self.frames_processed += 1
        self.camera_centers.append(np.asarray(frame.pose.camera_center(), dtype=np.float64))
        self.timings = {
            "perceive_s": t1 - t0,
            "backproject_s": t2 - t1,
            "associate_fuse_s": t3 - t2,
            "frame_s": t3 - t0,
        }
        return observations

    def build_from_frames(
        self,
        frames: Iterable[RGBDFrame],
        *,
        prompts: Optional[Sequence[str]] = None,
        reset: bool = True,
    ) -> SemanticMap:
        """从帧序列建图（最常用入口）。"""
        if reset:
            self.reset()

        frames = list(frames)
        t_start = time.perf_counter()
        for i, frame in enumerate(frames):
            self.add_frame(frame, prompts=prompts)
            if (i + 1) % 10 == 0 or (i + 1) == len(frames):
                logger.debug(
                    f"  {i + 1}/{len(frames)} 帧 | 观测 {len(self.observations)} | "
                    f"轨迹 {len(self.tracks)} | 体素 {self.grid.num_voxels if self.grid else 0}"
                )
        total = time.perf_counter() - t_start
        logger.info(
            f"建图完成：{len(frames)} 帧 → {len(self.observations)} 个观测、"
            f"{len(self.tracks)} 条物体轨迹，耗时 {total:.2f}s"
        )
        return self.finalize(extra_meta={"build_s": total, "num_frames": len(frames)})

    def build_from_observations(
        self,
        observations: Sequence[Observation],
        *,
        feature_dim: Optional[int] = None,
        reset: bool = True,
    ) -> SemanticMap:
        """直接从预计算观测建图（跳过感知；测试与复用场景）。"""
        if reset:
            self.reset()
        dim = int(feature_dim or self.feature_dim)
        self.grid = VoxelGrid(
            voxel_size=self.voxel_size,
            feature_dim=dim,
            mode=str(self.cfg.get("mapping.fusion", "mean")),
            max_voxels=int(self.cfg.get("mapping.max_voxels", 2_000_000)),
        )
        for obs in observations:
            if obs.num_points == 0:
                continue
            self._associate(obs)
            self.grid.add_observation(obs)
        self.observations = list(observations)
        return self.finalize()

    # ---------------- 内部：网格与关联 ----------------
    def _ensure_grid(self) -> None:
        if self.grid is not None:
            return
        self.grid = VoxelGrid(
            voxel_size=self.voxel_size,
            feature_dim=self.feature_dim,
            mode=str(self.cfg.get("mapping.fusion", "mean")),
            max_voxels=int(self.cfg.get("mapping.max_voxels", 2_000_000)),
        )

    def _get_matcher(self):
        if self._matcher is None:
            from roboground.mapping.query import LexicalMatcher  # 延迟导入

            self._matcher = LexicalMatcher()
        return self._matcher

    def _labels_compatible(self, a: str, b: str) -> bool:
        """两个标签是否语义兼容（用于判断"是不是同一个物体"）。"""
        a, b = str(a or "").strip(), str(b or "").strip()
        if not a or not b:
            return True                      # 缺标签时不做限制
        if a == b:
            return True
        try:
            score = float(self._get_matcher().score(a, labels=[b])[0])
        except Exception:
            return a == b
        return score > 0.5

    def _associate(self, obs: Observation) -> _ObjectTrack:
        """把**单个**观测关联到已有轨迹，或新建一条。返回命中的轨迹。

        `greedy` 策略的实现。注意轨迹容量不限：同一帧里的多个观测
        可以落进同一条轨迹 —— 这正是它在上面说的那种失效模式。
        """
        if self.object_mode == "clustering":
            # 聚类模式不做实例关联，直接用一条全局轨迹占位
            if not self.tracks:
                self.tracks.append(_ObjectTrack(0, self.feature_dim, self.object_max_points))
            self.tracks[0].add(obs)
            return self.tracks[0]

        best: Optional[_ObjectTrack] = None
        best_score = -1.0
        for track in self.tracks:
            score = self._match_score(obs, track)
            # ★ 严格大于：分数并列时保留**先建的那条**轨迹（行为固定，可复现）
            if score > best_score:
                best, best_score = track, score

        if best is None:
            best = _ObjectTrack(len(self.tracks), self.feature_dim, self.object_max_points)
            self.tracks.append(best)

        best.add(obs)
        return best

    def _match_score(self, obs: Observation, track: _ObjectTrack) -> float:
        """观测与轨迹的关联得分；`< 0` 表示**不允许**关联。

        ⚠️ 两条关联实现（`_associate` 与 `_associate_frame`）**必须共用**
        这一个打分函数。否则消融实验（`scripts/42`）比的就成了两套规则，
        得出来的差值归因不到"分配方式"上。
        """
        if self.assoc_require_label and not self._labels_compatible(obs.label, track.label):
            return -1.0

        center = obs.centroid
        if center is None:
            center = np.zeros(3)
        dist = float(np.linalg.norm(center - track.center))

        if dist <= self.assoc_radius:
            # 距离越近分越高（归一到 [0,1]）
            return 1.0 - dist / max(self.assoc_radius, 1e-6)

        # 距离不达标时再看 3D IoU（应对"中心漂移但体积重叠"的情况）。
        # `assoc_iou > 1` 视为**关闭**这条兜底（IoU 恒 ≤ 1）。
        # 消融实测：这条兜底在 `greedy` 下是有害的（它会把中心相距 0.6 m 以上
        # 的两个物体并起来），在 `hungarian` 下无害。
        if self.assoc_iou is not None and track.points_count > 0:
            iou = float(self._track_observation_iou(track, obs))
            if iou >= self.assoc_iou:
                return iou
        return -1.0

    def _associate_frame(self, observations: Sequence[Observation],
                         *, image_width: Optional[int] = None) -> List[_ObjectTrack]:
        """把**一帧**的观测一次性关联掉（按 `assoc_strategy` 分派）。

        为什么要按帧看：`hungarian` 要的是"这一帧的所有观测 ↔ 已有轨迹的
        **全局**最优一对一匹配"。逐个观测调用 `_associate` 拿不到全局信息 ——
        先处理的观测可能把某条轨迹占掉，而它本可以落到另一条上。

        `image_width` 用于识别"跨图像左右边界的接缝"（等距柱状全景的
        ±180°），调用方知道帧尺寸就传进来。
        """
        if self.object_mode == "clustering":
            return [self._associate(obs) for obs in observations]

        # 先把"同一帧里其实是同一个物体"的重复检测合掉（见方法注释）
        observations = self._merge_same_frame(observations,
                                              image_width=image_width)

        if self.assoc_strategy != "hungarian":
            return [self._associate(obs) for obs in observations]

        n_o, n_t = len(observations), len(self.tracks)
        if n_o == 0:
            return []

        # 代价矩阵：左边 `n_t` 列是真实轨迹，右边 `n_o` 列是"哑列"。
        # 哑列得分为 0，而非法配对给 −1e6 —— 于是"把某个观测空着"
        # 永远优于"硬塞一个不合法配对"，等价于允许不分配。
        from scipy.optimize import linear_sum_assignment  # 延迟导入

        M = np.zeros((n_o, n_t + n_o), dtype=np.float64)
        for i, obs in enumerate(observations):
            for j, track in enumerate(self.tracks):
                s = self._match_score(obs, track)
                # ★ 合法但得分为 0（距离正好等于门限）必须保留：
                #   旧路径用 `score > -1` 判断，0 是合法的。
                M[i, j] = s if s >= 0.0 else -1e6

        rows, cols = linear_sum_assignment(-M)
        out: List[_ObjectTrack] = []
        for i, j in zip(rows.tolist(), cols.tolist()):
            if j < n_t and M[i, j] > -1e5:
                track = self.tracks[j]
            else:
                track = _ObjectTrack(len(self.tracks), self.feature_dim,
                                     self.object_max_points)
                self.tracks.append(track)
            track.add(observations[i])
            out.append(track)
        return out

    def _merge_same_frame(self, observations: Sequence[Observation], *,
                          image_width: Optional[int] = None) -> List[Observation]:
        """把**同一帧里被拆成两段**的同一个物体合回一个观测。

        什么时候会拆：等距柱状全景里跨 ±180° 的物体会被拆成两个 bbox
        （一个 `Detection2D` 的 bbox 不能越出图像边界，见
        `pano_scene.box_to_pano` 的 `uv_parts`），于是**同一个物体在一帧里
        产生两个检测**。`greedy` 下无所谓（两条观测本来就会进同一条轨迹），
        但 `hungarian` 要求"一条轨迹在一帧里最多认领一个观测"，
        就会把它算成**两个物体**。实测（office_6，24 视角全景）：
        24 个可见 GT → 28 个检测，其中 4 个正是跨接缝的
        （table / window / wall / ceiling），而"有几个 X"这类计数
        恰好错了这 4 个 —— 一一对应，不是巧合。

        两条判据（任一命中即合并，且都要求**标签兼容**）：

        1. **图像接缝**（`image_width` 已知时）：一段的 `u1` 触到右边界、
           另一段的 `u0` 落在左边界，且竖直区间重叠 —— 这正是
           `box_to_pano` 拆框时留下的签名。
           ⚠️ 这一条**不能**用"3D IoU"代替：天花板/墙这类又宽又靠近极点的
           物体，被拆开的两段在 3D 里位于房间的**两侧**，点云几乎不相交，
           IoU ≈ 0，按 IoU 判永远合不上（我第一版就是这么写的，实测无效）。
        2. **3D IoU ≥ `assoc_merge_iou`**（默认 0.5）：覆盖"两段都投影到
           同一块体积"的普通重复检测。用 IoU 而不是中心距离，是为了不把
           "桌子底下的椅子"这类近距离异类并掉。

        两条都失效时（`assoc_merge_iou > 1` 且不传 `image_width`），
        这个方法退化成"不做任何合并"。
        """
        obs_list = list(observations)
        if len(obs_list) < 2:
            return obs_list
        iou_on = self.assoc_merge_iou is not None and self.assoc_merge_iou <= 1.0
        if not iou_on and image_width is None:
            return obs_list

        merged: List[Observation] = []
        for obs in obs_list:
            hit = -1
            for i, m in enumerate(merged):
                if not self._labels_compatible(obs.label, m.label):
                    continue
                if image_width is not None and self._wraps_seam(obs, m,
                                                               int(image_width)):
                    hit = i
                    break
                if iou_on and self._cloud_iou(obs.points_world,
                                              m.points_world) >= self.assoc_merge_iou:
                    hit = i
                    break
            if hit < 0:
                merged.append(obs)
                continue
            prev = merged[hit]
            best = prev.detection if prev.detection.score >= obs.detection.score \
                else obs.detection
            merged[hit] = Observation(
                detection=best,
                points_world=np.concatenate([prev.points_world, obs.points_world],
                                            axis=0),
                frame_id=prev.frame_id,
                timestamp=prev.timestamp,
            )
        return merged

    @staticmethod
    def _wraps_seam(a: Observation, b: Observation, width: int) -> bool:
        """两段是不是同一个"跨图像左右边界"的框被拆开的两半。

        `box_to_pano` 拆出来的两段长这样：`(u0, v0, W−1, v1)` 与
        `(0, v0, u1, v1)` —— 一段贴右边界、一段贴左边界，竖直区间相同。
        这里按这个签名判断，并要求竖直区间**有重叠**。
        """
        if width <= 1:
            return False
        box_a = np.asarray(a.detection.bbox, dtype=np.float64).reshape(-1)
        box_b = np.asarray(b.detection.bbox, dtype=np.float64).reshape(-1)
        if box_a.size < 4 or box_b.size < 4:
            return False

        def touches_left(bb) -> bool:
            return float(bb[0]) <= 0.0 + 1e-6

        def touches_right(bb) -> bool:
            return float(bb[2]) >= float(width - 1) - 1e-6

        def v_overlap(bb1, bb2) -> bool:
            lo = max(float(bb1[1]), float(bb2[1]))
            hi = min(float(bb1[3]), float(bb2[3]))
            return hi >= lo

        return ((touches_right(box_a) and touches_left(box_b))
                or (touches_right(box_b) and touches_left(box_a))) \
            and v_overlap(box_a, box_b)


    @staticmethod
    def _cloud_iou(a: np.ndarray, b: np.ndarray) -> float:
        """两团点云**轴对齐**包围盒的 3D IoU。"""
        a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
        b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
        if a.shape[0] == 0 or b.shape[0] == 0:
            return 0.0
        lo = np.maximum(a.min(axis=0), b.min(axis=0))
        hi = np.minimum(a.max(axis=0), b.max(axis=0))
        inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
        va = float(np.prod(np.clip(a.max(axis=0) - a.min(axis=0), 0.0, None)))
        vb = float(np.prod(np.clip(b.max(axis=0) - b.min(axis=0), 0.0, None)))
        union = va + vb - inter
        return float(inter / union) if union > 1e-12 else 0.0



    @staticmethod
    def _track_observation_iou(track: _ObjectTrack, obs: Observation) -> float:
        """轨迹 bbox 与观测点云 bbox 的轴对齐 3D IoU。"""
        if obs.num_points == 0 or not np.all(np.isfinite(track.bbox_min)):
            return 0.0
        o_lo = obs.points_world.min(axis=0)
        o_hi = obs.points_world.max(axis=0)
        lo = np.maximum(track.bbox_min, o_lo)
        hi = np.minimum(track.bbox_max, o_hi)
        inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
        vol_a = float(np.prod(np.clip(track.bbox_max - track.bbox_min, 0.0, None)))
        vol_b = float(np.prod(np.clip(o_hi - o_lo, 0.0, None)))
        union = vol_a + vol_b - inter
        return float(inter / union) if union > 1e-12 else 0.0

    # ---------------- 收尾 ----------------
    def finalize(self, *, extra_meta: Optional[Dict[str, Any]] = None) -> SemanticMap:
        """产出 `SemanticMap`（物体层 + 体素层）。"""
        if self.grid is None:
            self.grid = VoxelGrid(voxel_size=self.voxel_size, feature_dim=self.feature_dim)

        t0 = time.perf_counter()
        if self.object_mode == "clustering":
            objects = self._extract_objects_by_clustering()
        else:
            objects = self._extract_objects_by_association()
        t1 = time.perf_counter()

        # 把体素挂到最近的物体上（用于 `voxel_ids` 与可视化）
        if self.grid.num_voxels > 0 and objects:
            self._attach_voxels(objects)

        meta: Dict[str, Any] = {
            "num_frames": self.frames_processed,
            "num_observations": len(self.observations),
            "num_voxels": self.grid.num_voxels,
            "voxel_size": self.voxel_size,
            "fusion": str(self.cfg.get("mapping.fusion", "mean")),
            "object_mode": self.object_mode,
            "object_build_s": t1 - t0,
            "labels_seen": sorted({obs.label for obs in self.observations}),
            # 机器人（相机）位置：供"离我最近的 X"这类查询使用
            "robot_position": (
                self.camera_centers[-1].tolist() if self.camera_centers else [0.0, 0.0, 0.0]
            ),
            # 数据流丢弃统计（让"为什么地图里少了东西"可归因）
            "drop_stats": dict(self.drop_stats),
        }
        if extra_meta:
            meta.update(extra_meta)

        smap = SemanticMap(self.grid, objects, meta=meta)
        smap.text_encoder = self.text_encoder
        # 词法阈值：新键 `query.min_score_lexical` 优先，旧键 `query.min_score`
        # 作为兼容别名（含义相同，早期文档里写作 min_score）。
        #
        # ⚠️ 两个键都缺省时**不要注入任何默认值** —— 保留
        # `SemanticMap.query_min_score` 的类默认（0.5）。
        # 这里曾经写成 `self.cfg.get("query.min_score", 0.05)`，而
        # `configs/default.yaml` 里恰好写着 `min_score: 0.05`，于是类默认的
        # 0.5 被**静默覆盖**成 0.05，27.8% 的未见类别查询返回错误物体
        # （详见 AGENTS.md 二.7 与 tests/test_pipeline_e2e.py 的配置路径回归锁）。
        lex_thr = self.cfg.get("query.min_score_lexical", None)
        if lex_thr is None:
            lex_thr = self.cfg.get("query.min_score", None)
        if lex_thr is not None:
            smap.query_min_score = float(lex_thr)
        # 只有在配置里**显式**给了嵌入阈值时才覆盖；否则留 None，
        # 让 QueryEngine 按编码器类型自适应（SigLIP 0.02 / CLIP 校准后 0.9）
        emb_thr = self.cfg.get("query.min_score_embedding", None)
        if emb_thr is not None:
            smap.query_min_score_embedding = float(emb_thr)
        logger.info(
            f"物体构建完成（{self.object_mode}）：{self.grid.num_voxels} 体素 → "
            f"{len(objects)} 个物体，耗时 {t1 - t0:.3f}s"
        )
        ds = self.drop_stats
        if ds["no_valid_depth"]:
            logger.info(
                f"数据流统计：{ds['detections']} 个检测 → 丢弃 "
                f"{ds['no_valid_depth']} 个（掩码区域无有效深度，常见于门窗/玻璃/过远）"
                f"、{ds['low_confidence']} 个（低置信度）"
            )
        return smap

    # ---------------- 物体构建：实例关联 ----------------
    def _extract_objects_by_association(self) -> List[SemanticObject]:
        """从关联好的轨迹生成物体（推荐路径）。"""
        pct = float(self.cfg.get("mapping.object_bbox_percentile", 2.0))
        objects: List[SemanticObject] = []

        for track in self.tracks:
            if track.num_observations < self.min_observations:
                continue
            if track.points_count == 0:
                continue
            objects.append(track.to_object(-1, percentile=pct))

        # 按置信度排序后重编号，保证 obj_id 稳定且有意义
        objects.sort(key=lambda o: -o.confidence)
        for i, obj in enumerate(objects):
            obj.obj_id = i
        return objects

    # ---------------- 物体构建：体素聚类（降级路径）----------------
    def _extract_objects_by_clustering(self) -> List[SemanticObject]:
        """对体素做 DBSCAN 聚类（**没有实例信息时**才用）。

        ⚠️ 已知局限：两个靠得很近的物体会被合并（因为聚类只看空间距离，
        不知道"它们本来就是两个检测实例"）。这正是默认走关联模式的原因。
        """
        grid = self.grid
        if grid is None or grid.num_voxels == 0:
            return []

        centers = grid.centers
        features = grid.features
        counts = grid.counts
        top_votes = grid.top_labels(top=3)

        labels = cluster_points_dbscan(
            centers,
            eps=float(self.cfg.get("mapping.cluster_eps", 0.08)),
            min_samples=int(self.cfg.get("mapping.cluster_min_samples", 3)),
        )

        min_voxels = int(self.cfg.get("mapping.object_min_voxels", 5))
        pct = float(self.cfg.get("mapping.object_bbox_percentile", 2.0))

        objects: List[SemanticObject] = []
        for cluster_id in sorted(set(labels.tolist())):
            if cluster_id < 0:
                continue
            sel = np.flatnonzero(labels == cluster_id)
            if sel.size < min_voxels:
                continue

            pts = centers[sel]
            feats = features[sel]
            cnt = counts[sel].astype(np.float64)

            votes: Dict[str, int] = {}
            for i in sel:
                for label, n in (top_votes[int(i)] if int(i) < len(top_votes) else []):
                    if label:
                        votes[label] = votes.get(label, 0) + int(n)

            center = (pts * cnt[:, None]).sum(axis=0) / max(cnt.sum(), 1e-9)
            if pct > 0 and pts.shape[0] >= 5:
                lo = np.percentile(pts, pct, axis=0)
                hi = np.percentile(pts, 100.0 - pct, axis=0)
            else:
                lo, hi = pts.min(axis=0), pts.max(axis=0)

            feature = self._weighted_feature_mean(feats, cnt)
            total = sum(votes.values()) or 1
            label = max(votes, key=votes.get) if votes else "object"
            purity = votes.get(label, 0) / total
            confidence = float(
                (min(1.0, sel.size / 10.0) * min(1.0, cnt.sum() / 20.0) * max(purity, 1e-3))
                ** (1.0 / 3.0)
            )

            objects.append(SemanticObject(
                obj_id=0,
                label=label,
                center=center,
                bbox_min=lo,
                bbox_max=hi,
                feature=feature,
                num_voxels=int(sel.size),
                num_points=int(cnt.sum()),
                confidence=confidence,
                label_scores=dict(sorted(votes.items(), key=lambda kv: -kv[1])[:5]),
                voxel_ids=sel.astype(np.int64),
            ))

        objects.sort(key=lambda o: -o.confidence)
        for i, obj in enumerate(objects):
            obj.obj_id = i
        return objects

    # ---------------- 体素 → 物体挂载 ----------------
    def _attach_voxels(self, objects: List[SemanticObject], *, margin: float = 0.15) -> None:
        """把每个体素分配给最近的物体（在其 bbox 外扩 margin 范围内）。"""
        centers = self.grid.centers
        if centers.shape[0] == 0:
            return

        obj_centers = np.stack([o.center for o in objects], axis=0)      # (K,3)
        # 距离矩阵 (M,K)：体素数通常 1e3~1e5，K 通常 <100，内存可控
        d2 = ((centers[:, None, :] - obj_centers[None, :, :]) ** 2).sum(axis=-1)
        nearest = np.argmin(d2, axis=1)
        nearest_dist = np.sqrt(d2[np.arange(centers.shape[0]), nearest])

        assign: Dict[int, List[int]] = {i: [] for i in range(len(objects))}
        for vi in range(centers.shape[0]):
            oi = int(nearest[vi])
            obj = objects[oi]
            lo = obj.bbox_min - margin
            hi = obj.bbox_max + margin
            c = centers[vi]
            if np.all(c >= lo) and np.all(c <= hi):
                assign[oi].append(vi)

        for oi, voxels in assign.items():
            objects[oi].voxel_ids = np.asarray(voxels, dtype=np.int64)
            objects[oi].num_voxels = len(voxels)

    @staticmethod
    def _weighted_feature_mean(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        if feats.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        mean = (feats * w).sum(axis=0) / max(float(w.sum()), 1e-9)
        out = mean.astype(np.float32)
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 1e-8 else out

    # ---------------- 统计 ----------------
    def stats(self) -> Dict[str, Any]:
        return {
            "frames": self.frames_processed,
            "observations": len(self.observations),
            "tracks": len(self.tracks),
            "voxels": self.grid.num_voxels if self.grid else 0,
            "labels": sorted({o.label for o in self.observations}),
            **{k: round(v, 4) for k, v in self.timings.items()},
        }

    def __repr__(self) -> str:
        return (
            f"MapBuilder(voxel_size={self.voxel_size}, mode={self.object_mode!r}, "
            f"prompts={len(self.prompts)}, pipeline={self.pipeline!r})"
        )
