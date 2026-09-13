"""开放词汇 3D 语义地图的数据结构与查询接口。

地图的分层
----------
```
SemanticMap
├── VoxelGrid          3D 特征场（每个体素一个聚合特征 + 标签投票）
└── SemanticObject[]   把体素连通域封装成的"物体"（有名字、有 bbox、有中心）
```

- **体素层**：开放词汇的载体 —— 特征和任意文本可比，所以能查询没见过的类别；
- **物体层**：机器人实际要用的东西 —— 规划模块需要"一个杯子在哪"，
  而不是"327 个体素激活了"。

两层都保留，查询时可选择返回哪一层（`QueryResult.level`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from roboground.geometry.voxel import VoxelGrid
from roboground.types import SpatialRelation
from roboground.utils.io import ensure_dir, load_npz, save_npz, save_json, load_json


# ==========================================================================
# 物体
# ==========================================================================
@dataclass
class SemanticObject:
    """地图中的一个 3D 物体实例（由体素连通域聚类而来）。

    Attributes
    ----------
    obj_id : int
    label : str
        票数最高的文本标签。
    label_scores : dict
        标签 → 票数（多标签时体现歧义程度）。
    center : (3,) float64
        几何中心（体素坐标的均值）。
    bbox_min, bbox_max : (3,) float64
        轴对齐包围盒（用分位数裁剪，抗离群点）。
    num_voxels, num_points : int
    feature : (D,) float32
        物体级特征（成员体素特征的均值），用于文本相似度查询。
    confidence : float
        置信度 ∈ [0,1]，由体素数量、观测次数、标签票数比例综合而来。
    voxel_ids : (K,) int64
        成员体素在 `SemanticMap.voxel_grid` 中的行号。
    """

    obj_id: int
    label: str
    center: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    feature: np.ndarray
    num_voxels: int = 0
    num_points: int = 0
    confidence: float = 0.0
    label_scores: Dict[str, int] = field(default_factory=dict)
    voxel_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    frame_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.bbox_min = np.asarray(self.bbox_min, dtype=np.float64).reshape(3)
        self.bbox_max = np.asarray(self.bbox_max, dtype=np.float64).reshape(3)
        self.feature = np.asarray(self.feature, dtype=np.float32).reshape(-1)
        self.voxel_ids = np.asarray(self.voxel_ids, dtype=np.int64).reshape(-1)

    # ---------------- 几何属性 ----------------
    @property
    def extent(self) -> np.ndarray:
        """尺寸（长宽高，米）。"""
        return self.bbox_max - self.bbox_min

    @property
    def size(self) -> np.ndarray:
        """`extent` 的别名。"""
        return self.extent

    @property
    def volume(self) -> float:
        return float(np.prod(np.clip(self.extent, 0.0, None)))

    @property
    def height(self) -> float:
        return float(self.extent[2])

    def distance_to(self, other: Union["SemanticObject", np.ndarray]) -> float:
        """到另一个物体（或一个 3D 点）中心的欧氏距离（米）。"""
        target = other.center if isinstance(other, SemanticObject) else np.asarray(other, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(self.center - target))

    def contains_point(self, point: Sequence[float], *, margin: float = 0.0) -> bool:
        p = np.asarray(point, dtype=np.float64).reshape(3)
        return bool(np.all(p >= self.bbox_min - margin) and np.all(p <= self.bbox_max + margin))

    def iou_3d(self, other: "SemanticObject") -> float:
        """轴对齐 3D IoU（用于评测与去重）。"""
        lo = np.maximum(self.bbox_min, other.bbox_min)
        hi = np.minimum(self.bbox_max, other.bbox_max)
        inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
        union = self.volume + other.volume - inter
        return float(inter / union) if union > 1e-12 else 0.0

    # ---------------- 序列化 ----------------
    def to_dict(self, *, with_feature: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "obj_id": int(self.obj_id),
            "label": self.label,
            "center": [round(float(v), 4) for v in self.center],
            "bbox_min": [round(float(v), 4) for v in self.bbox_min],
            "bbox_max": [round(float(v), 4) for v in self.bbox_max],
            "size": [round(float(v), 4) for v in self.extent],
            "num_voxels": int(self.num_voxels),
            "num_points": int(self.num_points),
            "confidence": round(float(self.confidence), 4),
            "label_scores": dict(self.label_scores),
            "frame_ids": list(self.frame_ids),
        }
        if with_feature:
            payload["feature"] = self.feature.tolist()
        return payload

    def describe(self) -> str:
        """人类可读的一行描述（demo 输出用）。"""
        size = self.extent
        return (
            f"[{self.obj_id:>3}] {self.label:<14} center=("
            f"{self.center[0]:+.2f}, {self.center[1]:+.2f}, {self.center[2]:+.2f}) "
            f"size=({size[0]:.2f}×{size[1]:.2f}×{size[2]:.2f})m "
            f"conf={self.confidence:.2f} voxels={self.num_voxels}"
        )


# ==========================================================================
# 查询结果
# ==========================================================================
@dataclass
class QueryResult:
    """一次查询返回的一条命中。

    `level` 区分命中粒度：
    - `"object"`：命中一个聚好的物体（推荐，机器人直接用）
    - `"voxel"`：命中零散体素（物体层还没聚类好时的降级）
    """

    score: float
    level: str = "object"
    obj: Optional[SemanticObject] = None
    center: Optional[np.ndarray] = None
    label: Optional[str] = None
    voxel_ids: Optional[np.ndarray] = None
    matched_by: str = "lexical"          # lexical | embedding | hybrid
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        """命中位置（3D 点，米）。"""
        if self.center is not None:
            return np.asarray(self.center, dtype=np.float64).reshape(3)
        if self.obj is not None:
            return self.obj.center
        raise ValueError("该 QueryResult 没有位置信息")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 4),
            "level": self.level,
            "label": self.label if self.label is not None else (self.obj.label if self.obj else None),
            "position": [round(float(v), 4) for v in self.position],
            "matched_by": self.matched_by,
            "object": self.obj.to_dict() if self.obj is not None else None,
            "extra": self.extra,
        }

    def __repr__(self) -> str:
        label = self.label or (self.obj.label if self.obj else "?")
        pos = self.position
        return (
            f"QueryResult({label!r}, score={self.score:.3f}, "
            f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}), "
            f"level={self.level}, by={self.matched_by})"
        )


# ==========================================================================
# 地图
# ==========================================================================
class SemanticMap:
    """开放词汇 3D 语义地图。

    Examples
    --------
    >>> from roboground.geometry import VoxelGrid
    >>> grid = VoxelGrid(voxel_size=0.05, feature_dim=4)
    >>> smap = SemanticMap(voxel_grid=grid)
    >>> smap.num_objects
    0
    """

    def __init__(
        self,
        voxel_grid: VoxelGrid,
        objects: Optional[Iterable[SemanticObject]] = None,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.voxel_grid = voxel_grid
        self.objects: List[SemanticObject] = list(objects or [])
        self.meta: Dict[str, Any] = dict(meta or {})
        # 文本编码器（可选）：能被 builder 注入，用于 embedding 查询
        self.text_encoder = None
        # 查询分数下限：低于它的候选直接丢弃。
        # 为什么需要？词法匹配对不相关标签打 0 分，若不过滤会返回一堆
        # "零分命中"，让调用方误以为查到了东西（实测踩过这个坑）。
        #
        # ⚠️ 但**不能设得过低**：词法的第三层打分是字符 bigram Jaccard（上限 0.80），
        # 只要共享少量 bigram 就有 0.1~0.3 分，属于纯噪声。实测阈值 0.05 会让
        # 27.8% 的"地图里没有的类别"查询返回错误物体；抬到 0.5 后拒识率 100%
        # 且命名类命中率不降。详见 `QueryEngine.min_score_lexical` 的实测记录。
        self.query_min_score: float = 0.5
        # 嵌入匹配用**另一个阈值**：它的分数尺度与词法完全不同。
        # 为 None 时表示"按编码器自适应"（SigLIP 0.02 / CLIP 空文本校准 0.9），
        # 见 `QueryEngine._suggest_threshold`。**不要**硬编码一个全局值 ——
        # 实测用 0.5 会让 SigLIP 的所有查询都返回空。
        #
        # ⚠️ 还要知道：SigLIP 声明的 0.02 是在**检测区域特征**上标定的，而这里
        # 比对的是**多视角融合后的地图物体特征**，后者分数尺度小约 20 倍
        # （正确命中只有 1e-3 量级）。而且同一个阈值在不同模式下**角色不同**：
        # - hybrid：词法已负责接受判定，嵌入阈值退化为**精度过滤器** → 0.02 最优；
        # - 纯嵌入：它就是唯一的**接受阈值** → 0.0005 最优（差 40 倍）。
        # 引擎会按自身模式自动选（`QueryEngine._suggest_threshold(standalone=...)`）。
        # 实测 hybrid@(0.5, 0.02)：命名 100% / 描述 37.5% / 拒识 100%，综合分 0.834，
        # 优于纯词法的 0.828，也优于把嵌入阈值放宽到 0.01 的 0.818。
        # 调参前请先跑 scripts/16_eval_openvocab_query.py 看扫描曲线。
        self.query_min_score_embedding: Optional[float] = None

    # ---------------- 基本属性 ----------------
    @property
    def num_voxels(self) -> int:
        return self.voxel_grid.num_voxels

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    @property
    def feature_dim(self) -> int:
        return self.voxel_grid.feature_dim

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """地图的轴对齐包围盒 (min, max)。空地图返回全 0。"""
        if self.num_voxels == 0:
            return np.zeros(3), np.zeros(3)
        centers = self.voxel_grid.centers
        return centers.min(axis=0), centers.max(axis=0)

    @property
    def labels(self) -> List[str]:
        """地图中出现的所有标签（去重，按出现次数排序）。"""
        counts: Dict[str, int] = {}
        for obj in self.objects:
            counts[obj.label] = counts.get(obj.label, 0) + obj.num_voxels
        if not counts:
            for votes in self.voxel_grid.top_labels(top=1):
                if votes and votes[0][0]:
                    counts[votes[0][0]] = counts.get(votes[0][0], 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])]

    def object_by_id(self, obj_id: int) -> Optional[SemanticObject]:
        for obj in self.objects:
            if obj.obj_id == obj_id:
                return obj
        return None

    # ---------------- 空间查询（不需要语义） ----------------
    def objects_near(
        self,
        point: Sequence[float],
        radius: float = 1.0,
        *,
        exclude: Optional[SemanticObject] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[SemanticObject, float]]:
        """找某个 3D 点附近半径内的物体，按距离升序。"""
        p = np.asarray(point, dtype=np.float64).reshape(3)
        hits: List[Tuple[SemanticObject, float]] = []
        for obj in self.objects:
            if exclude is not None and obj.obj_id == exclude.obj_id:
                continue
            d = float(np.linalg.norm(obj.center - p))
            if d <= radius:
                hits.append((obj, d))
        hits.sort(key=lambda kv: kv[1])
        return hits[:top_k] if top_k else hits

    def nearest_object(
        self,
        point: Sequence[float],
        *,
        exclude: Optional[SemanticObject] = None,
    ) -> Optional[Tuple[SemanticObject, float]]:
        hits = self.objects_near(point, radius=float("inf"), exclude=exclude, top_k=1)
        return hits[0] if hits else None

    def objects_in_bbox(
        self,
        bbox_min: Sequence[float],
        bbox_max: Sequence[float],
    ) -> List[SemanticObject]:
        """找中心落在给定包围盒内的物体（用于"把这个区域的东西列出来"）。"""
        lo = np.asarray(bbox_min, dtype=np.float64).reshape(3)
        hi = np.asarray(bbox_max, dtype=np.float64).reshape(3)
        out = []
        for obj in self.objects:
            if np.all(obj.center >= lo) and np.all(obj.center <= hi):
                out.append(obj)
        return out

    # ---------------- 语义查询 ----------------
    def query_text(
        self,
        text: str,
        *,
        top_k: int = 5,
        level: str = "auto",
        text_encoder: Any = None,
        min_score: Optional[float] = None,
        **kwargs,
    ) -> List[QueryResult]:
        """自然语言查询：文本 → 3D 位置。

        Parameters
        ----------
        text
            自然语言 query，如 "杯子在哪" / "the red cup" / "table"。
        top_k
            返回条数。
        level
            `"object"` 只返回物体层；`"voxel"` 只返回体素层；`"auto"` 优先物体层，
            物体层为空时退回体素层。
        text_encoder
            可选文本编码器（CLIP 类）。不传则用地图自带的；都没有时
            自动走**词法匹配**（离线可用）。
        min_score
            词法路径的分数下限，None 时用 `self.query_min_score`（默认 **0.5**，
            不是 0.05 —— 见该属性的注释，调低会让未见类别被 bigram 噪声蒙中）。

        Returns
        -------
        List[QueryResult]
        """
        from roboground.mapping.query import QueryEngine  # 延迟导入避免循环

        engine = QueryEngine(self, text_encoder=text_encoder)
        return engine.query(
            text,
            top_k=top_k,
            level=level,
            min_score=self.query_min_score if min_score is None else float(min_score),
            min_score_embedding=self.query_min_score_embedding,
            **kwargs,
        )

    def describe(self, max_objects: int = 20) -> str:
        """地图摘要文本（可喂给 VLM 作为场景上下文）。"""
        lo, hi = self.bounds
        lines = [
            f"SemanticMap: {self.num_objects} objects, {self.num_voxels} voxels, "
            f"feature_dim={self.feature_dim}, voxel_size={self.voxel_grid.voxel_size}m",
            f"bounds: min=({lo[0]:+.2f},{lo[1]:+.2f},{lo[2]:+.2f}) "
            f"max=({hi[0]:+.2f},{hi[1]:+.2f},{hi[2]:+.2f})",
            f"labels: {', '.join(self.labels[:15]) or '(none)'}",
            "objects:",
        ]
        ordered = sorted(self.objects, key=lambda o: -o.confidence)
        for obj in ordered[:max_objects]:
            lines.append("  " + obj.describe())
        if len(ordered) > max_objects:
            lines.append(f"  ... 还有 {len(ordered) - max_objects} 个物体")
        return "\n".join(lines)

    def to_dict(self, *, with_features: bool = True) -> Dict[str, Any]:
        """导出为可序列化结构（体素特征以 ndarray 保留）。"""
        payload: Dict[str, Any] = {
            "meta": self.meta,
            "objects": [o.to_dict(with_feature=with_features) for o in self.objects],
            "object_voxel_ids": [o.voxel_ids for o in self.objects],
        }
        if with_features:
            payload.update({
                "voxel_size": np.float64(self.voxel_grid.voxel_size),
                "feature_dim": np.int64(self.voxel_grid.feature_dim),
                "mode": np.str_(self.voxel_grid.mode),
                "origin": self.voxel_grid.origin,
                "voxel_keys": self.voxel_grid.keys,
                "voxel_features": self.voxel_grid.features,
                "voxel_counts": self.voxel_grid.counts,
                "voxel_labels": np.array(
                    [v.most_common(1)[0][0] if v else "" for v in self.voxel_grid._label_votes],
                    dtype=object,
                ),
            })
        return payload

    # ---------------- 持久化 ----------------
    def save(self, path: Union[str, Path], *, save_json_summary: bool = True) -> Path:
        """保存地图（npz + 可选 json 摘要）。

        npz 存体素特征（大、二进制）；json 存物体列表（小、可读）。
        """
        path = Path(path)
        ensure_dir(path.parent)
        # 1) 物体列表 → json（小、可读、含特征）
        save_json(
            {
                "meta": self.meta,
                "objects": [o.to_dict(with_feature=True) for o in self.objects],
                "object_voxel_ids": [o.voxel_ids.tolist() for o in self.objects],
            },
            path.parent / (path.stem + "_objects.json"),
        )

        # 2) 体素特征场 → npz（大、二进制）
        arrays = {
            "voxel_size": np.float64(self.voxel_grid.voxel_size),
            "feature_dim": np.int64(self.voxel_grid.feature_dim),
            "mode": np.str_(self.voxel_grid.mode),
            "origin": self.voxel_grid.origin,
            "voxel_keys": self.voxel_grid.keys,
            "voxel_features": self.voxel_grid.features,
            "voxel_counts": self.voxel_grid.counts,
            "voxel_labels": np.array(
                [v.most_common(1)[0][0] if v else "" for v in self.voxel_grid._label_votes],
                dtype=object,
            ),
        }
        save_npz(path, **arrays)

        # 3) 人类可读摘要 → txt（方便快速 check 一张地图的内容）
        if save_json_summary:
            summary_path = path.parent / (path.stem + "_summary.txt")
            summary_path.write_text(self.describe(), encoding="utf-8")

        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SemanticMap":
        """加载地图（与 `save` 对应）。"""
        path = Path(path)
        arrays = load_npz(path)
        objects_path = path.parent / (path.stem + "_objects.json")
        obj_data = load_json(objects_path, default={"objects": []})

        grid = VoxelGrid.from_dict({
            "voxel_size": arrays["voxel_size"],
            "feature_dim": arrays["feature_dim"],
            "mode": arrays["mode"],
            "origin": arrays["origin"],
            "keys": arrays["voxel_keys"],
            "features": arrays["voxel_features"],
            "counts": arrays["voxel_counts"],
            "labels": arrays["voxel_labels"],
        })

        objects: List[SemanticObject] = []
        voxel_id_lists = obj_data.get("object_voxel_ids", [])
        for i, raw in enumerate(obj_data.get("objects", [])):
            objects.append(SemanticObject(
                obj_id=int(raw["obj_id"]),
                label=str(raw["label"]),
                center=np.asarray(raw["center"], dtype=np.float64),
                bbox_min=np.asarray(raw["bbox_min"], dtype=np.float64),
                bbox_max=np.asarray(raw["bbox_max"], dtype=np.float64),
                feature=np.asarray(raw.get("feature", []), dtype=np.float32),
                num_voxels=int(raw.get("num_voxels", 0)),
                num_points=int(raw.get("num_points", 0)),
                confidence=float(raw.get("confidence", 0.0)),
                label_scores=dict(raw.get("label_scores", {})),
                voxel_ids=np.asarray(
                    voxel_id_lists[i] if i < len(voxel_id_lists) else [], dtype=np.int64
                ),
                frame_ids=list(raw.get("frame_ids", [])),
            ))

        return cls(grid, objects, meta=obj_data.get("meta", {}))

    def __repr__(self) -> str:
        return (
            f"SemanticMap(objects={self.num_objects}, voxels={self.num_voxels}, "
            f"feature_dim={self.feature_dim}, labels={self.labels[:5]})"
        )
