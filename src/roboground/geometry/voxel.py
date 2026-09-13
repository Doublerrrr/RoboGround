"""体素化与 3D 特征场 —— 开放词汇 3D 语义地图的存储核心。

核心思想（OpenScene / CLIP-Fields 范式）
--------------------------------------
把 3D 空间切成固定边长的体素，每个体素聚合"落在它里面的所有 2D 像素特征"，
得到一张 **稠密的 3D 特征场**。之后任意自然语言 query 都能通过
"文本特征 × 体素特征" 的相似度直接在 3D 空间里定位 —— 这就是
"开放词汇"在 3D 侧的落地方式。

为什么用体素而不是纯点云？
- 点云稀疏且有冗余（一个物体几万个点，但语义只有一份）；
- 体素聚合天然完成**多视角融合**（同一体素被多帧看到 → 特征平均 → 抗噪）；
- 体素键是整数，方便做哈希、连通域聚类和序列化。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("geometry.voxel")

_VALID_MODES = ("mean", "max", "confidence_weighted")


# ==========================================================================
# 底层工具
# ==========================================================================
def voxel_keys(
    points: np.ndarray,
    voxel_size: float,
    origin: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """世界坐标 → 整数体素索引 (N,3)。

    Parameters
    ----------
    points : (N,3)
    voxel_size : float
        体素边长（米），必须 > 0。
    origin : (3,), optional
        体素网格原点，默认全 0。

    Returns
    -------
    np.ndarray, shape (N,3), dtype int64
        每个点所属体素的整数索引。
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size 必须为正，收到 {voxel_size}")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    org = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64).reshape(3)
    return np.floor((pts - org) / float(voxel_size)).astype(np.int64)


def voxel_centers(keys: np.ndarray, voxel_size: float,
                  origin: Optional[Sequence[float]] = None) -> np.ndarray:
    """体素索引 → 体素中心坐标 (M,3)。"""
    keys = np.asarray(keys, dtype=np.float64).reshape(-1, 3)
    org = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64).reshape(3)
    return (keys + 0.5) * float(voxel_size) + org


def _unique_rows(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 (N,K) 整数矩阵去重，返回 (唯一行, 反向索引, 每行计数)。

    用 `void` 视图把每行压成单个元素，比 `np.unique(axis=0)` 快数倍 ——
    这是热路径（每帧都要跑），所以做了这个优化。
    """
    rows = np.ascontiguousarray(rows)
    if rows.shape[0] == 0:
        return rows, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    view = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
    _, index, inverse, counts = np.unique(
        view, return_index=True, return_inverse=True, return_counts=True
    )
    return rows[index], inverse.ravel(), counts


def aggregate_features_by_voxel(
    points: np.ndarray,
    features: np.ndarray,
    voxel_size: float,
    *,
    mode: str = "mean",
    weights: Optional[np.ndarray] = None,
    origin: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一次性把点云特征聚合成体素特征场（非增量版本）。

    Returns
    -------
    centers : (M,3) float64
        体素中心。
    aggregated : (M,D) float32
        聚合后的特征。
    counts : (M,) int64
        每个体素包含的点数。
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode 必须是 {_VALID_MODES} 之一，收到 {mode!r}")

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    if pts.shape[0] != feats.shape[0]:
        raise ValueError(f"点与特征数量不一致：{pts.shape[0]} vs {feats.shape[0]}")

    if pts.shape[0] == 0:
        dim = feats.shape[1] if feats.ndim == 2 else 0
        return np.zeros((0, 3)), np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    w = np.ones(pts.shape[0], dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32).reshape(-1)
    if mode == "mean":
        w = np.ones_like(w)

    keys = voxel_keys(pts, voxel_size, origin)
    uniq, inverse, counts = _unique_rows(keys)
    m, d = uniq.shape[0], feats.shape[1]

    agg = np.zeros((m, d), dtype=np.float32)
    if mode == "max":
        agg[:] = -np.inf
        np.maximum.at(agg, inverse, feats)
        agg[~np.isfinite(agg)] = 0.0
    else:  # mean / confidence_weighted：加权求和后归一化
        weighted = feats * w[:, None]
        np.add.at(agg, inverse, weighted)
        wsum = np.zeros(m, dtype=np.float32)
        np.add.at(wsum, inverse, w)
        agg = agg / np.clip(wsum, 1e-8, None)[:, None]

    return voxel_centers(uniq, voxel_size, origin), agg, counts


# ==========================================================================
# 增量式体素网格（流式建图用）
# ==========================================================================
class VoxelGrid:
    """增量式 3D 特征场，支持逐帧累加与多视角特征融合。

    与 `aggregate_features_by_voxel` 的区别：这个类维护**跨帧持久**的状态，
    适合"机器人边走边建图"的流式场景。

    多视角融合的意义：同一个体素被 N 帧看到时，
    - `mean`  : 特征平均 → 抑制单帧噪声；
    - `max`   : 逐维取最大 → 保留最显著的语义响应（OpenScene 常用）；
    - `confidence_weighted` : 按检测置信度加权 → 高置信观测主导。

    Examples
    --------
    >>> grid = VoxelGrid(voxel_size=0.05, feature_dim=3)
    >>> grid.add(np.array([[0,0,0],[0.01,0,0]]), np.array([[1,0,0],[1,0,0]]))
    >>> grid.num_voxels
    1
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        feature_dim: int = 256,
        *,
        mode: str = "mean",
        max_voxels: int = 2_000_000,
        origin: Optional[Sequence[float]] = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode 必须是 {_VALID_MODES} 之一，收到 {mode!r}")
        self.voxel_size = float(voxel_size)
        self.feature_dim = int(feature_dim)
        self.mode = mode
        self.max_voxels = int(max_voxels)
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64).reshape(3)

        # 键 → 行号（字典是唯一状态源，其余数组按行对齐）
        self._index: Dict[Tuple[int, int, int], int] = {}
        self._keys: List[Tuple[int, int, int]] = []
        self._feat_sum: List[np.ndarray] = []       # 加权和（mean / conf_weighted）
        self._feat_max: List[np.ndarray] = []       # 逐维最大（max）
        self._weight_sum: List[float] = []
        self._counts: List[int] = []
        self._label_votes: List[Counter] = []
        self._frames_seen: set = set()

        # 统计
        self.total_points: int = 0

    # ---------------- 属性 ----------------
    @property
    def num_voxels(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> np.ndarray:
        if not self._keys:
            return np.zeros((0, 3), dtype=np.int64)
        return np.asarray(self._keys, dtype=np.int64)

    @property
    def centers(self) -> np.ndarray:
        return voxel_centers(self.keys, self.voxel_size, self.origin)

    @property
    def counts(self) -> np.ndarray:
        return np.asarray(self._counts, dtype=np.int64)

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self._weight_sum, dtype=np.float32)

    @property
    def features(self) -> np.ndarray:
        """返回当前聚合特征 (M,D) float32。"""
        m = self.num_voxels
        if m == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        if self.mode == "max":
            out = np.stack(self._feat_max, axis=0)
            out[~np.isfinite(out)] = 0.0
            return out.astype(np.float32)
        sums = np.stack(self._feat_sum, axis=0)
        wsum = np.asarray(self._weight_sum, dtype=np.float32)
        return (sums / np.clip(wsum, 1e-8, None)[:, None]).astype(np.float32)

    @property
    def frames_seen(self) -> List[str]:
        return sorted(self._frames_seen)

    def top_labels(self, top: int = 1) -> List[List[Tuple[str, int]]]:
        """每个体素票数最高的标签（用于可解释性与封闭集降级）。"""
        out: List[List[Tuple[str, int]]] = []
        for votes in self._label_votes:
            out.append(votes.most_common(top))
        return out

    # ---------------- 写入 ----------------
    def add(
        self,
        points: np.ndarray,
        features: Optional[np.ndarray] = None,
        *,
        weights: Optional[np.ndarray] = None,
        labels: Optional[Sequence[str]] = None,
        frame_id: Optional[str] = None,
    ) -> int:
        """加入一批点及其特征（通常来自一帧的一个或多个检测实例）。

        Parameters
        ----------
        points : (N,3)
            世界系点坐标。
        features : (N,D), optional
            每点的语义特征（同一实例内通常相同）。None 时用全 1 占位。
        weights : (N,), optional
            每点权重（置信度）。`mean` 模式下会被忽略。
        labels : (N,), optional
            每点的文本标签，用于投票统计。
        frame_id : str, optional
            来源帧，用于统计"覆盖了多少帧"。

        Returns
        -------
        int
            本次新增的体素数量。
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return 0

        if features is None:
            feats = np.ones((pts.shape[0], self.feature_dim), dtype=np.float32)
        else:
            feats = np.asarray(features, dtype=np.float32)
            if feats.ndim == 1:
                feats = np.tile(feats.reshape(1, -1), (pts.shape[0], 1))
            if feats.shape[0] != pts.shape[0]:
                raise ValueError(f"点与特征数量不一致：{pts.shape[0]} vs {feats.shape[0]}")
            if feats.shape[1] != self.feature_dim:
                # 允许首次调用时推断维度
                if self.num_voxels == 0:
                    self.feature_dim = int(feats.shape[1])
                else:
                    raise ValueError(
                        f"特征维度不匹配：网格是 {self.feature_dim}，输入是 {feats.shape[1]}"
                    )

        if self.mode == "mean":
            w = np.ones(pts.shape[0], dtype=np.float32)
        elif weights is None:
            w = np.ones(pts.shape[0], dtype=np.float32)
        else:
            w = np.asarray(weights, dtype=np.float32).reshape(-1)
            if w.shape[0] != pts.shape[0]:
                raise ValueError(f"权重数量不一致：{w.shape[0]} vs {pts.shape[0]}")

        if self.num_voxels >= self.max_voxels:
            logger.warn(
                f"体素数量已达上限 {self.max_voxels}，后续观测被丢弃（可调大 "
                f"mapping.max_voxels 或增大 geometry.voxel_size）"
            )
            return 0

        # ---- 帧内 numpy 聚合（把 Python 层操作降到"唯一体素数"量级）----
        k = voxel_keys(pts, self.voxel_size, self.origin)
        uniq, inverse, counts = _unique_rows(k)
        m, d = uniq.shape[0], feats.shape[1]

        weighted = feats * w[:, None]
        fsum = np.zeros((m, d), dtype=np.float64)
        np.add.at(fsum, inverse, weighted)

        wsum = np.zeros(m, dtype=np.float64)
        np.add.at(wsum, inverse, w.astype(np.float64))

        if self.mode == "max":
            fmax = np.full((m, d), -np.inf, dtype=np.float64)
            np.maximum.at(fmax, inverse, feats.astype(np.float64))

        # ---- 逐体素并入字典 ----
        added = 0
        label_arr = None if labels is None else list(labels)

        for row in range(m):
            key = (int(uniq[row, 0]), int(uniq[row, 1]), int(uniq[row, 2]))
            if label_arr is not None:
                # 该体素内出现次数最多的标签（同一实例内标签一致，取首个即可）
                first = int(np.flatnonzero(inverse == row)[0])
                label = str(label_arr[first]) if first < len(label_arr) else ""
            else:
                label = ""

            idx = self._index.get(key)
            if idx is None:
                if self.num_voxels >= self.max_voxels:
                    break
                self._index[key] = self.num_voxels
                self._keys.append(key)
                self._feat_sum.append(fsum[row].astype(np.float32))
                self._weight_sum.append(float(wsum[row]))
                self._counts.append(int(counts[row]))
                if self.mode == "max":
                    self._feat_max.append(np.where(np.isfinite(fmax[row]), fmax[row], 0.0).astype(np.float32))
                votes: Counter = Counter()
                if label:
                    votes[label] += int(counts[row])
                self._label_votes.append(votes)
                added += 1
                continue

            # 已存在 → 融合
            self._counts[idx] += int(counts[row])
            self._weight_sum[idx] += float(wsum[row])
            if self.mode == "max":
                cur = self._feat_max[idx]
                new = np.where(np.isfinite(fmax[row]), fmax[row], 0.0).astype(np.float32)
                self._feat_max[idx] = np.maximum(cur, new)
                # max 模式下 feat_sum 也维护，用于导出统计
                self._feat_sum[idx] += fsum[row].astype(np.float32)
            else:
                self._feat_sum[idx] += fsum[row].astype(np.float32)
            if label:
                self._label_votes[idx][label] += int(counts[row])

        self.total_points += int(pts.shape[0])
        if frame_id is not None:
            self._frames_seen.add(str(frame_id))
        return added

    def add_observation(self, observation, *, weights: Optional[np.ndarray] = None) -> int:
        """便捷入口：直接吃一个 `Observation`。

        自动把该实例的特征广播到它的所有点上，并按检测置信度加权。
        """
        pts = observation.points_world
        if pts.shape[0] == 0:
            return 0
        feat = observation.feature
        if feat is None:
            feats = None
        else:
            feats = np.tile(np.asarray(feat, dtype=np.float32).reshape(1, -1), (pts.shape[0], 1))
        w = weights
        if w is None and self.mode == "confidence_weighted":
            w = np.full(pts.shape[0], float(observation.detection.score), dtype=np.float32)
        labels = [observation.label] * pts.shape[0]
        return self.add(
            pts, feats, weights=w, labels=labels, frame_id=observation.frame_id
        )

    # ---------------- 序列化 ----------------
    def to_dict(self) -> Dict:
        """导出为可 np.savez 的字典（不含 Python 对象）。"""
        return {
            "voxel_size": np.float64(self.voxel_size),
            "feature_dim": np.int64(self.feature_dim),
            "mode": np.str_(self.mode),
            "origin": self.origin,
            "keys": self.keys,
            "features": self.features,
            "counts": self.counts,
            "weights": self.weights,
            "labels": np.array(
                [v.most_common(1)[0][0] if v else "" for v in self._label_votes],
                dtype=object,
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict, *, mode: Optional[str] = None) -> "VoxelGrid":
        """从 `to_dict()` 的结果重建（用于加载已建好的地图）。"""
        grid = cls(
            voxel_size=float(data["voxel_size"]),
            feature_dim=int(data["feature_dim"]),
            mode=mode or str(data.get("mode", "mean")),
            origin=data.get("origin"),
        )
        keys = np.asarray(data["keys"], dtype=np.int64).reshape(-1, 3)
        feats = np.asarray(data["features"], dtype=np.float32)
        counts = np.asarray(data["counts"], dtype=np.int64).reshape(-1)
        labels = data.get("labels")

        for i in range(keys.shape[0]):
            key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            grid._index[key] = i
            grid._keys.append(key)
            grid._feat_sum.append(feats[i].copy())
            grid._feat_max.append(feats[i].copy())
            grid._weight_sum.append(float(counts[i]))
            grid._counts.append(int(counts[i]))
            votes: Counter = Counter()
            if labels is not None and i < len(labels):
                lab = str(labels[i])
                if lab:
                    votes[lab] += int(counts[i])
            grid._label_votes.append(votes)
        grid.total_points = int(counts.sum()) if counts.size else 0
        return grid

    def __len__(self) -> int:
        return self.num_voxels

    def __repr__(self) -> str:
        return (
            f"VoxelGrid(voxels={self.num_voxels}, voxel_size={self.voxel_size}, "
            f"dim={self.feature_dim}, mode={self.mode!r}, points={self.total_points})"
        )


# ==========================================================================
# 聚类（把体素/点聚成"物体"）
# ==========================================================================
def cluster_points_dbscan(
    points: np.ndarray,
    eps: float = 0.08,
    min_samples: int = 3,
    *,
    fallback: bool = True,
) -> np.ndarray:
    """DBSCAN 聚类，返回 (N,) 的标签数组（-1 表示噪声）。

    优先用 sklearn（快且经过验证）；没有 sklearn 时回退到内置的
    网格加速实现（`fallback=True`），保证核心功能永远可用。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if pts.shape[0] < min_samples:
        return np.full(pts.shape[0], -1, dtype=np.int64)

    try:
        from sklearn.cluster import DBSCAN  # noqa: PLC0415

        labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(pts)
        return labels.astype(np.int64)
    except Exception as exc:  # pragma: no cover - 仅在无 sklearn 时触发
        if not fallback:
            raise
        logger.debug(f"sklearn DBSCAN 不可用（{exc}），回退到内置实现")
        return _dbscan_fallback(pts, eps=eps, min_samples=min_samples)


def _dbscan_fallback(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """内置 DBSCAN（用体素做邻域加速，复杂度接近 O(N)）。

    仅作为 sklearn 缺失时的兜底，精度与 sklearn 一致，速度略慢。
    """
    n = points.shape[0]
    cell = max(float(eps), 1e-6)
    keys = np.floor(points / cell).astype(np.int64)

    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(n):
        buckets.setdefault((int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])), []).append(i)

    eps2 = float(eps) ** 2
    labels = np.full(n, -1, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def neighbors(idx: int) -> List[int]:
        base = keys[idx]
        out: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    bucket = buckets.get((int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)))
                    if not bucket:
                        continue
                    for j in bucket:
                        diff = points[j] - points[idx]
                        if float(diff @ diff) <= eps2:
                            out.append(j)
        return out

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neigh = neighbors(i)
        if len(neigh) < min_samples:
            labels[i] = -1
            continue
        cluster_id += 1
        labels[i] = cluster_id
        seeds = list(neigh)
        k = 0
        while k < len(seeds):
            j = seeds[k]
            k += 1
            if not visited[j]:
                visited[j] = True
                neigh_j = neighbors(j)
                if len(neigh_j) >= min_samples:
                    seeds.extend(neigh_j)
            if labels[j] == -1:
                labels[j] = cluster_id
        # 移除噪声标记的边界点
    return labels
