"""体素 / 3D 特征场 / 语义地图测试。"""

from __future__ import annotations

import numpy as np
import pytest

from roboground.geometry.voxel import (
    VoxelGrid,
    aggregate_features_by_voxel,
    cluster_points_dbscan,
    voxel_centers,
    voxel_keys,
)


# ==========================================================================
# 体素化
# ==========================================================================
def test_voxel_keys_basic():
    pts = np.array([[0.01, 0.01, 0.01], [0.05, 0.05, 0.05], [0.11, 0.0, 0.0]])
    keys = voxel_keys(pts, 0.1)
    assert keys[0].tolist() == [0, 0, 0]
    assert keys[1].tolist() == [0, 0, 0]        # 同一格
    assert keys[2].tolist() == [1, 0, 0]        # 下一格


def test_voxel_keys_negative_coordinates():
    keys = voxel_keys(np.array([[-0.05, 0.0, 0.0]]), 0.1)
    assert keys[0].tolist() == [-1, 0, 0]       # floor 语义（不是截断向零）


def test_voxel_keys_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        voxel_keys(np.zeros((1, 3)), 0.0)


def test_voxel_centers_are_inside_cells():
    keys = np.array([[0, 0, 0], [1, 2, 3]])
    centers = voxel_centers(keys, 0.5)
    # center = (key + 0.5) * voxel_size
    assert np.allclose(centers[0], 0.25)
    assert np.allclose(centers[1], [0.75, 1.25, 1.75])


# ==========================================================================
# 特征聚合
# ==========================================================================
def test_aggregate_features_mean():
    pts = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.5, 0.0, 0.0]])
    feats = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 9.0]], dtype=np.float32)
    centers, agg, counts = aggregate_features_by_voxel(pts, feats, 0.1, mode="mean")

    assert centers.shape[0] == 2
    assert counts.tolist() == [2, 1]
    assert np.allclose(agg[0], [2.0, 0.0])      # 同一格内取均值


def test_aggregate_features_max():
    pts = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    feats = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
    _, agg, _ = aggregate_features_by_voxel(pts, feats, 0.1, mode="max")
    assert np.allclose(agg[0], [3.0, 5.0])      # 逐维最大


def test_aggregate_features_confidence_weighted():
    pts = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    feats = np.array([[1.0], [3.0]], dtype=np.float32)
    weights = np.array([3.0, 1.0], dtype=np.float32)
    _, agg, _ = aggregate_features_by_voxel(
        pts, feats, 0.1, mode="confidence_weighted", weights=weights
    )
    assert abs(float(agg[0, 0]) - (1.0 * 3 + 3.0 * 1) / 4.0) < 1e-6


def test_aggregate_rejects_bad_mode():
    with pytest.raises(ValueError):
        aggregate_features_by_voxel(np.zeros((1, 3)), np.zeros((1, 2)), 0.1, mode="nope")


def test_aggregate_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        aggregate_features_by_voxel(np.zeros((3, 3)), np.zeros((2, 4)), 0.1)


def test_aggregate_empty_input():
    centers, agg, counts = aggregate_features_by_voxel(
        np.zeros((0, 3)), np.zeros((0, 5), dtype=np.float32), 0.1
    )
    assert centers.shape == (0, 3)
    assert agg.shape == (0, 5)
    assert counts.shape == (0,)


# ==========================================================================
# 增量体素网格
# ==========================================================================
def test_voxel_grid_add_and_merge():
    grid = VoxelGrid(voxel_size=0.1, feature_dim=2)
    pts = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])

    added = grid.add(pts, np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32), labels=["cup", "cup"])
    assert added == 1
    assert grid.num_voxels == 1

    # 第二次同位置加入不同特征 → 特征应被平均（多视角融合）
    grid.add(pts, np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32), labels=["cup", "cup"])
    assert grid.num_voxels == 1
    assert np.allclose(grid.features[0], [0.5, 0.5], atol=1e-6)
    assert grid.counts[0] == 4
    assert grid.top_labels()[0][0][0] == "cup"


def test_voxel_grid_feature_dim_is_checked():
    grid = VoxelGrid(voxel_size=0.1, feature_dim=2)
    grid.add(np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 2.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="维度不匹配"):
        grid.add(np.array([[0.5, 0.0, 0.0]]), np.array([[1.0, 2.0, 3.0]], dtype=np.float32))


def test_voxel_grid_empty_add_is_noop():
    grid = VoxelGrid(voxel_size=0.1, feature_dim=3)
    assert grid.add(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.float32)) == 0
    assert grid.num_voxels == 0
    assert grid.centers.shape == (0, 3)
    assert grid.features.shape == (0, 3)


def test_voxel_grid_serialization_roundtrip():
    grid = VoxelGrid(voxel_size=0.2, feature_dim=2, mode="mean")
    grid.add(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
             np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
             labels=["cup", "table"], frame_id="f1")

    restored = VoxelGrid.from_dict(grid.to_dict())
    assert restored.num_voxels == grid.num_voxels
    assert np.allclose(restored.centers, grid.centers)
    assert np.allclose(restored.features, grid.features)
    assert restored.counts.tolist() == grid.counts.tolist()


def test_voxel_grid_max_mode():
    grid = VoxelGrid(voxel_size=0.1, feature_dim=2, mode="max")
    pts = np.array([[0.01, 0, 0], [0.02, 0, 0]])
    grid.add(pts, np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32))
    assert np.allclose(grid.features[0], [3.0, 5.0])


def test_voxel_grid_tracks_frames():
    grid = VoxelGrid(voxel_size=0.1, feature_dim=1)
    grid.add(np.array([[0.0, 0.0, 0.0]]), np.array([[1.0]], dtype=np.float32), frame_id="a")
    grid.add(np.array([[1.0, 0.0, 0.0]]), np.array([[1.0]], dtype=np.float32), frame_id="b")
    assert grid.frames_seen == ["a", "b"]


# ==========================================================================
# 聚类
# ==========================================================================
def test_cluster_dbscan_two_well_separated_groups():
    rng = np.random.default_rng(0)
    a = rng.normal(scale=0.01, size=(30, 3)) + np.array([0.0, 0.0, 0.0])
    b = rng.normal(scale=0.01, size=(30, 3)) + np.array([2.0, 0.0, 0.0])
    labels = cluster_points_dbscan(np.vstack([a, b]), eps=0.1, min_samples=3)

    uniq = set(labels.tolist()) - {-1}
    assert len(uniq) == 2                      # 分成两簇
    assert len(set(labels[:30].tolist())) == 1  # 第一组同簇
    assert len(set(labels[30:].tolist())) == 1


def test_cluster_dbscan_empty():
    labels = cluster_points_dbscan(np.zeros((0, 3)))
    assert labels.shape == (0,)


def test_cluster_dbscan_too_few_points():
    labels = cluster_points_dbscan(np.zeros((2, 3)), min_samples=5)
    assert np.all(labels == -1)
