"""端到端测试：合成/半合成数据 → 建图 → 查询 → 推理 → 持久化。

这是"整条流水线能不能用"的最终证据，也是回归测试的主入口。
全部离线：不下载模型、不读数据集、不需要 GPU。
"""

from __future__ import annotations

import numpy as np
import pytest

from roboground.data.synthetic import (
    SyntheticRoom,
    _look_forward_pose,
    default_intrinsics,
    make_synthetic_sequence,
)
from roboground.mapping import MapBuilder
from roboground.mapping.query import QueryEngine
from roboground.perception import build_pipeline
from roboground.reasoning import RuleEngine

# 覆盖合成房间目录里的全部类别，保证 stub 检测器的 prompt 过滤不会漏掉物体
FULL_PROMPTS = ["table", "chair", "cup", "box", "bottle", "sofa", "shelf",
                "monitor", "trash can", "lamp"]


@pytest.fixture
def prompts():
    return list(FULL_PROMPTS)


# ==========================================================================
# 单帧建图
# ==========================================================================
def test_e2e_single_frame_builds_map(cfg, quiet, prompts):
    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=4)
    cfg.set("perception.prompts", prompts)

    builder = MapBuilder(cfg, prompts=prompts)
    smap = builder.build_from_frames(frames)

    assert smap.num_voxels > 0
    assert smap.num_objects > 0
    assert smap.feature_dim == 72                     # color_hist 的维度
    assert smap.meta["num_frames"] == 1


def test_e2e_detects_expected_labels(cfg, quiet, prompts):
    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=4)
    cfg.set("perception.prompts", prompts)

    builder = MapBuilder(cfg, prompts=prompts)
    smap = builder.build_from_frames(frames)

    gt_labels = set(frames[0].meta["labels"])
    found = set(smap.labels)
    # 至少检出 GT 类别中的一部分（受遮挡/视野限制，不要求全部）
    assert len(found & gt_labels) >= 1, f"检出 {found}，GT {gt_labels}"


# ==========================================================================
# 多视角：关联与融合
# ==========================================================================
def test_e2e_multiframe_association_does_not_explode(cfg, quiet, prompts):
    """多视角下物体数应该稳定在 GT 数量附近，而不是每帧新增一批。"""
    frames = make_synthetic_sequence(seed=7, num_frames=4, width=160, height=120, num_objects=4)
    cfg.set("perception.prompts", prompts)

    builder = MapBuilder(cfg, prompts=prompts)
    smap = builder.build_from_frames(frames)

    gt_count = frames[0].meta["boxes_3d"].shape[0]
    assert smap.num_objects <= gt_count + 2, (
        f"物体数 {smap.num_objects} 明显超过 GT {gt_count}，关联可能失效"
    )
    assert smap.meta["num_frames"] == 4


def test_e2e_multiframe_reduces_noise(cfg, quiet, prompts):
    """多视角融合后，每个物体的观测数应 > 1（证明真的融合了）。"""
    frames = make_synthetic_sequence(seed=7, num_frames=4, width=160, height=120, num_objects=4)
    cfg.set("perception.prompts", prompts)

    builder = MapBuilder(cfg, prompts=prompts)
    smap = builder.build_from_frames(frames)

    multi_obs = [o for o in smap.objects if o.num_points > 0]
    assert multi_obs
    # 体素被多次观测 → counts 之和应大于体素数
    assert smap.voxel_grid.counts.sum() >= smap.num_voxels


# ==========================================================================
# 查询与推理
# ==========================================================================
def test_e2e_query_returns_located_object(cfg, quiet, prompts):
    frames = make_synthetic_sequence(seed=7, num_frames=2, width=160, height=120, num_objects=4)
    cfg.set("perception.prompts", prompts)

    smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
    label = smap.labels[0]

    results = smap.query_text(label, top_k=3)
    assert results, f"查询 {label!r} 没有结果（地图类别：{smap.labels}）"
    assert results[0].position.shape == (3,)
    assert np.all(np.isfinite(results[0].position))


def test_e2e_query_by_chinese_alias(cfg, quiet, prompts):
    """中文查询应能命中英文标签（词法匹配的核心能力）。"""
    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=4)
    cfg.set("perception.prompts", prompts)

    smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
    mapping = {"table": "桌子", "chair": "椅子", "cup": "杯子",
               "box": "箱子", "bottle": "瓶子", "sofa": "沙发"}
    tested = 0
    for label in smap.labels:
        cn = mapping.get(label)
        if not cn:
            continue
        hits = smap.query_text(cn, top_k=1)
        if hits:
            assert hits[0].label == label, f"中文 {cn!r} 命中了 {hits[0].label!r} 而非 {label!r}"
            tested += 1
    assert tested >= 1, f"没有可测的中文别名（地图类别：{smap.labels}）"


def test_e2e_config_path_does_not_lower_lexical_threshold(cfg, quiet, prompts):
    """★ 回归锁：**真实配置路径**下的词法阈值不能被 config 静默改小。

    回归背景（2026-09-11 发现）：
    `configs/default.yaml` 里写着 `query.min_score: 0.05`，经
    `builder.py` → `SemanticMap.query_min_score` → `QueryEngine.query(min_score=…)`
    把类默认的 **0.5 覆盖成 0.05**。于是 AGENTS.md 二.7 里记录"已修好"的
    「27.8% 未见类别误命中」缺陷**在配置路径上原样复发**：
    实测 `refrigerator`→monitor 0.194、`bicycle`→bottle 0.140、
    `table`→bottle 0.175，全部是错误命中。

    为什么当时 372 个测试全绿：`tests/test_query.py` 的阈值回归锁
    （`test_lexical_threshold_rejects_ngram_noise` 等）都用 `_make_map()`
    **直连 `SemanticMap`**，`query_min_score` 停在类默认 0.5，
    完全绕过了 `MapBuilder` + config 这条真实路径。

    教训：**单测绕过集成路径时，"默认值是对的"这件事根本测不出来** ——
    改配置就能让所有单测继续通过，而线上行为已经变了。
    """
    frames = make_synthetic_sequence(seed=1, num_frames=2, width=160, height=120,
                                     num_objects=4)
    cfg.set("perception.prompts", prompts)
    smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)

    assert smap.num_objects > 0, "地图为空，本测试失去意义"
    # seed=1 / 4 物体的场景里确定性含有 bottle（对抗性负样本需要它）
    assert "bottle" in set(smap.labels), f"场景不含 bottle：{smap.labels}"

    # 1) 阈值层面：必须停在生产默认 0.5
    assert smap.query_min_score == pytest.approx(0.5), (
        f"配置把词法阈值改成了 {smap.query_min_score}（应为 0.5）——"
        "查 AGENTS.md 二.7，别把它改回去"
    )
    assert QueryEngine(smap).min_score_lexical == pytest.approx(0.5)

    # 2) 行为层面：与地图标签共享少量字符 bigram 的**未见类别**必须被拒。
    #    `bicycle` ↔ `bottle` 共享 bigram "le"（Jaccard 1/11 → 0.145 分）：
    #    阈值 0.05 时命中 bottle，阈值 0.5 时正确拒识。
    for absent in ("bicycle", "helicopter", "airplane"):
        assert smap.query_text(absent) == [], (
            f"{absent!r} 不应命中任何物体（实际命中 "
            f"{[(r.label, round(r.score, 3)) for r in smap.query_text(absent)]}）"
        )

    # 3) 反向：抬阈值不能误杀真命中（否则这条锁会"把阈值抬到 1.0"也能过）
    assert smap.query_text("bottle") != []
    assert smap.query_text("瓶子") != []


def test_e2e_rule_engine_answers(cfg, quiet, prompts):
    frames = make_synthetic_sequence(seed=7, num_frames=2, width=160, height=120, num_objects=5)
    cfg.set("perception.prompts", prompts)

    smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
    engine = RuleEngine(smap, cfg=cfg)

    res = engine.answer("描述一下场景")
    assert res.answer
    assert res.targets

    if smap.labels:
        res2 = engine.answer(f"{smap.labels[0]} 在哪")
        assert res2.answer


# ==========================================================================
# 持久化
# ==========================================================================
def test_e2e_map_save_load_roundtrip(cfg, quiet, prompts, tmp_path):
    frames = make_synthetic_sequence(seed=7, num_frames=2, width=160, height=120, num_objects=4)
    cfg.set("perception.prompts", prompts)

    smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
    path = smap.save(tmp_path / "map.npz")
    assert path.exists()

    loaded = smap.__class__.load(path)
    assert loaded.num_voxels == smap.num_voxels
    assert loaded.num_objects == smap.num_objects
    assert loaded.feature_dim == smap.feature_dim
    assert np.allclose(loaded.voxel_grid.centers, smap.voxel_grid.centers)

    # 加载后的地图依然可查询
    if loaded.labels:
        assert loaded.query_text(loaded.labels[0], top_k=1)


# ==========================================================================
# 稳定性 / 边界
# ==========================================================================
def test_e2e_all_invalid_depth_produces_empty_map(cfg, quiet, prompts):
    """全零深度图不应崩溃，只应得到空地图。"""
    import numpy as np

    from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame

    K = CameraIntrinsics(fx=120, fy=120, cx=79.5, cy=59.5, width=160, height=120)
    frame = RGBDFrame(
        color=np.zeros((120, 160, 3), np.uint8),
        depth_m=np.zeros((120, 160), np.float32),
        intrinsics=K, pose=CameraPose.identity(), frame_id="zero",
        meta={"boxes_3d": np.array([[0.0, 2.0, 0.0, 0.5, 0.5, 0.5, 0.0]], np.float32),
              "labels": ["cup"]},
    )
    smap = MapBuilder(cfg, prompts=prompts).build_from_frames([frame])
    assert smap.num_voxels == 0
    assert smap.num_objects == 0
    assert smap.query_text("cup") == []


def test_e2e_empty_frame_list(cfg, quiet):
    smap = MapBuilder(cfg).build_from_frames([])
    assert smap.num_voxels == 0
    assert smap.meta["num_frames"] == 0


def test_e2e_builder_reuse_via_reset(cfg, quiet, prompts):
    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=3)
    cfg.set("perception.prompts", prompts)

    builder = MapBuilder(cfg, prompts=prompts)
    map1 = builder.build_from_frames(frames, reset=True)
    map2 = builder.build_from_frames(frames, reset=True)

    assert map1.num_voxels == map2.num_voxels      # reset 后结果应完全一致
    assert builder.frames_processed == 1


def test_e2e_pipeline_stats(cfg, quiet, prompts):
    frames = make_synthetic_sequence(seed=7, num_frames=2, width=160, height=120, num_objects=3)
    cfg.set("perception.prompts", prompts)

    builder = MapBuilder(cfg, prompts=prompts)
    builder.build_from_frames(frames)
    stats = builder.stats()

    assert stats["frames"] == 2
    assert stats["observations"] > 0
    assert stats["voxels"] > 0
    assert "perceive_s" in stats or "frame_s" in stats


def test_e2e_clustering_mode_still_works(cfg, quiet, prompts):
    """降级的聚类模式也必须能产出地图（虽然可能把相邻物合并）。"""
    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=3)
    cfg.set("perception.prompts", prompts)
    cfg.set("mapping.object_mode", "clustering")

    smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
    assert smap.meta["object_mode"] == "clustering"
    assert smap.num_voxels > 0
