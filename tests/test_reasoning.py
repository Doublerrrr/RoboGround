"""空间关系与规则引擎测试。"""

from __future__ import annotations

import numpy as np
import pytest

from roboground.eval.metrics import box3d_iou
from roboground.mapping import MapBuilder
from roboground.reasoning.rule_engine import RuleEngine, parse_intent
from roboground.reasoning.spatial_relations import (
    RELATION_NAMES,
    bbox_gap,
    center_distance,
    compute_all_relations,
    compute_relation,
    containment_ratio,
    dominant_relation,
)


# ==========================================================================
# 基础度量
# ==========================================================================
def test_center_distance(obs_factory):
    a = obs_factory("cup", (0, 0, 0), (0.2, 0.2, 0.2))
    b = obs_factory("cup", (3, 4, 0), (0.2, 0.2, 0.2))
    # 用物体级对象来测距离（Observation 的 centroid 是点云均值）
    from roboground.mapping.semantic_map import SemanticObject

    oa = SemanticObject(0, "a", np.array([0.0, 0, 0]), np.array([-0.1, -0.1, -0.1]),
                        np.array([0.1, 0.1, 0.1]), np.zeros(3))
    ob = SemanticObject(1, "b", np.array([3.0, 4, 0]), np.array([2.9, 3.9, -0.1]),
                        np.array([3.1, 4.1, 0.1]), np.zeros(3))
    assert abs(center_distance(oa, ob) - 5.0) < 1e-9


def _obj(obj_id, label, center, size):
    from roboground.mapping.semantic_map import SemanticObject

    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    return SemanticObject(
        obj_id=obj_id, label=label, center=center,
        bbox_min=center - size / 2.0, bbox_max=center + size / 2.0,
        feature=np.zeros(3),
    )


def test_bbox_gap_overlapping_is_zero():
    a = _obj(0, "a", (0, 0, 0), (1, 1, 1))
    b = _obj(1, "b", (0.2, 0, 0), (1, 1, 1))
    assert bbox_gap(a, b) == 0.0


def test_bbox_gap_separated():
    a = _obj(0, "a", (0, 0, 0), (1, 1, 1))
    b = _obj(1, "b", (2, 0, 0), (1, 1, 1))
    # 表面间隙 = (2 - 0.5) - 0.5 = 1.0
    assert abs(bbox_gap(a, b) - 1.0) < 1e-9


def test_containment_ratio():
    inner = _obj(0, "cup", (0, 0, 0), (0.1, 0.1, 0.1))
    outer = _obj(1, "table", (0, 0, 0), (2.0, 2.0, 2.0))
    assert containment_ratio(inner, outer) == 1.0
    assert containment_ratio(outer, inner) < 0.5


# ==========================================================================
# 主导关系
# ==========================================================================
def test_relation_above():
    a = _obj(0, "cup", (0, 0, 1.0), (0.2, 0.2, 0.2))
    b = _obj(1, "table", (0, 0, 0.2), (0.6, 0.6, 0.4))
    rel, ev = dominant_relation(a, b)
    assert rel == "above"
    assert ev["dz"] > 0


def test_relation_below():
    a = _obj(0, "cup", (0, 0, 0.1), (0.2, 0.2, 0.2))
    b = _obj(1, "shelf", (0, 0, 1.5), (0.6, 0.6, 0.4))
    rel, _ = dominant_relation(a, b)
    assert rel == "below"


def test_relation_left_right():
    a = _obj(0, "cup", (1.0, 0, 0), (0.2, 0.2, 0.2))
    b = _obj(1, "cup", (0.0, 0, 0), (0.2, 0.2, 0.2))
    rel, _ = dominant_relation(a, b)
    assert rel == "right_of"

    rel2, _ = dominant_relation(b, a)
    assert rel2 == "left_of"


def test_relation_front_behind():
    a = _obj(0, "cup", (0, 2.0, 0), (0.2, 0.2, 0.2))
    b = _obj(1, "cup", (0, 0.0, 0), (0.2, 0.2, 0.2))
    rel, _ = dominant_relation(a, b)
    assert rel == "in_front_of"
    assert dominant_relation(b, a)[0] == "behind"


def test_relation_inside():
    a = _obj(0, "cup", (0, 0, 0), (0.1, 0.1, 0.1))
    b = _obj(1, "box", (0, 0, 0), (1.0, 1.0, 1.0))
    assert dominant_relation(a, b)[0] == "inside"


def test_relation_near_when_horizontally_adjacent():
    """水平紧贴应判为 near，而不是勉强报一个方向。"""
    a = _obj(0, "cup", (0.0, 0, 0), (1.0, 1.0, 1.0))
    b = _obj(1, "box", (1.05, 0, 0), (1.0, 1.0, 1.0))
    rel, ev = dominant_relation(a, b)
    assert rel == "near"
    assert ev["bbox_gap"] < 0.1


def test_relation_above_not_overridden_by_near():
    """关键回归：杯子放在桌面上（间隙=0）必须仍是 above，不能变成 near。"""
    table = _obj(0, "table", (0, 0, 0.4), (1.0, 1.0, 0.8))
    cup = _obj(1, "cup", (0, 0, 0.86), (0.12, 0.12, 0.12))
    rel, ev = dominant_relation(cup, table)
    assert rel == "above", f"期望 above，实际 {rel}（gap={ev['bbox_gap']}）"


def test_relation_names_are_complete():
    for key in ("above", "below", "left_of", "right_of", "in_front_of",
                "behind", "inside", "near", "far", "overlapping", "unknown"):
        assert key in RELATION_NAMES


def test_compute_relation_returns_object():
    a = _obj(0, "cup", (0, 0, 0.9), (0.2, 0.2, 0.2))
    b = _obj(1, "table", (0, 0, 0.2), (1.0, 1.0, 0.4))
    rel = compute_relation(a, b)
    assert rel.subject == "cup" and rel.object == "table"
    assert rel.relation == "above"
    assert rel.distance > 0
    assert "bbox_gap" in rel.evidence
    assert "m" in rel.to_text()


def test_compute_all_relations_pairs():
    objs = [_obj(i, f"o{i}", (i * 0.5, 0, 0), (0.2, 0.2, 0.2)) for i in range(3)]
    rels = compute_all_relations(objs, max_distance=3.0)
    # 3 个物体 → 3 对 → 每对有向关系 → 6 条
    assert len(rels) == 6


def test_compute_all_relations_respects_max_distance():
    objs = [_obj(0, "a", (0, 0, 0), (0.2, 0.2, 0.2)),
            _obj(1, "b", (10, 0, 0), (0.2, 0.2, 0.2))]
    assert compute_all_relations(objs, max_distance=3.0) == []


# ==========================================================================
# 3D IoU
# ==========================================================================
def test_box3d_iou_identical():
    box = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 0.5, 0.0])
    assert abs(box3d_iou(box, box) - 1.0) < 1e-6


def test_box3d_iou_disjoint():
    a = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    b = np.array([5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    assert box3d_iou(a, b) == 0.0


def test_box3d_iou_half_overlap():
    """沿 x 平移半个边长：交集体积为一半 → IoU = 1/3。"""
    a = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    b = np.array([0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    iou = box3d_iou(a, b)
    assert abs(iou - (0.5 / 1.5)) < 1e-6


def test_box3d_iou_rotation_changes_result():
    """yaw 旋转后 IoU 必须变化（说明没有被简化成轴对齐）。"""
    a = np.array([0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.0])
    b0 = np.array([0.0, 0.0, 0.0, 1.0, 0.2, 1.0, 0.0])
    b90 = np.array([0.0, 0.0, 0.0, 1.0, 0.2, 1.0, np.pi / 2])
    assert box3d_iou(a, b0) > box3d_iou(a, b90)


def test_box3d_iou_degenerate_input_is_safe():
    assert box3d_iou(np.zeros(4), np.zeros(4)) == 0.0


# ==========================================================================
# 意图解析
# ==========================================================================
LABELS = ["cup", "table", "chair", "desk", "monitor"]


@pytest.mark.parametrize("query,kind", [
    ("杯子在哪", "locate"),
    ("where is the cup", "locate"),
    ("杯子离桌子多远", "distance"),
    ("桌子上有什么", "list_on"),
    ("有几个椅子", "count"),
    ("离我最近的椅子", "nearest"),
    ("杯子和桌子的距离", "distance"),
    ("描述一下场景", "describe"),
])
def test_parse_intent_kinds(query, kind):
    assert parse_intent(query, LABELS).kind == kind


def test_parse_intent_resolves_labels():
    intent = parse_intent("杯子离桌子多远", LABELS)
    assert intent.subject == "cup"
    assert intent.object_ == "table"


def test_parse_intent_mentions_object_not_in_map():
    """地图里没有的物体也要能被识别出来（否则无法回答"没找到"）。"""
    intent = parse_intent("冰箱在哪", LABELS)
    assert intent.kind == "locate"
    assert intent.mentions                      # 召回到了
    assert intent.matched_labels == []          # 但地图里没有


def test_parse_intent_detects_relation_term():
    intent = parse_intent("杯子在桌子上面吗", LABELS)
    assert intent.relation == "above"
    assert intent.kind == "relation"


def test_parse_intent_empty_query():
    assert parse_intent("", LABELS).kind == "describe"


# ==========================================================================
# 规则引擎（在真实建图产物上跑）
# ==========================================================================
def _build_map(cfg, obs_factory):
    obs = [
        obs_factory("table", (0.0, 1.0, 0.35), (1.0, 1.0, 0.7), feature=np.array([1, 0, 0, 0], np.float32)),
        obs_factory("cup", (0.1, 1.0, 0.78), (0.12, 0.12, 0.12), feature=np.array([1, 0, 0, 0], np.float32)),
        obs_factory("chair", (-1.2, 1.0, 0.4), (0.5, 0.5, 0.9), feature=np.array([0, 1, 0, 0], np.float32)),
        obs_factory("monitor", (0.0, 1.6, 0.9), (0.5, 0.1, 0.35), feature=np.array([0, 0, 1, 0], np.float32)),
    ]
    builder = MapBuilder(cfg)
    return builder.build_from_observations(obs, feature_dim=4)


def test_rule_engine_locate(cfg, obs_factory, quiet):
    smap = _build_map(cfg, obs_factory)
    engine = RuleEngine(smap, cfg=cfg)
    res = engine.answer("杯子在哪")
    assert res.targets
    assert res.targets[0].label == "cup"
    assert "m" in res.answer


def test_rule_engine_not_found(cfg, obs_factory, quiet):
    smap = _build_map(cfg, obs_factory)
    engine = RuleEngine(smap, cfg=cfg)
    res = engine.answer("冰箱在哪")
    assert res.targets == []
    assert "没有" in res.answer
    assert res.confidence == 0.0


def test_rule_engine_distance(cfg, obs_factory, quiet):
    smap = _build_map(cfg, obs_factory)
    engine = RuleEngine(smap, cfg=cfg)
    res = engine.answer("杯子离桌子多远")
    assert res.distances
    value = next(iter(res.distances.values()))
    assert value >= 0.0
    assert "m" in res.answer


def test_rule_engine_relation(cfg, obs_factory, quiet):
    smap = _build_map(cfg, obs_factory)
    engine = RuleEngine(smap, cfg=cfg)
    res = engine.answer("杯子在桌子上面吗")
    assert res.relations
    assert res.debug["intent"]["kind"] == "relation"


def test_rule_engine_list_on(cfg, obs_factory, quiet):
    smap = _build_map(cfg, obs_factory)
    engine = RuleEngine(smap, cfg=cfg)
    res = engine.answer("桌子上有什么")
    assert "table" in res.answer or len(res.targets) >= 1


def test_rule_engine_count(cfg, obs_factory, quiet):
    smap = _build_map(cfg, obs_factory)
    engine = RuleEngine(smap, cfg=cfg)
    res = engine.answer("有几个椅子")
    assert "1" in res.answer


def test_rule_engine_nearest(cfg, obs_factory, quiet):
    smap = _build_map(cfg, obs_factory)
    engine = RuleEngine(smap, cfg=cfg)
    res = engine.answer("离我最近的椅子")
    assert res.targets and res.targets[0].label == "chair"


def test_rule_engine_describe(cfg, obs_factory, quiet):
    smap = _build_map(cfg, obs_factory)
    engine = RuleEngine(smap, cfg=cfg)
    res = engine.answer("描述一下场景")
    assert "4" in res.answer or "物体" in res.answer


def test_rule_engine_empty_map_is_safe(cfg, quiet):
    from roboground.mapping.semantic_map import SemanticMap
    from roboground.geometry.voxel import VoxelGrid

    smap = SemanticMap(VoxelGrid(voxel_size=0.1, feature_dim=4))
    engine = RuleEngine(smap, cfg=cfg)
    for q in ("杯子在哪", "桌子上有什么", "描述一下场景", "有几个椅子"):
        res = engine.answer(q)
        assert isinstance(res.answer, str) and res.answer


def test_rule_engine_all_relations(cfg, obs_factory, quiet):
    smap = _build_map(cfg, obs_factory)
    engine = RuleEngine(smap, cfg=cfg)
    rels = engine.all_relations(max_distance=5.0)
    assert len(rels) > 0
    assert all(isinstance(r.distance, float) for r in rels)
