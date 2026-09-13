"""自动化标注（Stage 5 数据闭环）测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from roboground.data.auto_label import (
    PLAUSIBLE_SIZE_RANGES,
    AutoLabeler,
    LabelItem,
)
from roboground.data.synthetic import make_synthetic_sequence

PROMPTS = ["table", "chair", "cup", "box", "bottle", "sofa", "shelf",
           "monitor", "trash can", "lamp"]


@pytest.fixture
def labeler(cfg):
    cfg.set("perception.prompts", PROMPTS)
    return AutoLabeler(cfg, prompts=PROMPTS)


def test_label_frame_produces_items(labeler):
    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=4)
    items = labeler.label_frame(frames[0])
    assert len(items) > 0
    assert all(isinstance(it, LabelItem) for it in items)
    assert all(it.frame_id == frames[0].frame_id for it in items)


def test_label_frame_assigns_3d_boxes_to_accepted(labeler):
    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=4)
    items = labeler.label_frame(frames[0])
    accepted = [it for it in items if it.accepted]
    assert accepted
    for it in accepted:
        assert it.box3d is not None and it.box3d.shape == (7,)
        assert it.num_points >= labeler.min_points


def test_rejected_items_keep_reason(labeler):
    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=4)
    items = labeler.label_frame(frames[0])
    for it in items:
        if not it.accepted:
            assert it.reject_reason, "被拒绝的标注必须给出原因"
            assert it.accepted is False


def test_low_score_gate(cfg):
    """把分数阈值设到 1.1（不可能达到）→ 全部应因 low_score 被拒。"""
    cfg.set("perception.prompts", PROMPTS)
    labeler = AutoLabeler(cfg, prompts=PROMPTS)
    labeler.min_score = 1.1

    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=3)
    items = labeler.label_frame(frames[0])
    assert items
    assert all(not it.accepted for it in items)
    assert all(it.reject_reason == "low_score" for it in items)


def test_too_few_points_gate(cfg):
    """把最小点数设得极高 → 应全部因 too_few_points 被拒。"""
    cfg.set("perception.prompts", PROMPTS)
    labeler = AutoLabeler(cfg, prompts=PROMPTS)
    labeler.min_points = 10 ** 9

    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=3)
    items = labeler.label_frame(frames[0])
    assert items
    assert all(it.reject_reason == "too_few_points" for it in items)


def test_scale_gate_rejects_implausible(cfg):
    """把类别尺寸范围压到极小 → 正常物体应被判为尺度不合理。"""
    cfg.set("perception.prompts", PROMPTS)
    labeler = AutoLabeler(cfg, prompts=PROMPTS)
    labeler.check_scale = True

    import roboground.data.auto_label as al

    original = dict(al.PLAUSIBLE_SIZE_RANGES)
    try:
        al.PLAUSIBLE_SIZE_RANGES.clear()
        al.PLAUSIBLE_SIZE_RANGES["default"] = (0.001, 0.002)   # 荒谬的小范围
        for key in PROMPTS:
            al.PLAUSIBLE_SIZE_RANGES[key] = (0.001, 0.002)

        frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=3)
        items = labeler.label_frame(frames[0])
        checked = [it for it in items if it.reject_reason and it.reject_reason.startswith("implausible")]
        assert checked, "尺度闸门没有生效"
    finally:
        al.PLAUSIBLE_SIZE_RANGES.clear()
        al.PLAUSIBLE_SIZE_RANGES.update(original)


def test_label_frames_report_without_output(labeler):
    frames = make_synthetic_sequence(seed=7, num_frames=3, width=160, height=120, num_objects=4)
    report = labeler.label_frames(frames)

    assert report["num_frames"] == 3
    assert report["num_candidates"] > 0
    assert 0.0 <= report["accept_rate"] <= 1.0
    assert report["num_accepted"] + report["num_rejected"] == report["num_candidates"]
    assert "class_distribution" in report
    assert report["elapsed_s"] >= 0


def test_label_frames_writes_three_files(labeler, tmp_path):
    frames = make_synthetic_sequence(seed=7, num_frames=2, width=160, height=120, num_objects=4)
    report = labeler.label_frames(frames, out_dir=str(tmp_path / "labels"))

    out = tmp_path / "labels"
    assert (out / "coco.json").exists()
    assert (out / "boxes3d.json").exists()
    assert (out / "review.json").exists()

    coco = json.loads((out / "coco.json").read_text(encoding="utf-8"))
    assert set(coco.keys()) >= {"images", "annotations", "categories"}
    assert len(coco["images"]) == len(frames)

    for ann in coco["annotations"]:
        assert len(ann["bbox"]) == 4
        assert ann["bbox"][2] >= 0 and ann["bbox"][3] >= 0
        assert ann["area"] >= 0
        assert ann["category_id"] >= 1

    boxes3d = json.loads((out / "boxes3d.json").read_text(encoding="utf-8"))
    for item in boxes3d:
        assert "box3d" in item and len(item["box3d"]) == 7


def test_cross_frame_consistency_flags_outliers(obs_factory):
    """同类物体中心离群的点应被标为可疑。"""
    items = []
    for i in range(6):
        it = LabelItem(
            frame_id=f"f{i}", label="chair", score=0.9,
            bbox=np.array([0.0, 0.0, 10.0, 10.0]), num_points=100,
        )
        center = np.array([0.0, 0.0, 0.0]) if i < 5 else np.array([5.0, 0.0, 0.0])
        it.box3d = np.concatenate([center, [0.5, 0.5, 0.5], [0.0]]).astype(np.float32)
        items.append(it)

    suspects = AutoLabeler.cross_frame_consistency(items, max_deviation=0.5)
    assert len(suspects) == 1
    assert suspects[0]["frame_id"] == "f5"
    assert suspects[0]["reason"] == "cross_frame_inconsistent"


def test_cross_frame_consistency_ignores_small_groups():
    items = [LabelItem(frame_id="f0", label="cup", score=0.9,
                       bbox=np.zeros(4), num_points=10,
                       box3d=np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0]))]
    assert AutoLabeler.cross_frame_consistency(items) == []


def test_plausible_size_ranges_are_sane():
    """物理先验表本身要合理（不能出现"杯子 2 米"这种）。"""
    lo, hi = PLAUSIBLE_SIZE_RANGES["cup"]
    assert lo < hi < 0.5
    lo_t, hi_t = PLAUSIBLE_SIZE_RANGES["table"]
    assert lo_t < hi_t


def test_label_item_to_dict_serializable():
    it = LabelItem(frame_id="f0", label="cup", score=0.87,
                   bbox=np.array([1.0, 2.0, 3.0, 4.0]), num_points=42,
                   box3d=np.array([0.0, 1.0, 0.5, 0.1, 0.1, 0.1, 0.0]))
    d = it.to_dict()
    json.dumps(d)                # 必须可 JSON 序列化
    assert d["accepted"] is True
    assert d["score"] == 0.87
