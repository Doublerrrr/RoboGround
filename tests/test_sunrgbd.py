"""SUN RGB-D 真实数据测试（标记 `slow`：需要本机有数据集，默认不跑）。

运行方式::

    pytest -m slow tests/test_sunrgbd.py -v

这些测试校验的是"真实数据上的坐标系与几何是否正确" —— 也就是本项目
最容易出错的地方（`CameraPose.from_sunrgbd` 的约定、`coeffs` 是全边长等）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roboground.data.sunrgbd import (
    DEPTH_SCALE,
    STANDARD_10,
    SUNRGBDDataset,
    build_scene_index,
    canon_class,
    load_scene_index,
    load_sunrgbd_scene,
)
from roboground.geometry.projection import project_points_to_image

pytestmark = pytest.mark.slow

INDEX_PATH = Path(r"G:\RoboGround\data\cache\sunrgbd_index.npz")
RAW_ROOT = Path(r"G:\sunrgbd_raw")


def _skip_if_missing():
    if not RAW_ROOT.exists():
        pytest.skip(f"SUN RGB-D 原始数据不存在：{RAW_ROOT}")


@pytest.fixture(scope="module")
def index():
    _skip_if_missing()
    if not INDEX_PATH.exists():
        pytest.skip(
            f"场景索引不存在：{INDEX_PATH}；"
            "请先运行 python scripts/02_build_sunrgbd_index.py"
        )
    return load_scene_index(str(INDEX_PATH))


# ==========================================================================
# 类别名规范化（纯逻辑，无需数据）
# ==========================================================================
@pytest.mark.parametrize("raw,expected", [
    ("night_stand", "nightstand"),
    ("dresser", "cabinet"),
    ("paper_bag", "bag"),
    ("tv", "monitor"),
    ("computer", "laptop"),
    ("chair", "chair"),
    ("Unknown Thing", "unknown_thing"),
])
def test_canon_class(raw, expected):
    assert canon_class(raw) == expected


def test_standard_10_list():
    assert len(STANDARD_10) == 10
    assert "bed" in STANDARD_10 and "table" in STANDARD_10


# ==========================================================================
# 索引
# ==========================================================================
def test_index_has_expected_fields(index):
    for key in ("sequences", "K", "Rtilt", "box_flat", "box_offset", "label_flat"):
        assert key in index, f"索引缺少字段 {key}"


def test_index_shapes_consistent(index):
    n = len(index["sequences"])
    assert n > 0
    assert index["K"].shape == (n, 3, 3)
    assert index["Rtilt"].shape == (n, 3, 3)
    assert index["box_offset"].shape == (n + 1,)
    assert index["box_offset"][0] == 0
    assert index["box_offset"][-1] == index["box_flat"].shape[0] == len(index["label_flat"])


def test_index_has_many_scenes_and_boxes(index):
    assert len(index["sequences"]) > 1000, "索引场景数偏少，可能只扫了部分数据"
    assert index["box_flat"].shape[0] > 10000, "GT 框总数偏少"


# ==========================================================================
# 场景加载
# ==========================================================================
def test_load_scene_basic(index):
    scene = None
    for i in range(50):
        scene = load_sunrgbd_scene(index, i, max_depth=8.0)
        if scene is not None and scene.boxes_3d.shape[0] > 0:
            break
    assert scene is not None

    assert scene.color.ndim == 3 and scene.color.shape[2] == 3
    assert scene.depth_m.shape == scene.color.shape[:2]
    assert scene.intrinsics.fx > 0 and scene.intrinsics.fy > 0
    assert scene.intrinsics.width == scene.color.shape[1]


def test_depth_scale_is_10000(index):
    """深度必须是 raw/10000（写成 /1000 会让整个点云尺度错 10 倍）。"""
    assert DEPTH_SCALE == 10000.0
    scene = load_sunrgbd_scene(index, 0, max_depth=8.0)
    assert scene is not None
    valid = scene.depth_m[scene.depth_m > 0]
    assert valid.size > 0
    # 室内场景的合理深度范围
    assert valid.min() > 0.1
    assert valid.max() <= 8.0


def test_gt_boxes_have_plausible_sizes(index):
    """`coeffs` 是全边长 —— 床头柜 ~0.5m、床 ~2m，若按半边长解释会大一倍。"""
    sizes = []
    for i in range(200):
        scene = load_sunrgbd_scene(index, i, max_depth=8.0)
        if scene is None or scene.boxes_3d.shape[0] == 0:
            continue
        sizes.append(scene.boxes_3d[:, 3:6])
    assert sizes, "没有取到任何 GT 框"
    all_sizes = np.concatenate(sizes, axis=0)
    diag = np.linalg.norm(all_sizes, axis=1)
    # 室内物体的对角线尺度中位数应在 0.3~4m 之间
    assert 0.3 < float(np.median(diag)) < 4.0, f"尺度中位数 {np.median(diag):.2f}m 不合理"


def test_box_centers_project_into_image(index):
    """坐标系标定的核心断言：GT 框中心应投影到画面内且深度为正。"""
    total = 0
    valid = 0
    for i in range(60):
        scene = load_sunrgbd_scene(index, i, max_depth=8.0)
        if scene is None or scene.boxes_3d.shape[0] == 0:
            continue
        uv, z, ok = project_points_to_image(
            scene.boxes_3d[:, :3], scene.pose, scene.intrinsics, *scene.shape[::-1]
        )
        total += int(ok.size)
        valid += int(ok.sum())

    assert total > 0
    ratio = valid / total
    assert ratio > 0.6, f"GT 框中心投影有效率只有 {ratio:.2%}，坐标系可能标定错了"


def test_scene_to_frame_carries_gt(index):
    scene = None
    for i in range(50):
        s = load_sunrgbd_scene(index, i, max_depth=8.0)
        if s is not None and s.boxes_3d.shape[0] > 0:
            scene = s
            break
    assert scene is not None

    frame = scene.to_frame()
    assert "boxes_3d" in frame.meta
    assert "labels" in frame.meta
    assert frame.meta["boxes_3d"].shape[0] == len(frame.meta["labels"])
    assert frame.width == scene.color.shape[1]


def test_filter_classes(index):
    scene = load_sunrgbd_scene(index, 0, max_depth=8.0)
    assert scene is not None
    filtered = scene.filter_classes(STANDARD_10)
    assert len(filtered.labels) <= len(scene.labels)
    # 注意：要拿"规范化之后"的集合来比 —— STANDARD_10 用的是 SUN RGB-D 的原始名
    # （如 dresser / night_stand），而 canon_class 会把它们映射成 cabinet / nightstand。
    keep = {canon_class(x) for x in STANDARD_10}
    assert all(canon_class(lab) in keep for lab in filtered.labels)


def test_resize_keeps_intrinsics_in_sync(index):
    scene = load_sunrgbd_scene(index, 0, max_depth=8.0, resize=(160, 120))
    assert scene is not None
    assert scene.color.shape[:2] == (120, 160)
    assert scene.intrinsics.width == 160
    assert scene.intrinsics.height == 120
    # 内参与图像尺寸一致 → 反投影不会抛异常
    from roboground.geometry.projection import depth_to_points_camera

    pts = depth_to_points_camera(scene.depth_m, scene.intrinsics, max_depth=8.0)
    assert pts.shape[1] == 3


# ==========================================================================
# 数据集封装
# ==========================================================================
def test_dataset_len_and_sample(index):
    ds = SUNRGBDDataset(index, require_gt=True)
    assert len(ds) > 100
    scenes = ds.sample(3, seed=0)
    assert len(scenes) == 3
    assert all(s.color.ndim == 3 for s in scenes)


def test_dataset_label_histogram(index):
    ds = SUNRGBDDataset(index, require_gt=True)
    hist = ds.label_histogram(top=10)
    assert hist
    assert "chair" in hist or "table" in hist


def test_dataset_getitem_out_of_range(index):
    ds = SUNRGBDDataset(index)
    with pytest.raises(IndexError):
        _ = ds[len(ds) + 10]


# ==========================================================================
# 端到端（真实数据建图）
# ==========================================================================
def test_real_data_end_to_end(cfg, quiet, index):
    """真实 SUN RGB-D 帧 → 语义地图 → 查询。"""
    from roboground.mapping import MapBuilder

    cfg.set("perception.prompts", ["chair", "table", "desk", "monitor", "door",
                                   "window", "bookshelf", "trash can", "box", "bottle"])

    # 找一个框多、类别多样的场景，并缩放到较小分辨率以加速
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))
    scene = None
    for idx in np.argsort(-counts)[:40]:
        s = load_sunrgbd_scene(index, int(idx), max_depth=8.0, resize=(320, 240))
        if s is not None and len(set(s.labels)) >= 3 and s.boxes_3d.shape[0] >= 5:
            scene = s
            break
    if scene is None:
        pytest.skip("没找到合适的测试场景")

    smap = MapBuilder(cfg).build_from_frames([scene.to_frame()])
    assert smap.num_voxels > 0
    assert smap.num_objects > 0
    assert smap.meta["num_frames"] == 1

    label = smap.labels[0]
    hits = smap.query_text(label, top_k=1)
    assert hits, f"真实数据上查询 {label!r} 失败"


def test_auto_labeler_on_real_data(cfg, quiet, index):
    """自动化标注在真实数据上要产出可用标注，且采纳率不能是 0。"""
    from roboground.data.auto_label import AutoLabeler

    prompts = ["chair", "table", "desk", "monitor"]
    cfg.set("perception.prompts", prompts)

    # ⚠️ 必须挑**含这些类别**的场景：SUN RGB-D 里有大量只有 bed/cabinet 的场景，
    # 随便取前几个会让 stub 检测器（按 prompt 过滤）一个都检不出来。
    wanted = set(prompts)
    scenes = []
    for i in range(len(index["sequences"])):
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels_here = {str(x) for x in np.asarray(index["label_flat"][off0:off1]).ravel()}
        if not (labels_here & wanted):
            continue
        s = load_sunrgbd_scene(index, i, max_depth=8.0, resize=(320, 240))
        if s is not None and s.boxes_3d.shape[0] > 0:
            scenes.append(s)
        if len(scenes) >= 3:
            break
        if i > 3000:
            break

    if not scenes:
        pytest.skip("没有找到含目标类别的场景")

    labeler = AutoLabeler(cfg, prompts=prompts)
    report = labeler.label_frames([s.to_frame() for s in scenes])
    assert report["num_candidates"] > 0, (
        f"自动标注没产出任何候选（场景类别：{[sorted(set(s.labels)) for s in scenes]}）"
    )
    assert report["accept_rate"] > 0.0, "自动标注采纳率为 0，闸门可能过严"
