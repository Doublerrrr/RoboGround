# -*- coding: utf-8 -*-
"""2D-3D-S 适配器的测试（**不需要那 13.8 GB 数据**）。

测试策略：把"从实测中得来的**约定**"全部钉成断言。
这些约定每一条都曾经是未知项，而且**每一条错了都会静默出错**：

  · 位姿矩阵形状与方向（官方文档说是 4×3，实测是 3×4 且 world→camera）
  · 文件名是单下划线（官方文档写双下划线）
  · 深度 `raw / 512` 且 **`raw == 65535` = 无数据**（不是"离 128 米"）
  · GT 的 `Bbox` 是轴对齐（yaw 恒 0）
  · MATLAB v7.3 的 char 数组要按码点解码

数据相关的事实（真实文件）已由 `scripts/32/33` 在真数据上验证过；
这里守的是**代码里的约定**，保证以后有人"顺手改一下"时会红。
"""
from __future__ import annotations

import numpy as np
import pytest

from roboground.data.stanford2d3d import (
    DEPTH_SCALE,
    FILENAME_RE,
    INVALID_RAW,
    Location,
    MODALITY,
    NameParts,
    filter_objects,
    intrinsics_from_json,
    load_semantic_labels,
    objects_as_boxes,
    parse_name,
    pose_from_json,
    semantic_class,
)
from roboground.types import CameraPose


# ==========================================================================
# 1) 文件名解析（官方文档写错了下划线数量）
# ==========================================================================
def test_parse_real_filename():
    """真实的文件名必须能解析出 uuid / 房间 / 帧号 / 模态。"""
    name = "camera_0004591bfdc749a88db196a5d8b345cb_office_6_frame_0_domain_rgb.png"
    p = parse_name(name)
    assert p is not None
    assert p.uuid == "0004591bfdc749a88db196a5d8b345cb"
    assert p.room == "office_6"
    assert p.frame == 0
    assert p.modality == "rgb"
    assert not p.is_panorama


def test_parse_panorama_filename():
    """`frame_equirectangular` 是全景，要能被识别成 frame=-1。"""
    name = ("camera_0004591bfdc749a88db196a5d8b345cb_office_6"
            "_frame_equirectangular_domain_depth.png")
    p = parse_name(name)
    assert p is not None and p.is_panorama and p.frame == -1


def test_regex_rejects_official_double_underscore_form():
    """★ 官方文档写的**双下划线**形式必须被**拒掉**，而不是"勉强解析成功"。

    实测真实文件是单下划线。若正则太松，官方那种写法会被解析成
    `room='_office_6'`、`modality='_rgb'` —— 房间名和模态都带上一个前导
    下划线，然后在下游才炸，排查成本极高。所以这里要求**边界处就失败**。
    """
    official = "camera_0004591bfdc749a88db196a5d8b345cb__office_6_frame_0_domain__rgb.png"
    got = parse_name(official)
    assert got is None, (
        f"双下划线（官方文档写法）被解析成了 {got} —— 实测数据是单下划线，"
        "说明正则被改松了，脏文件名会一路渗透到下游")


def test_parse_rejects_junk():
    for bad in ("", "rgb.png", "camera_short_room_frame_0_domain_rgb.png",
                "0004591bfdc749a88db196a5d8b345cb_d0_0.png"):
        assert parse_name(bad) is None


def test_modality_map_covers_used_domains():
    """模态映射必须与真实 `domain` 后缀一致（normal 的真实后缀是 normals）。"""
    assert MODALITY["normal"] == "normals"
    for key in ("rgb", "depth", "pose", "semantic"):
        assert key in MODALITY


# ==========================================================================
# 2) 位姿约定（★ 最重要）
# ==========================================================================
def _pose_json(R, t):
    return {"camera_rt_matrix": np.hstack([R, np.asarray(t).reshape(3, 1)]).tolist()}


def test_pose_is_3x4_and_world_to_camera():
    """★ `camera_rt_matrix` 是 `(3,4)=[R|t]`，且语义是 **world→camera**。

    实测依据：`C = −Rᵀ·t` 与 json 里的 `camera_location` 差 1.2 µm，
    而反向假设差 38 m。这个断言直接验证"我们按 world→camera 解释"。
    """
    # 构造一个已知位姿：相机在 C，绕 z 转 30°
    ang = np.radians(30.0)
    R = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                  [np.sin(ang), np.cos(ang), 0.0],
                  [0.0, 0.0, 1.0]])
    C = np.array([-17.839951, 20.299704, 1.397828])
    t = -R @ C                                  # world→camera 的平移

    pose = pose_from_json(_pose_json(R, t))
    assert pose.R.shape == (3, 3)
    # 由 poset 反推光心，应当回到 C
    C_back = -np.asarray(pose.R).T @ np.asarray(pose.t)
    assert np.allclose(C_back, C, atol=1e-9), \
        f"光心反推失败：{C_back} != {C} → 说明约定解释错了"

    # 语义验证：相机正前方（光轴 z_cam）在世界系里应指向 R.T @ [0,0,1]
    fwd_world = np.asarray(pose.R).T @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(fwd_world, R.T @ np.array([0.0, 0.0, 1.0]))


def test_pose_accepts_documented_4x3_shape_as_fallback():
    """官方文档说的 (4,3) 形状也要能解析（兜底），值必须与 (3,4) 一致。"""
    ang = np.radians(15.0)
    R = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                  [np.sin(ang), np.cos(ang), 0.0],
                  [0.0, 0.0, 1.0]])
    t = np.array([1.0, 2.0, 3.0])
    p34 = pose_from_json(_pose_json(R, t))
    p43 = pose_from_json({"camera_rt_matrix": np.vstack([R, t.reshape(1, 3)]).tolist()})
    assert np.allclose(p34.R, p43.R) and np.allclose(p34.t, p43.t)


def test_pose_rejects_bad_shape():
    with pytest.raises(ValueError):
        pose_from_json({"camera_rt_matrix": [[1.0, 0.0], [0.0, 1.0]]})


def test_intrinsics_from_json():
    obj = {"camera_k_matrix": [[993.8245, 0.0, 540.0],
                              [0.0, 993.8245, 540.0],
                              [0.0, 0.0, 1.0]]}
    K = intrinsics_from_json(obj)
    assert K.fx == pytest.approx(993.8245)
    assert K.cx == pytest.approx(540.0) and K.cy == pytest.approx(540.0)
    assert K.width == 1080 and K.height == 1080


# ==========================================================================
# 3) 深度：换算 + 无效值（★ 曾经被我误判为"无缺失"）
# ==========================================================================
def test_depth_scale_and_invalid_constant():
    """换算常数与无效标记必须与实测一致（512 / 65535）。"""
    assert DEPTH_SCALE == 512.0, "深度换算不是 512 —— 实测按 512 才得到合理的 0.6~3.1 m"
    assert INVALID_RAW == 65535, "无效标记不是 65535"
    # 65535 / 512 = 127.998 = 官方的量程上限 128 m，
    # 这正是"65535 是无效标记、不是'真的在 128 米处'"的证据。
    assert INVALID_RAW / DEPTH_SCALE == pytest.approx(128.0, abs=0.01)


def test_depth_semantics_documented_in_helpers():
    """`depth_m` 必须把 65535 归零 —— 用一个临时目录真实走一遍读取路径。"""
    from PIL import Image

    from roboground.data.stanford2d3d import Location

    tmp = pytest.importorskip("tempfile").mkdtemp()
    from pathlib import Path

    root = Path(tmp)
    (root / "data" / "depth").mkdir(parents=True)
    (root / "data" / "rgb").mkdir(parents=True)
    (root / "data" / "pose").mkdir(parents=True)

    uuid = "0" * 32
    base = f"camera_{uuid}_office_6_frame_0"
    raw = np.array([[512, 1024], [INVALID_RAW, 0]], dtype=np.uint16)
    Image.fromarray(raw).save(root / "data" / "depth" / f"{base}_domain_depth.png")

    loc = Location(uuid=uuid, room="office_6", root=root, frame_ids=[0])
    d = loc.depth_m(0)
    assert d is not None
    assert d[0, 0] == pytest.approx(1.0)          # 512/512
    assert d[0, 1] == pytest.approx(2.0)          # 1024/512
    assert d[1, 0] == 0.0, "65535 必须记为无效（0），不能变成 128 m"
    assert d[1, 1] == 0.0, "raw==0 也应视为无效"

    mask = loc.depth_valid_mask(0)
    assert mask.tolist() == [[True, True], [False, False]]


# ==========================================================================
# 4) GT：物体框与类别
# ==========================================================================
def test_objects_as_boxes_shape_and_zero_yaw():
    """★ `objects_as_boxes` 输出 `(N,7)` 且 **yaw 恒为 0**。

    2D-3D-S 给的是**轴对齐**包围盒，所以**不能用 yaw 有向框 IoU 去评它** ——
    这条断言把这个限制钉住，避免以后有人拿 3D IoU（有向）去评轴对齐 GT。
    """
    objs = [
        {"name": "chair_1", "cls": "chair",
         "bbox": np.array([-1.0, -2.0, 0.0, 1.0, 2.0, 1.5]), "room": "office_6_1"},
        {"name": "table_2", "cls": "table",
         "bbox": np.array([3.0, 3.0, 0.0, 5.0, 4.0, 0.8]), "room": "office_6_1"},
    ]
    boxes, labels = objects_as_boxes(objs)
    assert boxes.shape == (2, 7)
    assert np.allclose(boxes[:, 6], 0.0), "yaw 必须恒为 0（GT 是轴对齐盒）"
    # 中心与尺寸
    assert np.allclose(boxes[0, :3], [0.0, 0.0, 0.75])
    assert np.allclose(boxes[0, 3:6], [2.0, 4.0, 1.5])
    assert labels == ["chair", "table"]


def test_objects_as_boxes_skips_bad_entries():
    """缺字段 / 含 NaN 的物体应被跳过，而不是产生垃圾框。"""
    objs = [
        {"name": "good", "cls": "chair", "bbox": np.array([0, 0, 0, 1, 1, 1.0]), "room": "r"},
        {"name": "short", "cls": "chair", "bbox": np.array([0.0, 0.0, 0.0]), "room": "r"},
        {"name": "nan", "cls": "chair",
         "bbox": np.array([np.nan, 0, 0, 1, 1, 1.0]), "room": "r"},
    ]
    boxes, labels = objects_as_boxes(objs)
    assert boxes.shape == (1, 7) and labels == ["chair"]


def test_objects_as_boxes_empty():
    boxes, labels = objects_as_boxes([])
    assert boxes.shape == (0, 7) and labels == []


def test_filter_objects_matches_case_insensitively_and_by_substring():
    objs = [{"cls": "Chair"}, {"cls": "table"}, {"cls": "bookcase"},
            {"cls": "wall"}, {"cls": "clutter"}]
    got = {o["cls"] for o in filter_objects(objs, ["chair", "TABLE"])}
    assert got == {"Chair", "table"}
    # 空 prompts 表示全要
    assert len(filter_objects(objs, [])) == len(objs)


# ==========================================================================
# 5) 语义标签（数据集不带这个文件）
# ==========================================================================
def test_load_semantic_labels_missing_file_gives_actionable_error(tmp_path):
    """文件不在数据集里，报错必须说清"去哪拿" —— 否则会浪费半小时排查。"""
    with pytest.raises(FileNotFoundError) as exc:
        load_semantic_labels(tmp_path / "nope.json")
    assert "semantic_labels.json" in str(exc.value)
    assert "2D-3D-Semantics" in str(exc.value)


def test_load_semantic_labels_and_class_mapping(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text('["<UNK>_0_<UNK>_0_0", "beam_10_hallway_6_1", "chair_1_office_2_1"]',
                 encoding="utf-8")
    labels = load_semantic_labels(p)
    assert len(labels) == 3
    assert semantic_class(1, labels) == "beam"
    assert semantic_class(2, labels) == "chair"
    # 越界索引 = 无数据标记（#0D0D0D 编码后约 856845）
    assert semantic_class(856845, labels) is None
    assert semantic_class(-1, labels) is None


# ==========================================================================
# 6) 采集点：同一位置、不同朝向（多视角融合的前提）
# ==========================================================================
def test_location_groups_frames_by_uuid(tmp_path):
    """★ 一个采集点内的多帧**共享同一个光心** —— 这是"融合成全景"的前提。

    实测：同一 uuid 下 frame_0 与 frame_10 的 `camera_location` 完全相同、
    但 FOV 与旋转不同。这条断言用一个合成的临时目录验证**分组逻辑**，
    真数据上的事实由 `scripts/32` 验证。
    """
    import json
    from pathlib import Path

    root = Path(tmp_path)
    (root / "data" / "pose").mkdir(parents=True)
    uuid = "a" * 32
    C = np.array([1.0, 2.0, 3.0])
    for f, ang_deg in ((0, 0.0), (1, 40.0)):
        ang = np.radians(ang_deg)
        R = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                      [np.sin(ang), np.cos(ang), 0.0],
                      [0.0, 0.0, 1.0]])
        t = -R @ C
        obj = {"camera_rt_matrix": np.hstack([R, t.reshape(3, 1)]).tolist(),
               "camera_k_matrix": [[500, 0, 540], [0, 500, 540], [0, 0, 1]],
               "camera_location": C.tolist(), "frame_num": f}
        (root / "data" / "pose" /
         f"camera_{uuid}_office_6_frame_{f}_domain_pose.json").write_text(
            json.dumps(obj), encoding="utf-8")

    from roboground.data.stanford2d3d import list_locations

    locs = list_locations(root)
    assert len(locs) == 1, "同一 uuid 的多帧应聚成一个采集点"
    assert locs[0].frame_ids == [0, 1]

    # 两帧的光心必须一致（同一位置不同朝向）
    C0 = -np.asarray(locs[0].pose(0).R).T @ np.asarray(locs[0].pose(0).t)
    C1 = -np.asarray(locs[0].pose(1).R).T @ np.asarray(locs[0].pose(1).t)
    assert np.allclose(C0, C1, atol=1e-9), "同采集点的光心必须一致"
    # 朝向必须不同
    assert not np.allclose(np.asarray(locs[0].pose(0).R), np.asarray(locs[0].pose(1).R))
