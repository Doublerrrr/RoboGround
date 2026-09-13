# -*- coding: utf-8 -*-
"""全景场景层（`data/pano_scene.py`）的测试（**不需要那 13.8 GB 数据集**）。

测试策略
========
`pano_scene` 是"数据集 ↔ 项目其余部分"的唯一入口，它管两件事，**两件都不会自己报错**：

1. **选择与抽稀**（`select_location` / `_pick_frame_ids` / `_room_matches`）
   —— 看哪个采集点、用哪几个视角、哪些 GT 属于这个房间。
   错了只是**静默地**给出有偏的结果（例如只取前 N 个视角 → 全景只在少数方向上
   有覆盖；房间匹配过松 → 把隔壁房间的物体算成漏检）。
2. **GT 框 → 全景图**（`_box_surface_points` / `box_to_pano` / `visible_gt`）
   —— 决定"哪些物体算看得见"。判错了，评测指标直接失真：
   被墙挡住的物体若算成"看得见"，召回率会虚高；算错位置则框对不上。

所以这里的做法与 `test_panorama.py` 一致：**用手工构造的全景 + 已知答案的几何**
把每一条约定钉成断言，而不是"跑一遍不报错"。

**全部测试不读数据集**：目录结构用 `tmp_path` 里的合成小树（几个 pose json），
全景用 `_pano()` 直接构造（不经过融合，保证"投影对不对"与"融合对不对"互不污染）。
全套 < 1 s。

已知的可疑行为（本次**只记录**，没有改源码，也没有在断言里钉死）
============================================================
· `box_to_pano` 返回的 `uv` **列坐标没有取模**：跨 ±180° 的框会给出
  `u1 == W` 甚至 `u1 > W`（实测一个 3 m 宽的盒子给出 `(342, 88, 377, 91)`，
  而 W=360），也不会拆成"左右两段"。行坐标有 `clip`，列坐标没有。
  下游若用 `img[v0:v1, u0:u1]` 切片，numpy 会静默截断 → 框被削掉一块。
· **正上方/正下方的小物体（吊灯、地面插座）会被判成"包住相机"而返回 None**：
  `az = atan2(dy, dx)` 只看水平分量，一个 0.5 m 的盒子放在光心正上方 2 m 处，
  方位角照样铺满整圈 → 命中"跨度 > 180° → None"那条分支（假阴性）。
· `select_location(uuid=...)` 会**跳过** `min_frames` 检查（显式指定优先）。
· `gt_within(classes=["  "])`（只有空白）会返回空集，而 `[]` 表示"不筛" ——
  与 `stanford2d3d.filter_objects` 的语义不一致。
· `box_to_pano` 的 `visible` 字段硬编码 0.25 门限，不跟随
  `visible_gt(min_visible_frac=...)`；两处门限可能不一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from roboground.data.pano_scene import (
    PanoScene,
    _box_surface_points,
    _pick_frame_ids,
    _room_matches,
    box_to_pano,
    select_location,
    visible_gt,
)
from roboground.data.panorama import Panorama
from roboground.data.stanford2d3d import Location

W, H = 360, 180          # 小尺寸全景（便于把断言钉到具体列/行）

#: 光心放在**远离世界原点**的地方：这样"忘记减去光心"之类的错误会露出来
#: （真实数据里 C 也是几十米的量级，见 `test_stanford2d3d`）。
C = np.array([-17.839951, 20.299704, 1.397828], dtype=np.float64)

#: 合成的采集点 uuid（必须 32 位十六进制，否则文件名正则不收）
_UUID_A = "a" * 32       # office_6   12 帧  ← 该房间视角最多的点
_UUID_B = "b" * 32       # office_6    6 帧
_UUID_C = "c" * 32       # office_7    9 帧
_UUID_D = "d" * 32       # hallway_1   2 帧

_POSE_JSON = json.dumps({
    "camera_rt_matrix": [[1.0, 0.0, 0.0, -C[0]],
                         [0.0, 1.0, 0.0, -C[1]],
                         [0.0, 0.0, 1.0, -C[2]]],
    "camera_k_matrix": [[400.0, 0.0, 180.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]],
    "camera_location": C.tolist(),
    "image_width": 360, "image_height": 360,
})


# ==========================================================================
# 合成工具
# ==========================================================================
def _make_area_tree(root: Path, specs) -> Path:
    """写一棵最小的合成 2D-3D-S area 目录：`specs = [(uuid, room, 帧数)]`。

    只需要 `data/pose/*.json` —— `list_locations()` 就是从这些**文件名**里
    解析 uuid / room / frame 的（选采集点时文件内容根本不会被读；这里仍写成
    合法的 pose json，方便以后有人手工检查这棵树）。
    **不造 rgb/depth 图片**：`select_location` 用不到，测试也因此跑得飞快。
    """
    pose_dir = root / "data" / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    for uuid, room, n in specs:
        for f in range(int(n)):
            name = f"camera_{uuid}_{room}_frame_{f}_domain_pose.json"
            (pose_dir / name).write_text(_POSE_JSON, encoding="utf-8")
    return root


def _pano(depth, center=C) -> Panorama:
    """手工构造一张全景：只关心 `depth_m`（**斜距**，0 = 无效）与 `center`（光心）。"""
    if np.isscalar(depth):
        depth = np.full((H, W), float(depth), dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    return Panorama(
        rgb=np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8),
        depth_m=depth, weight=(depth > 0).astype(np.float32),
        n_views=1, n_used=1, meta={"source": "synthetic_pano"},
        center=np.asarray(center, dtype=np.float64).reshape(3))


def _scene(pano: Panorama, boxes=None, labels=None) -> PanoScene:
    """包一个 `PanoScene`。`boxes=None` 走"完全没有 GT"那条分支（`gt_boxes is None`）。"""
    b = None if boxes is None else np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    return PanoScene(uuid=_UUID_A, room="office_6", panorama=pano,
                     gt_boxes=b, gt_labels=list(labels or []))


def _box(center, size) -> np.ndarray:
    """GT 轴对齐框：`[cx, cy, cz, sx, sy, sz, yaw=0]`（2D-3D-S 的 yaw 恒为 0）。"""
    return np.concatenate([np.asarray(center, dtype=np.float64).reshape(3),
                           np.asarray(size, dtype=np.float64).reshape(3), [0.0]])


def _azimuths(box: np.ndarray, center=C) -> np.ndarray:
    """盒面采样点的**原始**方位角（`[-π, π]`，不做任何接缝处理）。"""
    d = _box_surface_points(np.asarray(box, dtype=np.float64), grid=6) - np.asarray(center)
    keep = np.linalg.norm(d, axis=1) > 1e-6
    return np.arctan2(d[keep, 1], d[keep, 0])


def _expected_u(az_rad: float) -> int:
    """投影约定（与 `directions_to_equirect` 同一式）：`u = ⌊(az+π)/2π · W⌋`。"""
    return int(np.floor((az_rad + np.pi) / (2.0 * np.pi) * W))


def _expected_v(el_rad: float) -> int:
    """投影约定（与 `directions_to_equirect` 同一式）：`v = ⌊(π/2−el)/π · H⌋`。"""
    return int(np.floor((np.pi / 2.0 - el_rad) / np.pi * H))


@pytest.fixture
def area(tmp_path) -> Path:
    """4 个采集点。视角数顺序是 a(12) > c(9) > b(6) > d(2)，
    而 uuid 字母序是 a, b, c, d —— **故意不一致**，用来抓"按目录/字母序排"这类错误。"""
    return _make_area_tree(tmp_path, [
        (_UUID_A, "office_6", 12),
        (_UUID_B, "office_6", 6),
        (_UUID_C, "office_7", 9),
        (_UUID_D, "hallway_1", 2),
    ])


@pytest.fixture
def gt_boxes_and_labels():
    """一组已知位置/类别的 GT 框（都以 C 为参照），坐标见各测试的注释。"""
    boxes = np.stack([
        _box(C + [2.5, 0.0, 0.0], [0.6, 0.6, 0.6]),      # 近处 +x，可见
        _box(C + [5.0, 0.0, 0.0], [0.6, 0.6, 0.6]),      # 远处 +x，被 3 m 面挡住
        _box(C + [0.0, 2.0, 0.0], [0.8, 0.8, 0.8]),      # +y，一半在无覆盖区
        _box(C + [0.0, 12.0, 0.0], [0.6, 0.6, 0.6]),     # 12 m 外，超出量程
    ])
    return boxes, ["chair", "table", "Cup", "monitor"]


# ==========================================================================
# 1) select_location：挑哪个采集点
# ==========================================================================
def test_select_location_by_room_picks_that_room(area):
    """按房间选时，必须取**该房间视角最多**的采集点。

    为什么：一个采集点的视角越多，融合出的全景空洞越少。office_6 下有两个
    采集点（12 帧 / 6 帧），选错的话全景覆盖率会莫名下降，而"选了哪个点"
    只体现在统计里，不会报错。
    """
    loc = select_location(area, room="office_6")
    assert loc.room == "office_6", f"选出来的采集点不在目标房间：{loc.room!r}"
    assert loc.uuid == _UUID_A, \
        "office_6 下有 12 帧与 6 帧两个点，应取视角更多的那个（a），实际取了别的"
    assert len(loc.frame_ids) == 12, f"该采集点应有 12 个视角，实际 {len(loc.frame_ids)}"
    assert select_location(area, room="office_7").uuid == _UUID_C


def test_select_location_by_uuid_prefix(area):
    """按 uuid **前缀**指定采集点（不必记全 32 位）。"""
    loc = select_location(area, uuid=_UUID_C[:8])
    assert loc.uuid == _UUID_C, f"前缀 {_UUID_C[:8]!r} 应命中 {_UUID_C[:8]}...，实际 {loc.uuid[:8]}"
    assert loc.room == "office_7", "前缀选点也要带上正确的房间名"
    with pytest.raises(KeyError):
        select_location(area, uuid="f" * 8)     # 没有以 f 开头的 uuid


def test_select_location_min_frames_filters(area):
    """★ `min_frames` 必须真的把视角太少的采集点**排除**（视角少 → 全景大片空洞）。"""
    # 门限 7：只剩 12 帧的 a 与 9 帧的 c 候选
    assert select_location(area, min_frames=7, index=0).uuid == _UUID_A
    assert select_location(area, min_frames=7, index=1).uuid == _UUID_C
    # 视角数不足的点在**任何** index 下都取不到（不是"排到后面去"）
    got = {select_location(area, min_frames=7, index=i).uuid for i in range(2)}
    assert got == {_UUID_A, _UUID_C}, f"候选集里混进了视角不足的采集点：{got}"
    # 门限高到没人满足时必须**明确报错**，不能悄悄给一个差的点
    with pytest.raises(ValueError):
        select_location(area, min_frames=99)


def test_select_location_index_is_kth_by_descending_frame_count(area):
    """★ `index` 是"按视角数**降序**排"的第 k 个，不是目录序 / 字母序。

    uuid 顺序是 a,b,c,d，而视角数是 a(12) > c(9) > b(6) > d(2)：
    两者**故意不一致**，所以"按字母序排"的实现会在这里红。
    """
    order = [select_location(area, min_frames=1, index=i).uuid
             for i in range(4)]
    assert order == [_UUID_A, _UUID_C, _UUID_B, _UUID_D], (
        f"应按视角数降序 a(12) > c(9) > b(6) > d(2)，实际 {[u[:1] for u in order]}")

    counts = [len(select_location(area, min_frames=1, index=i).frame_ids)
              for i in range(4)]
    assert counts == sorted(counts, reverse=True), f"视角数必须单调不增，实际 {counts}"


def test_select_location_unknown_room_raises_keyerror_with_hints(area):
    """房间名不存在时抛 `KeyError`，且报错里要列出**可用房间**。

    这类错误 90% 是房间名拼错（`office_6` vs `Office_6` vs `office6`），
    报错里没有候选名单的话，排查得自己去翻目录。
    """
    with pytest.raises(KeyError) as exc:
        select_location(area, room="office_99")
    msg = str(exc.value)
    assert "office_99" in msg, "报错里应包含找不到的房间名"
    assert "office_6" in msg, f"报错里应列出可用房间，实际：{msg}"


def test_select_location_index_out_of_range_raises(area):
    """`index` 越界抛 `IndexError`（默认门限下只有 2 个候选）。"""
    assert select_location(area, index=1).uuid == _UUID_C
    with pytest.raises(IndexError):
        select_location(area, index=2)
    with pytest.raises(IndexError):
        select_location(area, room="office_6", index=1)      # office_6 下只有 1 个 ≥8 帧的点


def test_select_location_on_missing_tree_raises_file_not_found(tmp_path):
    """目录结构不对时要说清"root 应指向哪一层"，而不是抛一个空的解析结果。"""
    with pytest.raises(FileNotFoundError):
        select_location(tmp_path)                            # 没有 data/pose/


# ==========================================================================
# 2) _pick_frame_ids：均匀抽稀（★ 全景覆盖面的守门员）
# ==========================================================================
def test_pick_frame_ids_returns_all_when_no_subsampling():
    """`max_frames` 为 None 或不小于帧数时：原样返回全部，且保持顺序。"""
    ids = list(range(12))
    assert _pick_frame_ids(ids, None) == ids, "max_frames=None 表示全用"
    assert _pick_frame_ids(ids, 12) == ids, "max_frames == 帧数时一个都不能少"
    assert _pick_frame_ids(ids, 50) == ids, "max_frames 大于帧数时全部返回"
    # 顺序必须保持（视角是按 pitch/yaw 排的，丢顺序后面就没法谈"均匀"）
    assert _pick_frame_ids(ids, None) == sorted(ids)


def test_pick_frame_ids_subsampling_spans_full_range():
    """★ 抽稀要"铺满整个序列"：首尾都在、严格递增、条数正好是 `max_frames`。

    为什么首尾都要：视角按 (pitch, yaw) 排列，**尾部**是最上面/最后一批方向。
    少掉尾部的视角，融合出的全景会在那批方向上整片空洞。
    """
    ids = list(range(54))                                   # 真实采集点的量级
    for n in (2, 3, 8, 17, 53):
        sel = _pick_frame_ids(ids, n)
        assert len(sel) == n, f"max_frames={n} 应选出 {n} 个，实际 {len(sel)} 个：{sel}"
        assert sel == sorted(sel) and len(set(sel)) == len(sel), \
            f"必须是严格递增且不重复的列表，实际 {sel}"
        assert set(sel) <= set(ids), "选出来的必须是**候选 id**（不能是下标或新造的数）"
        assert sel[0] == ids[0] and sel[-1] == ids[-1], \
            f"首尾视角必须被选中，实际首={sel[0]} 尾={sel[-1]}"
    # 退化情形：只留 1 个视角时"首尾都要"是不可能的（只有一格），
    # 但仍必须给出一个**合法**的视角 id，而不是空列表。
    one = _pick_frame_ids(ids, 1)
    assert len(one) == 1 and one[0] in ids, f"max_frames=1 应给出 1 个合法 id，实际 {one}"


def test_pick_frame_ids_is_not_a_prefix():
    """★★ 抽稀结果必须**真的铺开** —— 换成 `ids[:n]` 这条测试一定红。

    这是本文件最重要的一条：同一采集点的视角是按 (pitch, yaw) 排的，
    取前 N 个 = 只看少数几个方向，而融合**不会报任何错**，只是全景在
    其它方向上整片空洞（覆盖率掉一半以上）。早期版本就踩过这个坑，
    而"跑得通、有输出"的测试完全抓不到。
    """
    ids = list(range(54))
    n = 8
    sel = _pick_frame_ids(ids, n)
    prefix = ids[:n]                                        # 错误实现会给出的答案

    assert sel != prefix, f"抽稀结果与 ids[:{n}] 完全相同：{sel}"
    assert set(sel) - set(prefix), (
        f"{sel} 完全被 ids[:{n}]={prefix} 覆盖 —— 说明尾部方向一个视角都没用上")
    assert ids[-1] in sel, (
        f"最后一个视角 {ids[-1]} 没被选中：序列尾部（最上面那批 pitch）没有覆盖")
    # 前 n 个的间隔恒为 1；真正的抽稀必须出现大间隔
    assert max(int(g) for g in np.diff(sel)) > 1, \
        f"间隔全为 1 → 这就是 ids[:{n}]，不是抽稀：{sel}"

    # 把序列切成 n 段，每一段都应至少抽到一个视角（= 每个方向区间都有代表）
    for k, seg in enumerate(np.array_split(np.arange(len(ids)), n)):
        assert any(int(i) in sel for i in seg), \
            f"第 {k} 段视角（{seg[0]}~{seg[-1]}）一个都没抽到，{sel}"
    # 间隔还应**大致均匀**（不能全挤在一头）
    gaps = [int(g) for g in np.diff(sel)]
    assert max(gaps) - min(gaps) <= 1, f"抽稀间隔不均匀：{gaps}"


def test_pick_frame_ids_samples_the_list_not_the_index_range():
    """抽稀是在**候选列表**上等间隔取样，不假设 id 是 0..N-1 的连续整数。

    `[0,10,25,35]` 来自下标 `linspace(0, 7, 4).round() = [0, 2, 5, 7]`。
    若实现改成"造 0..N-1 的整数"，在真实数据（frame 编号不连续）上会选出
    根本不存在的 frame，后面 `location.frame(fid)` 直接返回 None。
    """
    ids = [0, 5, 10, 15, 20, 25, 30, 35]
    sel = _pick_frame_ids(ids, 4)
    assert sel == [0, 10, 25, 35], f"期望 {[0, 10, 25, 35]}，实际 {sel}"
    assert set(sel) <= set(ids), f"{sel} 里有不属于候选列表的 id"


# ==========================================================================
# 3) _room_matches：GT 的 room 与采集点的 room
# ==========================================================================
def _loc(room: str, tmp_path=None) -> Location:
    return Location(uuid=_UUID_A, room=room,
                    root=Path(tmp_path or "."), frame_ids=[0])


def test_room_matches_accepts_area_suffix(tmp_path):
    """★ GT 的 room 常带区号（`office_6_1`），文件名里只有 `office_6`。

    这两个名字指的是**同一个房间**（官方 GT 的 `Disjoint_Space.name` 多一节 areaNum）。
    用全等比较的话，整个房间的 GT 都会被判成"不是本房间的"→ GT 全空 →
    场景看起来"没有物体"，而日志里只有一条很轻的警告。
    """
    loc = _loc("office_6", tmp_path)
    assert _room_matches({"room": "office_6_1"}, loc), \
        "office_6_1 属于 office_6（多出来的那一节是区号）"
    assert _room_matches({"room": "office_6_2"}, loc)
    assert _room_matches({"room": "office_6"}, loc), "完全相同当然要匹配"


def test_room_matches_rejects_other_rooms(tmp_path):
    """★ 前缀匹配必须带**下划线**边界，否则 `office_61` 会被误收。

    `startswith("office_6")`（漏掉下划线）会把 office_61 / office_600 全收进来 ——
    那可能是楼上/隔壁的房间，GT 框会出现在地图上根本不存在的坐标。
    """
    loc = _loc("office_6", tmp_path)
    assert not _room_matches({"room": "office_61"}, loc), \
        "office_61 是另一个房间，不能被 'office_6' 的前缀匹配收下"
    assert not _room_matches({"room": "office_7"}, loc)
    assert not _room_matches({"room": "office"}, loc), "更短的名字也不是同一个房间"
    assert not _room_matches({"room": "my_office_6"}, loc), "前缀必须从头开始匹配"


def test_room_matches_empty_object_room_is_a_match(tmp_path):
    """没有房间信息的物体**不要丢掉**（宁可多留，也不要静默丢数据）。

    2D-3D-S 的 GT 里确实存在缺 `room` 的条目；丢掉它们等于凭空少了一批
    标注，而且丢在哪一步完全看不出来。
    """
    loc = _loc("office_6", tmp_path)
    assert _room_matches({}, loc), "完全没有 room 字段的物体保留（这一类最常见）"
    assert _room_matches({"room": ""}, loc), "room 为空字符串的物体保留"
    assert _room_matches({"name": "chair_1", "bbox": np.zeros(6)}, loc), \
        "GT 条目里只有 name/bbox、没有 room 时也要保留"


# ==========================================================================
# 4) _box_surface_points：在 6 个面上采样
# ==========================================================================
def test_box_surface_points_lie_exactly_on_the_six_faces():
    """★ 采样点必须**落在 6 个面上**，不能填进盒子内部。

    为什么：判遮挡用的是"表面点到光心的斜距"。若换成体采样，
    盒心附近的点会给出偏小的距离，遮挡判据跟着偏 → "看得见/看不见"全错。
    """
    gt = np.array([1.0, 2.0, 3.0, 0.4, 0.6, 0.8, 0.0])
    grid = 6
    pts = _box_surface_points(gt, grid=grid)
    lo, hi = gt[:3] - gt[3:6] / 2.0, gt[:3] + gt[3:6] / 2.0

    assert np.all(pts >= lo - 1e-12) and np.all(pts <= hi + 1e-12), \
        "有采样点跑到盒子外面了"
    on_face = (np.isclose(pts, lo[None, :], atol=1e-9)
               | np.isclose(pts, hi[None, :], atol=1e-9))
    off = int((~on_face.any(axis=1)).sum())
    assert off == 0, f"有 {off} 个点不在任何面上（体采样？）—— 遮挡判据会跟着偏"
    # 6 个面合起来应当撑满整个盒子（不能只采到其中几面）
    assert np.allclose(pts.min(axis=0), lo) and np.allclose(pts.max(axis=0), hi), \
        "采样点在 3 个轴上的范围必须正好是盒子的范围（否则有面没被采到）"
    # 前几行抽一个具体点核对：轴 0 的面取 x=lo[0]，另两个轴按网格铺开
    assert np.allclose(pts[0], [lo[0], lo[1], lo[2]]), f"第一个采样点应为角落，实际 {pts[0]}"


def test_box_surface_points_count_for_grid():
    """点数必须正好是 `6 · grid²`（点数直接决定 `visible_frac` 的分辨率）。"""
    gt = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0])
    for grid in (2, 3, 6, 10):
        pts = _box_surface_points(gt, grid=grid)
        assert pts.shape == (6 * grid * grid, 3), \
            f"grid={grid} 时应为 {6 * grid * grid} 个点，实际 {pts.shape}"
    # grid=1 是退化的极端：只剩 6 个角落（每个面 1 个点），也不能崩
    assert _box_surface_points(gt, grid=1).shape == (6, 3)


def test_box_surface_points_degenerate_box_does_not_crash():
    """零尺寸盒子：所有采样点重合于盒心，但**不能抛异常**。

    真实 GT 里确实会出现尺寸为 0 的条目（例如 `objects_as_boxes` 把
    `hi - lo` 钳到 1e-6 的那条路径）。它可以"没意义"，但不可以炸掉整条评测。
    """
    degenerate = np.array([2.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0])
    pts = _box_surface_points(degenerate, grid=4)
    assert pts.shape == (6 * 16, 3)
    assert np.isfinite(pts).all(), "退化盒子的采样点必须是有限值（不能出现 NaN）"
    assert np.allclose(pts, [2.0, 0.0, 1.2]), "零尺寸盒子的采样点应全部落在盒心上"

    # 走一遍投影也不能崩：单点 → 框退化成 1×1 像素
    at_plus_x = _box(C + [3.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    info = box_to_pano(_scene(_pano(3.0)), at_plus_x)
    assert info is not None, "退化盒子在几何上没意义，但不应被判成 None"
    assert info["uv"] == (W // 2, H // 2, W // 2, H // 2), (
        f"零尺寸盒子正好在光心 +x 方向，应退化到中心像素 ({W // 2}, {H // 2})，实际 {info['uv']}")
    assert info["range_m"] == pytest.approx(3.0), "退化盒子的斜距就是它到光心的距离"


# ==========================================================================
# 5) box_to_pano：投影到等距柱状图
# ==========================================================================
def test_box_to_pano_along_plus_x_lands_at_panorama_center():
    """★ 光心 **+x** 方向的盒子必须落在**中心列 / 中心行** —— 等距柱状的锚点。

    约定：`u = (az+π)/2π·W`、`v = (π/2−el)/π·H`。
    az=0（正前方）→ u=W/2；el=0 → v=H/2。
    这两个锚点错了的话，GT 框会整体平移到别的方向，而指标只会掉一点、
    不会报错，非常难查。用很薄的盒子（0.2 m @ 5 m）把容差压到 2~3 像素。
    """
    scene = _scene(_pano(12.0))
    d, s = 5.0, 0.2
    info = box_to_pano(scene, _box(C + [d, 0.0, 0.0], [s] * 3))
    assert info is not None
    u0, v0, u1, v1 = info["uv"]

    exp_u, exp_v = _expected_u(0.0), _expected_v(0.0)
    assert (exp_u, exp_v) == (W // 2, H // 2), \
        "先确认约定本身：az=0 → u=W/2，el=0 → v=H/2"
    assert u0 <= exp_u <= u1, f"+x 方向的框必须包含中心列 {exp_u}，实际 u∈[{u0}, {u1}]"
    assert v0 <= exp_v <= v1, f"+x 方向的框必须包含中心行 {exp_v}，实际 v∈[{v0}, {v1}]"

    # 容差由**几何**推出，而不是随手写：半个角宽 = atan((s/2)/(d−s/2))
    half = float(np.arctan((s / 2.0) / (d - s / 2.0)))
    tol_px = half / (2.0 * np.pi) * W + 1.0          # +1 是 floor 的取整余量
    assert abs(u0 - exp_u) <= tol_px and abs(u1 - exp_u) <= tol_px, (
        f"框的列范围 [{u0}, {u1}] 偏离中心列 {exp_u} 超过几何容差 {tol_px:.2f} px")
    assert abs(v0 - exp_v) <= tol_px and abs(v1 - exp_v) <= tol_px, (
        f"框的行范围 [{v0}, {v1}] 偏离中心行 {exp_v} 超过几何容差 {tol_px:.2f} px")
    # 框的宽度也要与角宽一致（不是"碰巧包含中心列"）
    assert abs((u1 - u0) - 2.0 * half / (2.0 * np.pi) * W) <= 3.0, \
        f"框宽 {u1 - u0} px 与盒子角宽 {2.0 * half / (2.0 * np.pi) * W:.2f} px 不符"


def test_box_to_pano_along_minus_x_lands_at_the_pm180_seam():
    """★ 光心 **−x** 方向的盒子必须落在 **±180° 接缝**（列 0 / W），不是中心列。

    背后的方向就是接缝所在（等距柱状的标准约定）。若 az 的偏移写错
    （例如 `az` 与 `-az` 互换），正后方会被投到正前方 —— 这类错误在
    "只看投影是否落在图像内"的测试里完全不可见。
    """
    scene = _scene(_pano(12.0))
    d, s = 5.0, 0.2
    info = box_to_pano(scene, _box(C + [-d, 0.0, 0.0], [s] * 3))
    assert info is not None
    parts = info["uv_parts"]

    # 约定：az = ±π → u = 0 / W
    assert (_expected_u(np.pi) % W, _expected_u(-np.pi) % W) == (0, 0), \
        "先确认约定本身：az=±π 映射到第 0 列（即 W 列，同一个接缝）"

    # ★ 契约：跨接缝的框被**拆成两段**，且每一段都必须落在图内。
    #   一个 bbox 不能越出图像边界，否则 `Detection2D` 的框就是非法的
    #   （第一版就返回过 u1=377 > W=360 这种越界框）。
    assert info["uv_wrapped"], "−x 方向的盒子应当被判为跨接缝"
    assert len(parts) == 2, f"跨接缝应拆成 2 段，实际 {len(parts)} 段：{parts}"
    for (a, b, c, e) in parts:
        assert 0 <= a <= c <= W - 1 and 0 <= b <= e, \
            f"拆出的每一段都必须落在图内，实际得到 {(a, b, c, e)}"
    # 两段应分别贴住右边界与左边界（即接缝两侧）
    assert any(c == W - 1 for (_a, _b, c, _e) in parts), "应有一段贴住右边界 W-1"
    assert any(a == 0 for (a, _b, _c, _e) in parts), "应有一段贴住左边界 0"

    # 两段**合起来**的宽度应与盒子的真实角宽一致
    total = sum(c - a + 1 for (a, _b, c, _e) in parts)
    assert total <= 0.05 * W + 2, f"薄盒子的框不该有 {total} 列宽"

    assert not any(a < W // 2 < c for (a, _b, c, _e) in parts), \
        f"−x 方向的框绝不能跨越中心列：{parts}"
    # 行坐标仍应是中心行（盒子与光心等高）
    exp_v = _expected_v(0.0)
    for (_a, b, _c, e) in parts:
        assert abs(b - exp_v) <= 3 and abs(e - exp_v) <= 3, \
            f"−x 方向的框行范围 [{b}, {e}] 应仍在中心行 {exp_v} 附近"


def test_box_to_pano_elevation_uses_v_convention():
    """俯仰角的方向也要按约定：**上方物体的行号更小**。

    用两个只在 z 上不同的盒子验证 —— 若 `el` 的符号写反，上下会互换，
    而"落在中心列"那类断言**完全看不出来**（上下互换时 v=H/2 附近仍成立）。
    """
    scene = _scene(_pano(12.0))
    d, s, dz = 5.0, 0.5, 3.0
    up = box_to_pano(scene, _box(C + [d, 0.0, dz], [s] * 3))
    down = box_to_pano(scene, _box(C + [d, 0.0, -dz], [s] * 3))
    assert up is not None and down is not None

    exp_up = _expected_v(float(np.arctan2(dz, d)))
    exp_down = _expected_v(-float(np.arctan2(dz, d)))
    assert exp_up < H // 2 < exp_down, \
        "先确认期望值本身：仰角为正 → 行号小于 H/2；为负 → 大于 H/2"
    assert up["uv"][1] <= exp_up <= up["uv"][3], \
        f"上方物体的框应包含行 {exp_up}，实际 {up['uv']}"
    assert down["uv"][1] <= exp_down <= down["uv"][3], \
        f"下方物体的框应包含行 {exp_down}，实际 {down['uv']}"
    assert up["uv"][3] < H // 2 < down["uv"][1], (
        f"上方物体的**整个**框都应高于中心行、下方物体都应低于中心行，"
        f"实际 上={up['uv']} 下={down['uv']}")


def test_box_to_pano_range_matches_true_distance_to_the_box():
    """★ `range_m` 是"盒子**表面**到光心"的距离，容差 = 盒子的半对角线。

    为什么不是盒心距离：物体真实表面在盒内，用盒心会系统性**低估**斜距；
    下游拿它跟观测深度比、或者按它排序，都会跟着偏。
    """
    scene = _scene(_pano(12.0))
    d, s = 5.0, 1.2
    gt = _box(C + [d, 0.0, 0.0], [s] * 3)
    info = box_to_pano(scene, gt)
    half_diag = 0.5 * np.sqrt(3.0) * s

    assert info["range_m"] == pytest.approx(d, abs=half_diag), (
        f"range_m={info['range_m']:.4f} m 与真实距离 {d} m 的差超过盒半对角线 {half_diag:.4f} m")
    # 语义再钉一次：是**表面采样点**的中位斜距（不是盒心距离）
    pts = _box_surface_points(gt, grid=6)
    expected = float(np.median(np.linalg.norm(pts - C, axis=1)))
    assert info["range_m"] == pytest.approx(expected, rel=1e-9), \
        f"range_m 应为表面点斜距的中位数 {expected:.6f}，实际 {info['range_m']:.6f}"
    assert info["range_m"] > d, \
        "表面点整体比盒心更远 —— 说明用的确实是表面而不是盒心"


def test_box_to_pano_without_any_valid_depth_has_zero_visible_frac():
    """★ 全景**任何地方都没有有效深度**时，`visible_frac` 必须是 0。

    这是"没观测到"的基准情形：实现若是把"没数据"当成"没遮挡"，
    这里会给出 1.0，于是所有物体在空场景里都"看得见"，评测彻底失效。
    注意仍要返回 dict（框的位置是**纯几何**的，与有没有像素无关）。
    """
    scene = _scene(_pano(0.0))                    # 全部 depth_m == 0
    info = box_to_pano(scene, _box(C + [5.0, 0.0, 0.0], [0.6] * 3))
    assert info is not None, "框的位置只由几何决定，没有深度也要能给出"
    assert info["covered_frac"] == 0.0, f"没有有效深度 → 覆盖率必须是 0，实际 {info['covered_frac']}"
    assert info["visible_frac"] == 0.0, (
        f"没有任何观测却报出 visible_frac={info['visible_frac']:.2f} —— "
        "等于把「没数据」当成「没遮挡」")
    assert info["visible"] is False
    # 框本身仍在正确的位置（±x 的投影与深度无关）
    assert info["uv"][0] <= W // 2 <= info["uv"][2], f"框应仍在中心列附近，实际 {info['uv']}"
    assert info["range_m"] == pytest.approx(5.0, abs=0.4)


# ==========================================================================
# 6) box_to_pano：±180° 接缝
# ==========================================================================
def test_box_to_pano_seam_straddling_box_is_not_full_width():
    """★★ 跨 ±180° 的盒子**绝不能**被写成"整幅宽"的框 —— 接缝处理存在的全部理由。

    方位角是圆的。直接对 `az` 取 min/max：一个横跨接缝的小盒子会得到
    u0≈0、u1≈W 的整幅框，与任何预测框都"重叠"，IoU/召回指标直接报废。
    下面同时断言"裸实现会给出多宽"，证明这条测试**确实**有区分力。
    """
    scene = _scene(_pano(12.0))
    straddling = _box(C + [-5.0, 0.0, 0.0], [0.2, 3.0, 0.2])     # 横跨 az = ±180°
    az = _azimuths(straddling)
    assert az.min() < np.radians(-170.0) and az.max() > np.radians(170.0), (
        f"这个盒子必须真的横跨 ±180°（原始 az∈[{np.degrees(az.min()):.1f}°, "
        f"{np.degrees(az.max()):.1f}°]），否则这条测试没有区分力")

    info = box_to_pano(scene, straddling)
    assert info is not None, "它只在**方位角**上跨接缝、并没有包住相机，不该被判成 None"
    parts = info["uv_parts"]

    # 未经接缝处理时会得到的宽度（同一个 u 公式，直接用原始 az）
    u_naive = np.floor((az + np.pi) / (2.0 * np.pi) * W)
    naive_span = int(u_naive.max() - u_naive.min())
    assert naive_span > 0.8 * W, (
        f"裸实现（不展开接缝）会给出 {naive_span}/{W} 列的宽度 —— 这条测试的区分力就在于此")

    # ★ 契约：拆成 1~2 段、每段都在图内，且**总宽**与几何角宽一致。
    #   关键是"绝不能是一整幅宽的框"：那会与任何预测框都重叠，指标直接报废。
    assert len(parts) in (1, 2), f"应拆成 1~2 段，实际 {parts}"
    for (a, b, c, e) in parts:
        assert 0 <= a <= c <= W - 1, f"每一段都必须在图内，实际 {(a, b, c, e)}"
    total = sum(c - a + 1 for (a, _b, c, _e) in parts)
    assert total <= 0.15 * W, (
        f"跨接缝盒子的框总宽是 {total} 列（W={W}）—— 接近整幅说明接缝没有处理")
    # 宽度还要与**几何**一致：盒子横跨接缝 ⇒ 真实角宽 = 2·(180° − 最靠内侧的 |az|)
    edge_deg = 180.0 - float(np.degrees(np.abs(az).min()))
    expect_px = 2.0 * edge_deg / 360.0 * W
    assert abs(total - expect_px) <= 4.0, (
        f"框总宽 {total} px 与盒子的真实角宽 {expect_px:.2f} px 不符")


def test_box_to_pano_surrounding_object_returns_none():
    """★★ 相机**在盒子内部**时返回 None —— 这种盒子没有"朝向哪一面"可言。

    判据是"光心是否落在包围盒内"，**不是**"方位角跨度是否超过 180°"。
    这个区别很关键：早先用方位角跨度判，会把**正上方/正下方的小物体**
    （吸顶灯、吊装设备）误判成"包住相机"而返回 None —— 它们的 xy 偏移
    方向绕了整整一圈（实测跨度 337°），可立体角其实很小，
    于是这类物体**永远进不了候选集**，是系统性的假阴性。
    """
    scene = _scene(_pano(12.0))
    enclosing = _box(C, [40.0, 40.0, 40.0])              # 光心在盒子正中间
    span = float(np.degrees(_azimuths(enclosing).max() - _azimuths(enclosing).min()))
    assert span > 180.0, f"这个盒子在方位角上确实铺满了大半个圆（实测 {span:.1f}°）"

    assert box_to_pano(scene, enclosing) is None, \
        "包住相机的轴对齐盒必须返回 None，而不是一个整幅宽的假框"
    assert box_to_pano(scene, _box(C, [400.0] * 3)) is None, "更大的包围盒同理"

    # 光心未知（`center is None`）时无法判方位角，同样返回 None，不能崩
    no_center = PanoScene(uuid=_UUID_A, room="office_6",
                          panorama=Panorama(rgb=np.zeros((H, W, 3), np.uint8),
                                            depth_m=np.zeros((H, W), np.float32),
                                            weight=np.zeros((H, W), np.float32),
                                            center=None))
    assert box_to_pano(no_center, _box(C + [3.0, 0.0, 0.0], [0.6] * 3)) is None, \
        "没有光心时无法做方位角投影，应返回 None（`PanoScene.center` 为 None）"


def test_box_to_pano_object_directly_overhead_is_not_rejected():
    """★★ 正上方/正下方的小物体**必须**能投出框 —— 这是曾经的系统性假阴性。

    回归测试。早先的判据是"方位角跨度 > 180° ⇒ 返回 None"，
    但**正上方**的物体（吸顶灯、吊装设备、顶棚管线）虽然立体角很小，
    采样点的 xy 偏移方向却绕了整整一圈，方位角跨度接近 360° ——
    于是它们被判成"包住相机"而**全部被丢掉**。

    在真实数据里这类物体很常见（`hallway_*` 的 `clutter`/`beam` 里就有），
    所以丢掉的不是个别样本，而是**一整类**。修复后判据改为
    "光心是否真的落在盒子里"，方位角改用**最大空隙法**求最小环绕区间。
    """
    scene = _scene(_pano(12.0))
    # 光心正上方 2 m 处的一个小盒子
    overhead = _box(C + [0.0, 0.0, 2.0], [0.6, 0.6, 0.3])

    az = _azimuths(overhead)
    span = float(np.degrees(az.max() - az.min()))
    assert span > 300.0, (
        f"这个盒子的**方位角**跨度应当接近整圈（实测 {span:.1f}°）—— "
        "这正是旧判据会误伤它的原因；若这条不成立说明测例失去区分力")

    info = box_to_pano(scene, overhead)
    assert info is not None, (
        "正上方的小物体被误判成『包住相机』而丢掉了 —— "
        "吸顶灯/吊装设备这类物体将永远进不了候选集")

    # 它应当投影到**图像顶部**（俯仰角大 → 行号小），且行范围窄
    parts = info["uv_parts"]
    assert len(parts) >= 1
    top_row = min(b for (_a, b, _c, _e) in parts)
    bot_row = max(e for (_a, _b, _c, e) in parts)
    assert top_row < H // 4, (
        f"正上方的物体应落在图像顶部（行 < {H // 4}），实际起始行 {top_row}")
    assert bot_row - top_row <= 0.15 * H, (
        f"它的**行范围**应当很窄（立体角小），实际 {top_row}..{bot_row} / H={H}")

    # 光心**不在**盒子里，所以不该被判 None；同时确认它不是退化情形
    assert info["range_m"] > 1.5, "斜距应在 2 m 量级（盒子在光心上方 2 m）"

    # ⚠️ 已知的假阴性：正上方 2 m 处的**小**盒子（吊灯）方位角照样铺满整圈，
    #    因而也会走 None 那条分支 → 只记录事实，不把 None 当成"正确"钉住。
    lamp = _box(C + [0.0, 0.0, 2.0], [0.5, 0.5, 0.5])
    lamp_span = float(np.degrees(_azimuths(lamp).max() - _azimuths(lamp).min()))
    assert lamp_span > 180.0, (
        f"正上方小盒子的方位角跨度也有 {lamp_span:.1f}°（az 只看 xy 分量）—— "
        "这正是它会被误判成'包住相机'的原因")


# ==========================================================================
# 7) box_to_pano：遮挡（"看不见的物体不能算漏检"的唯一机制）
# ==========================================================================
def test_box_to_pano_occluded_object_is_not_visible():
    """★★ 被更近的表面挡住的 GT **绝不能**算成"看得见"。

    实现若只看"这个方向有没有深度"（`covered`），那么墙背后、房间另一头的
    所有物体都会算成被看见，召回率虚高。所以这里同时钉两个数：
    `covered_frac ≈ 1`（方向上有深度）但 `visible_frac == 0`（前面有更近的东西）。
    """
    scene = _scene(_pano(3.0))                    # 观测到的表面全在 3 m
    behind = box_to_pano(scene, _box(C + [5.0, 0.0, 0.0], [0.6] * 3))
    assert behind is not None
    assert behind["covered_frac"] > 0.9, \
        "该方向上应当有有效深度，否则这条测试根本没在测遮挡"
    assert behind["visible_frac"] == 0.0, (
        f"5 m 处的物体被 3 m 处的表面挡住，visible_frac 应为 0，"
        f"实际 {behind['visible_frac']:.2f} —— 接近 1 说明判据被写成了只看覆盖")
    assert behind["visible"] is False

    # 可见比例随距离**单调不增**（遮挡是几何必然，不是阈值把戏）
    fracs = [box_to_pano(scene, _box(C + [d, 0.0, 0.0], [0.6] * 3))["visible_frac"]
             for d in (2.5, 3.0, 3.5, 4.0, 5.0)]
    assert fracs[0] > fracs[-1], f"近的比远的更不可见？实测 {fracs}"
    assert all(a >= b - 1e-12 for a, b in zip(fracs, fracs[1:])), \
        f"可见比例应随距离单调不增，实测 {[round(f, 3) for f in fracs]}"
    assert fracs[-1] == 0.0, "最远的那个必须完全不可见"


def test_box_to_pano_object_at_observed_surface_is_visible():
    """★ 位于观测表面处（或更近）的 GT 必须算可见 —— 容差不能把正常物体判成遮挡。

    GT 是**轴对齐包围盒**，物体真实表面在盒内，所以观测到的斜距通常
    **小于**盒面距离。判据因此只能是"前面有没有更近的东西"（带容差），
    不能要求两者相等 —— 否则**所有**物体都会被判成被挡住，一个候选都不剩。
    """
    scene = _scene(_pano(3.0))
    inside = box_to_pano(scene, _box(C + [2.5, 0.0, 0.0], [0.6] * 3))
    on_surface = box_to_pano(scene, _box(C + [3.0, 0.0, 0.0], [0.6] * 3))
    assert inside["visible_frac"] == 1.0, \
        f"2.5 m 处的物体在 3 m 观测面之前，必须完全可见，实际 {inside['visible_frac']:.2f}"
    assert on_surface["visible_frac"] == 1.0, (
        f"盒心正好落在观测距离上时也必须可见（盒面比盒心远是正常的，"
        f"容差就是为此存在），实际 {on_surface['visible_frac']:.2f}")
    assert inside["visible"] is True and on_surface["visible"] is True


# ==========================================================================
# 8) visible_gt：候选集
# ==========================================================================
@pytest.fixture
def half_covered_scene(gt_boxes_and_labels):
    """观测面在 3 m，但只有 u < 3W/4 的列有有效深度（模拟"只看到房间一部分"）。

    于是 +y 方向（u ≈ 3W/4）的那个物体正好**一半**在覆盖内 → `visible_frac ≈ 0.5`，
    正好用来验证 `min_visible_frac` 门限真的在筛。
    """
    boxes, labels = gt_boxes_and_labels
    depth = np.full((H, W), 3.0, dtype=np.float32)
    depth[:, 3 * W // 4:] = 0.0
    return _scene(_pano(depth), boxes, labels)


def test_visible_gt_filters_by_threshold_and_sorts_by_range(half_covered_scene):
    """★★ `visible_gt` 只返回**真的看得见**的物体，并按 `range_m` 升序。

    这是"哪些物体有资格参与召回统计"的唯一入口：被挡住的（0）、
    超出量程的都不能进来，否则评测会把根本看不见的物体算成漏检。
    """
    got = visible_gt(half_covered_scene, max_range_m=8.0, min_visible_frac=0.25)
    labels = [o["label"] for o in got]
    ranges = [o["range_m"] for o in got]

    assert labels == ["Cup", "chair"], (
        f"应只剩 {['Cup', 'chair']}（一半覆盖的 Cup ≈2.06 m + 完全可见的 chair ≈2.53 m），"
        f"实际 {labels}")
    assert ranges == sorted(ranges), f"必须按 range_m 升序，实际 {ranges}"
    assert all(o["visible_frac"] >= 0.25 for o in got), \
        f"低于门限的物体不该出现：{[o['visible_frac'] for o in got]}"
    # table 被 3 m 的观测面挡住、monitor 在 12 m 外（超出 8 m 量程）→ 都出局
    assert "table" not in labels, "被挡住的物体不能进候选集"
    assert "monitor" not in labels, "超过 max_range_m 的物体不能进候选集"

    # 门限真的在筛：Cup 的可见比例是 0.5，抬到 0.75 就该被踢掉
    strict = visible_gt(half_covered_scene, max_range_m=8.0, min_visible_frac=0.75)
    assert [o["label"] for o in strict] == ["chair"], (
        f"门限 0.75 时应只剩完全可见的 chair，实际 {[o['label'] for o in strict]}")
    cup = [o for o in got if o["label"] == "Cup"][0]
    assert 0.3 < cup["visible_frac"] < 0.7, \
        f"Cup 应当只有一半左右可见（实测 {cup['visible_frac']:.2f}），否则门限测试没意义"

    # 量程内没有物体 → 空列表（不是 None）
    assert visible_gt(half_covered_scene, max_range_m=1.0) == []


def test_visible_gt_entries_carry_projection_fields(half_covered_scene):
    """每条结果必须带 `label` / `uv` / `visible_frac` / `range_m` / `box` —— 下游评测全靠它们。"""
    got = visible_gt(half_covered_scene, max_range_m=8.0, min_visible_frac=0.25)
    assert got, "这条测试需要一个非空结果"
    for o in got:
        for key in ("label", "uv", "visible_frac", "range_m", "box"):
            assert key in o, f"visible_gt 的条目缺少 {key}：{sorted(o)}"
        assert isinstance(o["label"], str) and o["label"], f"label 应是非空字符串：{o['label']!r}"
        assert np.shape(o["box"]) == (7,), \
            f"box 应当是 (7,)=[cx,cy,cz,sx,sy,sz,yaw]，实际 {np.shape(o['box'])}"
        assert 0.0 <= o["visible_frac"] <= 1.0, f"visible_frac 越界：{o['visible_frac']}"
        assert o["range_m"] > 0.0, f"range_m 应为正：{o['range_m']}"
        u0, v0, u1, v1 = o["uv"]
        assert 0 <= v0 <= v1 <= H - 1, f"行坐标必须在 [0, {H - 1}] 内：{o['uv']}"
        # 注意：这里**不**断言 u 落在 [0, W-1] —— 跨接缝的框会给出 u1 == W
        # 甚至更大（见模块 docstring 的"已知的可疑行为"），列坐标没做取模。
        assert all(float(x).is_integer() for x in o["uv"]), \
            f"uv 应当是整数像素坐标（要拿去索引图像），实际 {o['uv']}"
        assert "index" in o, "条目里还应有它在 `gt_within` 结果里的下标，便于回溯"


# ==========================================================================
# 9) PanoScene.gt_within：按距离 / 类别筛 GT
# ==========================================================================
def test_gt_within_filters_by_range_from_scene_center(gt_boxes_and_labels):
    """★ `gt_within` 只保留光心附近的 GT 框（GT 覆盖整个房间，全景只看到一部分）。

    注意筛的是"**到光心**"的距离，不是到世界原点的距离 ——
    2D-3D-S 的世界系里光心离原点几十米，写成"到原点"会把所有物体都筛掉。
    """
    boxes, labels = gt_boxes_and_labels
    scene = _scene(_pano(3.0), boxes, labels)
    got, got_labels = scene.gt_within(8.0)
    assert got.shape == (3, 7), f"8 m 内应有 3 个物体，实际 {got.shape}"
    assert got_labels == ["chair", "table", "Cup"], \
        f"顺序应保持 GT 原顺序，实际 {got_labels}"
    assert "monitor" not in got_labels, "12 m 外的物体不该进 8 m 的候选集"

    # 收紧到 3 m：5 m 的 table 也出局
    assert scene.gt_within(3.0)[1] == ["chair", "Cup"]

    # ★ 把光心挪到 100 m 外，同一批框就该全部出局 ——
    #   若实现里忘了减 `center`，这条会红（"到原点距离"在这批框上仍 < 12 m）
    moved = _scene(_pano(3.0, center=C + np.array([100.0, 0.0, 0.0])), boxes, labels)
    assert moved.gt_within(8.0)[1] == [], \
        "光心挪到 100 m 外后所有物体都该超出量程 —— 否则说明用的是到世界原点的距离"


def test_gt_within_class_filter_is_case_insensitive(gt_boxes_and_labels):
    """类别过滤大小写不敏感（GT 里是 `Cup`，用户很可能写 `cup`）。"""
    boxes, labels = gt_boxes_and_labels
    scene = _scene(_pano(3.0), boxes, labels)
    assert scene.gt_within(8.0, ["CHAIR"])[1] == ["chair"], "全大写也要能查到"
    assert scene.gt_within(8.0, ["cUp"])[1] == ["Cup"], "混合大小写也要能查到"
    assert scene.gt_within(8.0, ["chair", "cup"])[1] == ["chair", "Cup"], \
        "多个类别应一起筛"
    assert scene.gt_within(8.0, ["nosuch"])[1] == []
    # `[]`/`None` = 不按类别筛（保持文档语义）
    assert len(scene.gt_within(8.0, [])[1]) == 3
    assert len(scene.gt_within(8.0, None)[1]) == 3


def test_gt_within_empty_result_is_empty_array_not_none(gt_boxes_and_labels):
    """★★ 没匹配到东西时必须返回 `(0, 7)` 数组 + `[]`，**不是 None**。

    下游（`visible_gt`、评测脚本）会直接 `boxes.shape[0]` / `len(labels)`；
    返回 None 的话会在很远的地方以 AttributeError/TypeError 炸掉，
    而现场（"这个房间没有目标类别的物体"）本来是正常情况。
    """
    boxes, labels = gt_boxes_and_labels
    scene = _scene(_pano(3.0), boxes, labels)
    got, got_labels = scene.gt_within(8.0, ["nosuch"])
    assert got is not None, "必须返回空数组，不能是 None"
    assert isinstance(got, np.ndarray) and got.shape == (0, 7), \
        f"空结果应当是 (0, 7) 数组，实际 {got!r}"
    assert got_labels == [], f"空结果的标签列表应当是 []，实际 {got_labels!r}"

    # 完全没有 GT 的场景（`gt_boxes` 保持 None 默认值）同样如此
    no_gt = _scene(_pano(3.0))
    assert no_gt.gt_boxes is None, "这条测试要走 gt_boxes is None 那条分支"
    empty, empty_labels = no_gt.gt_within(8.0)
    assert isinstance(empty, np.ndarray) and empty.shape == (0, 7), \
        f"没有 GT 时也应返回 (0, 7) 数组，实际 {empty!r}"
    assert empty_labels == []

    # 空结果必须能**直接喂给** visible_gt（返回空列表而不是崩）
    assert visible_gt(scene, max_range_m=8.0, classes=["nosuch"]) == []
    assert visible_gt(no_gt, max_range_m=8.0) == []
