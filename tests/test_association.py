# -*- coding: utf-8 -*-
"""`mapping.assoc_strategy`：逐帧最优分配（hungarian）vs 逐观测贪心（greedy）。

为什么要专门测这个开关
====================
`scripts/42_ablate_association.py` 在 12 个真实采集点上测出
**逐帧一对一分配**明显好于旧的逐观测贪心（碎裂率 0.79 → 1.04、
中心误差中位 0.502 → 0.425 m），于是它成了默认。既然默认换了，
就必须有**解析可验证**的单元测试把两件事钉住：

1. `greedy` 路径的行为**一个字都没变**（老配置/老结果还能复现）；
2. `hungarian` 路径确实做到了"一帧内一条轨迹只认领一个观测"，
   并且在贪心会做错的那个构造例子上做对。

测试不碰感知，直接构造 `Observation` 与轨迹，走 `_associate_frame`。
"""
from __future__ import annotations

import numpy as np
import pytest

from roboground.mapping.builder import MapBuilder, _ObjectTrack
from roboground.types import Detection2D, Observation


# ==========================================================================
# 构造用的小工具
# ==========================================================================
class _Cfg(dict):
    """最小可用的 cfg：`MapBuilder` 只用到 `get`。"""

    def get(self, key, default=None):        # noqa: D102
        return dict.get(self, key, default)


def _cfg(**over) -> _Cfg:
    base = {
        "mapping.assoc_radius": 0.60,
        "mapping.assoc_iou": 0.10,
        "mapping.assoc_require_label": True,
        "mapping.assoc_strategy": "hungarian",
        # ⚠️ 与 `configs/default.yaml` 一致：IoU 合并默认**关闭**。
        #    需要它的用例自己显式打开（见下面的 `test_cloud_iou_merge_*`）。
        "mapping.assoc_merge_iou": None,
        "mapping.object_mode": "association",
        "perception.encoder_kwargs.feature_dim": 4,
        "geometry.voxel_size": 0.05,
        "geometry.min_depth": 0.1,
        "geometry.max_depth": 8.0,
        "geometry.depth_scale": 1000.0,
    }
    base.update(over)
    return _Cfg(base)


def _builder(**over) -> MapBuilder:
    b = MapBuilder(_cfg(**over), pipeline=None) if False else MapBuilder.__new__(MapBuilder)
    # 直接手工初始化，避开"要建感知流水线"这件事：
    # 本测试只关心关联，不需要任何编码器/检测器。
    b.cfg = _cfg(**over)
    b.prompts = []
    b.pipeline = None
    b.text_encoder = None
    b.depth_scale = 1000.0
    b.min_depth = 0.1
    b.max_depth = 8.0
    b.ghost_percentile = None
    b.voxel_size = 0.05
    b.assoc_radius = float(b.cfg.get("mapping.assoc_radius", 0.6))
    _iou = b.cfg.get("mapping.assoc_iou", 0.1)
    b.assoc_iou = None if _iou is None else float(_iou)
    b.assoc_require_label = bool(b.cfg.get("mapping.assoc_require_label", True))
    b.assoc_strategy = str(b.cfg.get("mapping.assoc_strategy", "hungarian")).lower()
    _mg = b.cfg.get("mapping.assoc_merge_iou", 0.5)
    b.assoc_merge_iou = None if _mg is None else float(_mg)
    b.object_mode = str(b.cfg.get("mapping.object_mode", "association")).lower()
    b.object_max_points = 10_000
    b.min_observations = 1
    b.grid = None
    b.observations = []
    b.tracks = []
    b.frames_processed = 0
    b.timings = {}
    b._matcher = None
    b.camera_centers = []
    b.drop_stats = {"detections": 0, "no_valid_depth": 0, "low_confidence": 0}
    return b


def _obs(x: float, *, label: str = "chair", frame: str = "f0",
         n: int = 100, spread: float = 0.0) -> Observation:
    """一个中心在 `(x, 0, 3)` 的观测。

    `spread=0`（默认）时点云退化成**一个点**，于是质心精确等于名义坐标，
    距离/得分都能手算。需要测 IoU 分支时才给一个非零 `spread`。
    """
    rng = np.random.default_rng(int(abs(x) * 1000) + 7)
    pts = (np.tile([x, 0.0, 3.0], (n, 1)) if spread == 0.0
           else rng.normal(loc=(x, 0.0, 3.0), scale=spread, size=(n, 3)))
    return Observation(
        detection=Detection2D(label=label, score=1.0,
                              bbox=np.array([0.0, 0.0, 10.0, 10.0]),
                              prompt=label),
        points_world=pts, frame_id=frame)


def _track(b: MapBuilder, center: tuple, *, label: str = "chair",
           track_id: int = 0, spread: float = 0.0) -> _ObjectTrack:
    """造一条已经有 1 个观测的轨迹，中心落在 `center`。"""
    t = _ObjectTrack(track_id, b.feature_dim, b.object_max_points)
    cx, cy, cz = center
    rng = np.random.default_rng(track_id + 101)
    pts = (np.tile([cx, cy, cz], (100, 1)) if spread == 0.0
           else rng.normal(loc=(cx, cy, cz), scale=spread, size=(100, 3)))
    t.add(Observation(
        detection=Detection2D(label=label, score=1.0,
                              bbox=np.array([0.0, 0.0, 10.0, 10.0]), prompt=label),
        points_world=pts, frame_id="seed"))
    return t


# ==========================================================================
# 1) 打分函数：两条路径共用，规则没变
# ==========================================================================
def test_match_score_uses_distance_then_iou():
    # 半径取 0.5：`0.5` 在二进制里是精确值，边界点才不会因浮点误差飘掉
    # （实测：点云取均值后 `0.6` 会变成 `0.600000000000001`，正好卡在门限外）
    b = _builder(**{"mapping.assoc_radius": 0.5, "mapping.assoc_iou": 2.0})
    t = _track(b, (0.0, 0.0, 3.0))
    # 距离 0.25 m、半径 0.5 → 得分 0.5
    assert b._match_score(_obs(0.25), t) == pytest.approx(0.5, abs=1e-6)
    # 距离**正好等于**半径 → 0.0，是合法配对（不是 −1）。
    # 这个区别很重要：`hungarian` 里把它写成 `s > 0` 就会误判成不合法。
    assert b._match_score(_obs(0.5), t) == pytest.approx(0.0, abs=1e-9)
    # 超出半径 → −1
    assert b._match_score(_obs(0.6), t) < 0.0


def test_label_incompatible_is_never_scored_positive():
    b = _builder()
    t = _track(b, (0.0, 0.0, 3.0), label="chair")
    assert b._match_score(_obs(0.05, label="refrigerator"), t) < 0.0
    # 关掉标签约束后又可以关联了
    b2 = _builder(**{"mapping.assoc_require_label": False})
    t2 = _track(b2, (0.0, 0.0, 3.0), label="chair")
    assert b2._match_score(_obs(0.05, label="refrigerator"), t2) > 0.0


def test_assoc_iou_above_one_disables_the_iou_fallback():
    """`assoc_iou > 1` 视为关闭兜底（IoU 恒 ≤ 1）。

    构造：两个**体积很大**的点云，质心相距 ~0.85 m（远超半径 0.6），
    但 AABB 大幅重叠 → 默认参数下 IoU 兜底会把它俩连起来；
    把 `assoc_iou` 设成 2.0（不可能达到）后就必须判为不合法。

    （实测：`spread=0.4` 的两团点在质心相距 0.85 m 时 IoU ≈ 0.34，
    相距 1.3 m 时 IoU ≈ 0.08 —— 后者低于默认阈值 0.10，兜底本来就不会触发，
    所以这个测试必须挑**够近**但**超过半径**的那一档。）
    """
    b_on = _builder()
    b_off = _builder(**{"mapping.assoc_iou": 2.0})
    t_on = _track(b_on, (0.0, 0.0, 3.0), spread=0.4)
    t_off = _track(b_off, (0.0, 0.0, 3.0), spread=0.4)
    far = _obs(0.7, spread=0.4)
    assert float(np.linalg.norm(far.centroid - t_on.center)) > b_on.assoc_radius

    s_off = b_off._match_score(far, t_off)
    assert s_off < 0.0, "关掉兜底后不该还能关联"
    # 开着时：距离不达标，但 AABB 重叠 → 走 IoU 分支，得分为正
    assert b_on._match_score(far, t_on) > 0.0


# ==========================================================================
# 2) 贪心路径行为不变（老配置能复现老结果）
# ==========================================================================
def test_greedy_lets_one_track_absorb_two_observations_in_the_same_frame():
    b = _builder(**{"mapping.assoc_strategy": "greedy"})
    b.tracks = [_track(b, (0.0, 0.0, 3.0), track_id=0)]
    a, c = _obs(0.10, frame="f0"), _obs(-0.10, frame="f0")
    out = b._associate_frame([a, c])
    assert len(b.tracks) == 1
    assert out == [b.tracks[0], b.tracks[0]]
    assert b.tracks[0].num_observations == 3        # 种子 1 + 本帧 2


def test_hungarian_forces_one_observation_per_track_per_frame():
    """同样的输入，`hungarian` 会把第二个观测推到**另一条**轨迹上。

    这就是实测里"碎裂率 0.79 → 1.04"的来源：两个真的不同的物体
    （或同一物体被拆成的两段）不会再被并进同一条轨迹。
    """
    b = _builder(**{"mapping.assoc_strategy": "hungarian"})
    b.tracks = [_track(b, (0.0, 0.0, 3.0), track_id=0)]
    a, c = _obs(0.10, frame="f0"), _obs(-0.10, frame="f0")
    out = b._associate_frame([a, c])
    assert len(b.tracks) == 2
    assert out[0] is not out[1]
    assert b.tracks[0].num_observations + b.tracks[1].num_observations == 3


# ==========================================================================
# 3) 全局最优确实优于贪心（教科书式的"悔值"构造）
# ==========================================================================
def test_hungarian_beats_greedy_on_a_regret_case():
    """构造一个贪心必然做错的例子，并量化它把**物体中心**拉偏了多少。

    几何（半径 0.6，得分 = 1 − d/0.6）：

    · 轨迹 T1 在 `x=0`（这是"真物体"），轨迹 T2 在 `x=1.0`
    · 观测 B 在 `x=0.05`：属于 T1 的那个物体（得分 0.917），
      且它离 T2 有 0.95 m → **超半径，进不了 T2**
    · 观测 A 在 `x=0.45`：到 T1 0.45（得分 0.25）、到 T2 0.55（得分 0.083），
      两个都合法，但它**更想进 T1**

    贪心（先 A 后 B）：A 挑 T1，B 也只能进 T1 → T1 同时装下"真物体"和
    0.45 m 外的另一个观测，轨迹中心被拉到 0.167 m，T2 什么也没得到。
    最优一对一：B→T1、A→T2 → T1 的中心只被拉动 0.025 m。

    ⚠️ 这里**不能**拿"总得分"比高低：一对一模型把可行解集合**缩小**了，
    它的最优总得分反而可能更低（0.917+0.083 < 0.25+0.917）。
    两种模型比的是**出来的轨迹对不对**，不是打分高低。
    """
    b_g = _builder(**{"mapping.assoc_strategy": "greedy"})
    b_h = _builder(**{"mapping.assoc_strategy": "hungarian"})
    for b in (b_g, b_h):
        b.tracks = [_track(b, (0.0, 0.0, 3.0), track_id=0),
                    _track(b, (1.0, 0.0, 3.0), track_id=1)]

    A, B = _obs(0.45, frame="f0"), _obs(0.05, frame="f0")
    truth = np.array([0.0, 0.0, 3.0])

    # 先用**同一批对象**把前提核一遍（不然下面的结论可能只是构造错了）
    t1, t2 = b_g.tracks
    sA1, sA2 = b_g._match_score(A, t1), b_g._match_score(A, t2)
    sB1, sB2 = b_g._match_score(B, t1), b_g._match_score(B, t2)
    assert sA1 > sA2 > 0, "A 应当更想进 T1，但仍能进 T2"
    assert sB1 > 0 > sB2, "B 只能进 T1（它离 T2 超过半径）"

    # 贪心：A 和 B 落进同一条轨迹
    out_g = b_g._associate_frame([A, B])
    assert out_g[0] is out_g[1]
    # 最优：两条观测分开
    out_h = b_h._associate_frame([A, B])
    assert out_h[0] is not out_h[1]

    err_g = float(np.linalg.norm(b_g.tracks[0].center - truth))
    err_h = float(np.linalg.norm(b_h.tracks[0].center - truth))
    assert err_g == pytest.approx(0.1667, abs=1e-3)
    assert err_h == pytest.approx(0.0250, abs=1e-3)
    assert err_h < err_g / 5.0


# ==========================================================================
# 4) 标签不兼容时不能"硬塞"，只能新建轨迹
# ==========================================================================
def test_incompatible_label_creates_a_new_track_under_both_strategies():
    for strat in ("greedy", "hungarian"):
        b = _builder(**{"mapping.assoc_strategy": strat})
        b.tracks = [_track(b, (0.0, 0.0, 3.0), track_id=0, label="chair")]
        out = b._associate_frame([_obs(0.02, label="refrigerator", frame="f0")])
        assert len(b.tracks) == 2, strat
        assert out[0] is b.tracks[1], strat


# ==========================================================================
# 5) 空输入 / 聚类模式
# ==========================================================================
def test_empty_frame_is_a_noop():
    for strat in ("greedy", "hungarian"):
        b = _builder(**{"mapping.assoc_strategy": strat})
        assert b._associate_frame([]) == []
        assert b.tracks == []


def test_clustering_mode_bypasses_the_strategy_switch():
    for strat in ("greedy", "hungarian"):
        b = _builder(**{"mapping.assoc_strategy": strat,
                        "mapping.object_mode": "clustering"})
        out = b._associate_frame([_obs(0.0, frame="f0"), _obs(1.0, frame="f0")])
        assert len(b.tracks) == 1
        assert out[0] is out[1] is b.tracks[0]


# ==========================================================================
# 6) 跨帧：下一帧又是干净的一对一
# ==========================================================================
def test_one_to_one_applies_per_frame_not_globally():
    b = _builder(**{"mapping.assoc_strategy": "hungarian"})
    b.tracks = [_track(b, (0.0, 0.0, 3.0), track_id=0)]
    b._associate_frame([_obs(0.05, frame="f0"), _obs(-0.05, frame="f0")])
    assert len(b.tracks) == 2
    # 下一帧：两条轨迹各认领一个观测，仍然只有 2 条轨迹
    b._associate_frame([_obs(0.04, frame="f1"), _obs(-0.04, frame="f1")])
    assert len(b.tracks) == 2
    assert sum(t.num_observations for t in b.tracks) == 5      # 1 种子 + 4


# ==========================================================================
# 7) 同帧重复检测的合并（等距柱状跨接缝物体的两段）
# ==========================================================================
def _dup(x: float, *, label: str = "table", frame: str = "f0",
         jitter: float = 0.02) -> Observation:
    """造一个"和 `_obs(x)` 几乎同一团点云"的检测（模拟跨接缝被拆出的第二段）。"""
    rng = np.random.default_rng(int(abs(x) * 1000) + 7)
    pts = rng.normal(loc=(x, 0.0, 3.0), scale=jitter, size=(100, 3))
    return Observation(
        detection=Detection2D(label=label, score=1.0,
                              bbox=np.array([0.0, 0.0, 10.0, 10.0]), prompt=label),
        points_world=pts, frame_id=frame)


def test_same_frame_duplicates_are_merged_before_assignment():
    """一个物体在一帧里出现两次（跨接缝被拆成两段）→ 只能算**一个**物体。

    这正是等距柱状全景的实际情况：`uv_parts` 会把跨 ±180° 的物体拆成
    两个 bbox。若不合并，一对一分配会把它算成两个物体
    （实测 office_6：4 个跨接缝物体 → 恰好 4 次计数错误）。
    """
    W = 2048
    b = _builder(**{"mapping.assoc_strategy": "hungarian"})
    a = _seg(0.5, (1900, 40, W - 1, 344), label="table")
    c = _seg(0.5, (0, 40, 800, 344), label="table")     # 同一物体的另一段
    out = b._associate_frame([a, c], image_width=W)
    assert len(b.tracks) == 1, "同帧重复检测没有被合并"
    # 合并后**只剩一个观测**参与分配，所以返回值长度为 1（不是 2）
    assert out == [b.tracks[0]]
    assert b.tracks[0].num_observations == 1     # 合并后只算一次观测
    assert b.tracks[0].points_count == 100       # 点云是两段之和


def test_cloud_iou_merge_is_off_by_default():
    """IoU 合并**默认关闭**：并排的两个真不同物体包围盒本来就会重叠。

    实测依据（`scripts/42`，12 个真实采集点）：把"同标签 + 点云 IoU ≥ 0.5"
    的检测也合并，针孔臂命中率 98.1% → 96.7%。所以默认只保留接缝判据。
    """
    b = _builder(**{"mapping.assoc_strategy": "hungarian"})
    assert b.assoc_merge_iou is None
    # 两团完全重合的点云（IoU ≈ 1）：默认下**不**合并
    b._associate_frame([_dup(0.5), _dup(0.5)])
    assert len(b.tracks) == 2

    # 显式打开后才合并（这是复现那次测量的开关）
    on = _builder(**{"mapping.assoc_strategy": "hungarian",
                     "mapping.assoc_merge_iou": 0.5})
    on._associate_frame([_dup(0.5), _dup(0.5)])
    assert len(on.tracks) == 1


def test_same_frame_merge_respects_labels():
    """不同标签即使点云完全重合也不许合并（桌子底下的椅子）。"""
    b = _builder(**{"mapping.assoc_strategy": "hungarian"})
    a = _dup(0.5, label="table")
    c = _dup(0.5, label="chair")
    b._associate_frame([a, c])
    assert len(b.tracks) == 2
    assert sorted(t.label for t in b.tracks) == ["chair", "table"]


def test_same_frame_merge_can_be_disabled_and_needs_overlap():
    """关闭接缝合并（不传 `image_width`）后不合并；IoU 不够也不合并。"""
    off = _builder(**{"mapping.assoc_strategy": "hungarian"})
    W = 2048
    a = _seg(0.5, (1900, 40, W - 1, 344), label="table")
    c = _seg(0.5, (0, 40, 800, 344), label="table")
    off._associate_frame([a, c])                  # 不给 image_width → 不判接缝
    assert len(off.tracks) == 2

    # 点云错开 1 m（IoU≈0）→ 即使开着 IoU 合并也是两个物体
    b = _builder(**{"mapping.assoc_strategy": "hungarian",
                    "mapping.assoc_merge_iou": 0.5})
    b.assoc_radius = 3.0        # 放宽半径，保证它们会被分配到两条不同轨迹
    b._associate_frame([_dup(0.5), _dup(1.5)])
    assert len(b.tracks) == 2


def test_cloud_iou_is_one_for_identical_and_zero_for_disjoint():
    p = np.tile([1.0, 2.0, 3.0], (10, 1))
    q = np.tile([9.0, 9.0, 9.0], (10, 1))
    assert MapBuilder._cloud_iou(p, p) == pytest.approx(0.0)   # 退化盒子体积 0
    assert MapBuilder._cloud_iou(p, q) == pytest.approx(0.0)
    assert MapBuilder._cloud_iou(np.zeros((0, 3)), p) == 0.0
    big = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert MapBuilder._cloud_iou(big, big) == pytest.approx(1.0)


# ==========================================================================
# 8) ★ 等距柱状接缝：两段在 3D 里可能位于**房间两侧**，只能按图像判
# ==========================================================================
def _seg(x: float, bbox, *, label: str = "ceiling", frame: str = "f0"
         ) -> Observation:
    """造一个"贴在图像某一侧边界上"的检测（模拟跨 ±180° 被拆的段）。"""
    pts = np.tile([x, 0.0, 3.0], (50, 1))
    return Observation(
        detection=Detection2D(label=label, score=1.0,
                              bbox=np.asarray(bbox, dtype=np.float64), prompt=label),
        points_world=pts, frame_id=frame)


def test_seam_split_parts_are_merged_even_though_3d_iou_is_zero():
    """天花板这类"又宽又靠极点"的物体，被拆开的两段在 3D 里几乎不相交。

    实测（office_6 全景）：4 个跨接缝物体（table / window / wall / ceiling）
    造成 4 次计数错误。若只用 3D IoU 判合并，这 4 个**一个都合不上**，
    因为两段落在房间的两侧、点云 IoU ≈ 0。所以必须同时用图像接缝判据。
    """
    W = 2048
    left = _seg(-8.0, (1900, 40, W - 1, 344))     # 贴右边界
    right = _seg(+8.0, (0, 40, 1312, 344))        # 贴左边界
    assert MapBuilder._cloud_iou(left.points_world,
                                 right.points_world) == pytest.approx(0.0)

    b = _builder(**{"mapping.assoc_strategy": "hungarian"})
    b._associate_frame([left, right], image_width=W)
    assert len(b.tracks) == 1, "跨接缝的两段没有被合并"
    assert b.tracks[0].points_count == 100


def test_seam_merge_needs_both_borders_and_vertical_overlap():
    W = 2048
    # 只贴右边界、另一段在画面中间 → 不是接缝，不合并
    b = _builder(**{"mapping.assoc_strategy": "hungarian"})
    b.assoc_merge_iou = 2.0                     # 关掉 IoU 判据，只测接缝判据
    b.assoc_radius = 50.0
    b._associate_frame([_seg(0.0, (1900, 40, W - 1, 344)),
                        _seg(1.0, (100, 40, 300, 344))], image_width=W)
    assert len(b.tracks) == 2

    # 竖直区间不重叠（上下半图各一个）→ 也不是同一个框
    b2 = _builder(**{"mapping.assoc_strategy": "hungarian"})
    b2.assoc_merge_iou = 2.0
    b2.assoc_radius = 50.0
    b2._associate_frame([_seg(0.0, (1900, 40, W - 1, 200)),
                         _seg(1.0, (0, 300, 1312, 500))], image_width=W)
    assert len(b2.tracks) == 2


def test_seam_merge_still_requires_label_compatibility():
    W = 2048
    b = _builder(**{"mapping.assoc_strategy": "hungarian"})
    b.assoc_merge_iou = 2.0
    b._associate_frame([_seg(0.0, (1900, 40, W - 1, 344), label="wall"),
                        _seg(1.0, (0, 40, 1312, 344), label="ceiling")],
                       image_width=W)
    assert len(b.tracks) == 2


def test_wraps_seam_symmetry_and_edge_cases():
    W = 100
    a = _seg(0.0, (90, 10, W - 1, 50))
    b = _seg(1.0, (0, 10, 30, 50))
    assert MapBuilder._wraps_seam(a, b, W) is True
    assert MapBuilder._wraps_seam(b, a, W) is True          # 顺序无关
    assert MapBuilder._wraps_seam(a, a, W) is False         # 都贴右边不算
    assert MapBuilder._wraps_seam(a, b, 1) is False         # 非法宽度
    # 脏数据防御：bbox 长度不足 4 时不能炸。
    # （`Detection2D.__post_init__` 已经把 bbox 规整成 (4,)，所以正常的
    #  `Observation` 走不到这里；这条用一个最小替身把这个分支钉住。）
    class _Stub:
        bbox = np.zeros(2)

    class _StubObs:
        detection = _Stub()

    assert MapBuilder._wraps_seam(a, _StubObs(), W) is False
    assert MapBuilder._wraps_seam(_StubObs(), a, W) is False
