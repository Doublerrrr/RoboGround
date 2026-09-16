"""空间关系计算：把两个 3D 物体的几何关系转成**带米制证据**的语言描述。

世界系约定（全项目统一）
-----------------------
`x` 向右、`y` 向前（机器人前方/深度方向）、`z` 向上。

关系判定策略：**主轴主导**
------------------------
两个物体在三个轴上都有分离量，但人描述位置时只会说一个主导方向
（"杯子在桌子**上面**"，而不是"杯子在桌子的上面偏右一点"）。
所以这里先算三个轴的归一化分离量，取绝对值最大的那个轴作为主导关系，
再用该轴的符号决定方向。

"归一化"用的是**两个物体的尺寸之和**，而不是 1 米这种固定阈值 ——
否则"两个大柜子相距 30cm"会被判成同一位置，而"两个小杯子相距 30cm"
却是"离得很远"。用尺寸做尺度，判断才符合直觉。

"inside"（包含）单独处理：用包围盒包含比例判定，不参与主轴竞争。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roboground.types import SpatialRelation

#: 支持的空间关系（英文名 → 中文名，供话术与测试使用）
RELATION_NAMES: Dict[str, str] = {
    "above": "上面",
    "below": "下面",
    "left_of": "左边",
    "right_of": "右边",
    "in_front_of": "前面",
    "behind": "后面",
    "inside": "里面",
    "near": "附近",
    "far": "远处",
    "overlapping": "重叠",
    "unknown": "未知",
}

#: 默认阈值
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "above_delta": 0.10,      # 米：z 差超过它才算"上/下"
    "inside_ratio": 0.60,     # 包围盒包含比例超过它才算"里面"
    "near_threshold": 0.50,   # 米：中心距离小于它算"附近"
    "axis_ratio": 0.15,       # 归一化分离量低于它认为"没有明显方向"
    "adjacent_gap": 0.15,     # 米：水平方向表面间隙小于它算"旁边"
}


# ==========================================================================
# 基础度量
# ==========================================================================
def center_distance(a: Any, b: Any) -> float:
    """两个物体中心的欧氏距离（米）。"""
    ca = np.asarray(a.center, dtype=np.float64).reshape(3)
    cb = np.asarray(b.center, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(ca - cb))


def bbox_gap(a: Any, b: Any) -> float:
    """两个 AABB 之间的**最小间隙**（米）；相交时为 0。

    这比中心距离更贴近"我离那个东西还有多远"的物理含义 ——
    两个大柜子中心相距 2m，但表面可能只差 10cm。
    """
    a_lo, a_hi = np.asarray(a.bbox_min, float), np.asarray(a.bbox_max, float)
    b_lo, b_hi = np.asarray(b.bbox_min, float), np.asarray(b.bbox_max, float)
    gap = np.maximum(0.0, np.maximum(a_lo - b_hi, b_lo - a_hi))
    return float(np.linalg.norm(gap))


def containment_ratio(inner: Any, outer: Any) -> float:
    """`inner` 的中心所在处，被 `outer` 包围盒覆盖的比例（用于判定 inside）。

    实现：把 inner 的包围盒切成 3×3×3 的采样点，统计落在 outer 内的比例。
    比单纯判"中心是否在内"更鲁棒（能处理半个物体探出来的情况）。
    """
    i_lo, i_hi = np.asarray(inner.bbox_min, float), np.asarray(inner.bbox_max, float)
    o_lo, o_hi = np.asarray(outer.bbox_min, float), np.asarray(outer.bbox_max, float)

    lo = np.minimum(i_lo, i_hi)
    hi = np.maximum(i_lo, i_hi)
    axes = [np.linspace(lo[k], hi[k], 3) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    inside = np.all((grid >= o_lo[None, :]) & (grid <= o_hi[None, :]), axis=1)
    return float(inside.mean())


# ==========================================================================
# 主导关系
# ==========================================================================
def dominant_relation(
    a: Any,
    b: Any,
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[str, Dict[str, float]]:
    """判定 a 相对 b 的主导空间关系。

    Returns
    -------
    (relation, evidence)
        `relation` ∈ `RELATION_NAMES`；`evidence` 里是各轴的分离量（米），
        便于在输出里给出可核查的证据。
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    ca = np.asarray(a.center, dtype=np.float64).reshape(3)
    cb = np.asarray(b.center, dtype=np.float64).reshape(3)
    delta = ca - cb

    # 尺寸尺度：两个物体在各轴上的平均尺寸（防止除 0）
    size_a = np.abs(np.asarray(a.bbox_max, float) - np.asarray(a.bbox_min, float))
    size_b = np.abs(np.asarray(b.bbox_max, float) - np.asarray(b.bbox_min, float))
    scale = np.clip((size_a + size_b) / 2.0, 1e-3, None)

    norm = delta / scale                        # 各轴归一化分离量
    evidence = {
        "dx": float(delta[0]), "dy": float(delta[1]), "dz": float(delta[2]),
        "norm_x": float(norm[0]), "norm_y": float(norm[1]), "norm_z": float(norm[2]),
        "center_distance": center_distance(a, b),
        "bbox_gap": bbox_gap(a, b),
        "containment": containment_ratio(a, b),
    }

    # 1) 包含关系优先（不参与主轴竞争）
    if evidence["containment"] >= th["inside_ratio"]:
        return "inside", evidence

    # 1.5) ★ 竖直方向**完全不重叠** → "上/下"没有歧义，直接判，不参与主轴竞争
    #
    # 为什么必须放在主轴竞争**之前**：主轴竞争的判据是"归一化分离量最大的
    # 那个轴"，而归一化用的是**物体尺寸**。对横跨整个房间的结构
    # （天花板 / 墙 / 地板），它与小物体的**中心**在水平方向差得很远，
    # 归一化之后 x/y 轴反而占优 —— 于是"天花板在门上面吗"会被答成
    # "不是：天花板在门的左边"。实测（`scripts/43`，office_6 全景）就是这么错的。
    # 而"A 的最低点高过 B 的最高点"是人类语义里**没有歧义**的"上面"。
    # ⚠️ 取的是 **z 轴**（索引 2）的上下界，不是"所有轴的最小值"。
    a_lo = float(np.asarray(a.bbox_min, dtype=np.float64).reshape(3)[2])
    a_hi = float(np.asarray(a.bbox_max, dtype=np.float64).reshape(3)[2])
    b_lo = float(np.asarray(b.bbox_min, dtype=np.float64).reshape(3)[2])
    b_hi = float(np.asarray(b.bbox_max, dtype=np.float64).reshape(3)[2])
    if a_lo >= b_hi - 1e-9:
        return "above", evidence
    if a_hi <= b_lo + 1e-9:
        return "below", evidence

    # 2) 主轴竞争
    axis = int(np.argmax(np.abs(norm)))
    strength = float(abs(norm[axis]))

    if strength < th["axis_ratio"]:
        # 没有明显方向：退化为"附近 / 重叠"
        if evidence["bbox_gap"] <= 1e-6:
            return "overlapping", evidence
        if evidence["center_distance"] <= th["near_threshold"]:
            return "near", evidence
        return "unknown", evidence

    positive = norm[axis] > 0
    if axis == 2:                               # z 轴
        # 竖直方向额外要求绝对高度差达标，避免"贴着桌面"被判成 above
        if abs(evidence["dz"]) < th["above_delta"]:
            return ("near" if evidence["center_distance"] <= th["near_threshold"]
                    else "unknown"), evidence
        return ("above" if positive else "below"), evidence
    # ---- 水平轴：先看是不是"贴在一起"（旁边），再看方向 ----
    # 为什么只在水平轴做这个判断？
    # 因为"上/下"是接触关系（杯子放在桌上，间隙就是 0），如果对竖直方向也
    # 用"间隙小 = 旁边"，那"杯子在桌子上面"会被误判成"杯子在桌子旁边"。
    # 水平方向才是人类口语里"旁边/挨着"的含义。
    if evidence["bbox_gap"] <= th["adjacent_gap"]:
        return "near", evidence

    if axis == 0:                               # x 轴（右为正）
        return ("right_of" if positive else "left_of"), evidence
    # y 轴（前为正）
    return ("in_front_of" if positive else "behind"), evidence


def compute_relation(
    a: Any,
    b: Any,
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> SpatialRelation:
    """构造一个 `SpatialRelation` 对象（a 相对 b）。"""
    relation, evidence = dominant_relation(a, b, thresholds=thresholds)
    label_a = getattr(a, "label", "object")
    label_b = getattr(b, "label", "object")
    return SpatialRelation(
        subject=str(label_a),
        relation=relation,
        object=str(label_b),
        distance=center_distance(a, b),
        evidence={k: float(v) for k, v in evidence.items()},
    )


def compute_all_relations(
    objects: Sequence[Any],
    *,
    thresholds: Optional[Dict[str, float]] = None,
    max_distance: Optional[float] = 3.0,
    same_label: bool = True,
) -> List[SpatialRelation]:
    """计算所有物体两两之间的关系。

    Parameters
    ----------
    max_distance
        超过该中心距离的物体对不生成关系（避免 N² 爆炸且大多无意义）。
    same_label
        是否也计算同类物体之间的关系（如"椅子在椅子旁边"），默认计算。
    """
    out: List[SpatialRelation] = []
    objs = list(objects)
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            if not same_label and getattr(a, "label", None) == getattr(b, "label", None):
                continue
            d = center_distance(a, b)
            if max_distance is not None and d > max_distance:
                continue
            out.append(compute_relation(a, b, thresholds=thresholds))
            out.append(compute_relation(b, a, thresholds=thresholds))
    return out


def find_relations(
    relations: Iterable[SpatialRelation],
    *,
    subject: Optional[str] = None,
    relation: Optional[Any] = None,
    object_: Optional[str] = None,
) -> List[SpatialRelation]:
    """按条件过滤关系列表（支持字符串或可调用对象作为条件）。"""
    def _match(value: Any, cond: Any) -> bool:
        if cond is None:
            return True
        if callable(cond):
            return bool(cond(value))
        return str(value).lower() == str(cond).lower()

    return [
        r for r in relations
        if _match(r.subject, subject)
        and _match(r.relation, relation)
        and _match(r.object, object_)
    ]


def describe_relations(
    relations: Sequence[SpatialRelation],
    *,
    top: int = 8,
    use_chinese: bool = True,
) -> str:
    """把关系列表渲染成人类可读文本（供 VLM 上下文与 demo 输出）。"""
    lines: List[str] = []
    for r in list(relations)[:top]:
        name = RELATION_NAMES.get(r.relation, r.relation) if use_chinese else r.relation
        lines.append(f"{r.subject} 在 {r.object} 的{name}（中心距 {r.distance:.2f} m）")
    return "\n".join(lines)
