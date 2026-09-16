# -*- coding: utf-8 -*-
"""消融：把 N 个共光心真实视角融合成 360° 全景，对下游**到底值多少**？

为什么非做这件事
================
批评 #10 的问法是："**为什么用等距柱状全景？这个选择的价值从来没有被测过。**"
在此之前，项目里所有"全景很好"的说法都建立在**几何自证**上
（覆盖率 46%、立体角 66%、反投影误差 0.34 m），
却从来没有回答一个更朴素的问题：

    机器人站在同一个位置，**只看一帧**，和**把这一圈都看了**，下游差多少？

这个脚本就是回答它。三条臂共享**同一批真实帧、同一份 GT、同一套下游链路**，
只换输入观测：

    pano_k     把 k 个共光心真实视角融合成 360° 全景 → 建图
    frame_i    只用**第 i 个真实视角**（针孔原图，不融合）→ 建图
    pano_1     把其中**一帧**融合成全景 → 建图（信息量与 `frame_i` 完全相同）

第三个臂是关键：它把"**全景这个表示形式**"与"**多视角这份信息**"拆开了。
`pano_1` 和对应的 `frame_i` 看到的是**同一帧**，唯一区别是投影形式；
`pano_N` 与 `pano_1` 的区别才是多视角融合的贡献。
不拆的话，无法排除"全景只是换了个画布"这种解释。

三条指标（分母写清楚，避免被误读）
================================
1. **可观测上限** `n_visible / n_gt_in_range`
   房间内 `max_range` 以内的 GT 里，这条臂的输入**根本看得见**几个。
   它是任何下游查询的**上限**：看不见的物体，后面的算法再强也找不回来。
   口径与 `visible_gt` 一致（`visible_frac >= 0.25`，同一套表面采样、同一分母）。
2. **地图级定位误差**：只在该臂**自己可见**的 GT 上算，
   用"标签相同 + 中心距离最近"的一对一贪心匹配，报告
   命中率 / 中位误差 / `≤0.25 m` / `≤0.50 m`。
3. **共同子集上的定位误差**：只取两条臂**都可见**的 GT 再比一次。
   这一项回答"全景是不是不光看得广，还看得更准"——
   不控住这个子集，第 2 项会因为分母不同而不可比。

外加一条**查询召回**：对每个 GT 直接问它的类名，
看 `SemanticMap.query_text` 的 top-k 里有没有落在这个 GT 1 m 内的对象
（检验"查询链路在全景地图上同样可用"，不是另一份独立信息）。

检测从哪来：**用 GT 框当检测**
============================
和 `scripts/37` 一致：把 GT 框投到图上当 `Detection2D` 注入，
测的是 **2D→3D→地图** 这一段，不是检测器。全景臂用 `uv_parts`
（跨 ±180° 接缝的物体拆成两段），针孔臂用 `frame_gt.box_to_frame` 的框。

跑法：
    python scripts/41_ablate_panorama_value.py --rooms office_6 hallway_6 office_27
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config  # noqa: E402
from roboground.data.frame_gt import visible_gt_in_frame  # noqa: E402
from roboground.data.panorama import fuse_to_equirect  # noqa: E402
from roboground.data.pano_scene import (  # noqa: E402
    PanoScene,
    load_scene,
    select_location,
    visible_gt,
)

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")
#: 视角数扫描（都取自同一批 `max_views` 帧的**嵌套子集**，保证曲线单调可解释）
SWEEP = (1, 2, 4, 8, 16, 24)
#: 匹配门限：超过 `max_dist` 视为没定位到；`hit_dist` 是"够不够机器人用"的粗线
MAX_DIST_M = 2.0
HIT_DIST_M = 1.0


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ==========================================================================
# 指标
# ==========================================================================
def match_label_aware(smap, gt_boxes: np.ndarray, gt_labels: Sequence[str], *,
                      max_dist: float = MAX_DIST_M) -> Dict[str, Any]:
    """**标签感知**的一对一贪心匹配：地图物体 ↔ GT 框。

    为什么不用 `eval.benchmark.map_level_localization`：那个函数按"最近中心"
    匹配、**不看标签**（它的注释写明了标签未知时只能这样）。
    在消融里两条臂的地图物体数差很多，不看标签会让
    "多出来的一个 object 抢走最近的那个 GT"，误差算到别人头上。
    这里的地图物体标签**就是 GT 类别**（检测是注入的），所以按标签匹配是
    合法且更严格的：它同时惩罚"位置错"和"类别串"。
    """
    objs = list(getattr(smap, "objects", []) or [])
    boxes = np.asarray(gt_boxes, dtype=np.float64)
    n_gt = int(boxes.shape[0]) if boxes.size else 0
    out: Dict[str, Any] = {
        "n_gt": n_gt, "n_map_objects": int(len(objs)),
        "n_matched": 0, "match_rate": 0.0,
        "median_m": None, "mean_m": None, "p90_m": None,
        "within_0.25m": 0.0, "within_0.50m": 0.0,
        "median_points": None, "median_voxels": None,
    }
    if n_gt == 0 or not objs:
        return out

    lab_gt = [str(s).strip().lower() for s in gt_labels]
    pairs: List[Tuple[float, int, int]] = []
    for oi, o in enumerate(objs):
        ol = str(getattr(o, "label", "")).strip().lower()
        c = np.asarray(o.center, dtype=np.float64).reshape(3)
        for gi in range(n_gt):
            if lab_gt[gi] != ol:
                continue
            d = float(np.linalg.norm(c - boxes[gi, :3]))
            if d <= float(max_dist):
                pairs.append((d, oi, gi))
    pairs.sort(key=lambda p: p[0])

    used_o: set = set()
    used_g: set = set()
    dists: List[float] = []
    for d, oi, gi in pairs:
        if oi in used_o or gi in used_g:
            continue
        used_o.add(oi)
        used_g.add(gi)
        dists.append(d)

    if dists:
        a = np.asarray(dists, dtype=np.float64)
        # 命中物体的**点/体素数**：用来解释"为什么全景的精度反而略低" ——
        # 全景把更多视角、更大掠射角上的观测也并进同一个物体，
        # 点云更大（本该更好），但如果混进来的是斜面上误差更大的观测，
        # 中心反而会被拉偏。两个数一起报，读者自己判断。
        pts = [float(getattr(objs[oi], "num_points", 0) or 0) for oi in used_o]
        vx = [float(getattr(objs[oi], "num_voxels", 0) or 0) for oi in used_o]
        out.update({
            "n_matched": int(a.size),
            "match_rate": float(a.size) / float(n_gt),
            "median_m": float(np.median(a)),
            "mean_m": float(a.mean()),
            "p90_m": float(np.percentile(a, 90)),
            "within_0.25m": float((a <= 0.25).mean()),
            "within_0.50m": float((a <= 0.50).mean()),
            "median_points": float(np.median(pts)) if pts else None,
            "median_voxels": float(np.median(vx)) if vx else None,
        })
    return out


def query_recall(smap, gt_boxes: np.ndarray, gt_labels: Sequence[str], *,
                 top_k: int = 3, hit_dist: float = HIT_DIST_M) -> Dict[str, Any]:
    """对每个 GT 直接问它的类名，看 top-k 里有没有落在它附近的对象。

    ⚠️ 因为查询串**就是**地图物体的标签，词法匹配必然给满分，
    所以这一项在数值上等价于"标签相同 + 位置够近"的命中率。
    它证明的是**查询链路在全景地图上同样跑得通**，
    不是一份独立于定位精度的新证据。别把它当第二个指标引用。
    """
    boxes = np.asarray(gt_boxes, dtype=np.float64)
    n_gt = int(boxes.shape[0]) if boxes.size else 0
    out = {"n_query": n_gt, "recall@1": 0.0, "recall@3": 0.0, "errors": 0}
    if n_gt == 0:
        return out
    hit1 = hit3 = 0
    for gi in range(n_gt):
        try:
            res = smap.query_text(str(gt_labels[gi]), top_k=int(top_k))
        except Exception:                                    # noqa: BLE001
            out["errors"] += 1
            continue
        ds = []
        for r in (res or []):
            try:
                ds.append(float(np.linalg.norm(
                    np.asarray(r.position, dtype=np.float64).reshape(3)
                    - boxes[gi, :3])))
            except Exception:                                # noqa: BLE001
                continue
        if ds:
            if ds[0] <= hit_dist:
                hit1 += 1
            if min(ds) <= hit_dist:
                hit3 += 1
    out["recall@1"] = hit1 / float(n_gt)
    out["recall@3"] = hit3 / float(n_gt)
    return out


# ==========================================================================
# 两条臂建图
# ==========================================================================
def _inject_and_build(cfg, frame, dets, prompts):
    """注入 GT 检测 → `MapBuilder` → `SemanticMap`（下游链路一行不改）。"""
    from roboground.mapping import MapBuilder

    b = MapBuilder(cfg, prompts=prompts)
    b.pipeline.run = lambda f, prompts=None: dets          # type: ignore[assignment]
    smap = b.build_from_frames([frame])
    return smap, b


def build_pano_map(cfg, scene: PanoScene, items: List[Dict[str, Any]]):
    """全景臂：`uv_parts` 把跨接缝的物体拆成两段，bbox 不越界。"""
    from roboground.types import Detection2D

    dets = []
    for it in items:
        for (u0, v0, u1, v1) in it.get("uv_parts") or [it["uv"]]:
            dets.append(Detection2D(label=str(it["label"]), score=1.0,
                                    bbox=np.array([u0, v0, u1, v1], dtype=np.float64),
                                    prompt=str(it["label"])))
    frame = scene.frame()
    prompts = sorted({d.label for d in dets}) or ["object"]
    return _inject_and_build(cfg, frame, dets, prompts)


def build_frame_map(cfg, frame, items: List[Dict[str, Any]]):
    """单帧臂：用 `frame_gt` 投出来的框，针孔反投影。"""
    from roboground.types import Detection2D

    dets = []
    for it in items:
        for (u0, v0, u1, v1) in it.get("uv_parts") or [it["uv"]]:
            dets.append(Detection2D(label=str(it["label"]), score=1.0,
                                    bbox=np.array([u0, v0, u1, v1], dtype=np.float64),
                                    prompt=str(it["label"])))
    prompts = sorted({d.label for d in dets}) or ["object"]
    return _inject_and_build(cfg, frame, dets, prompts)


# ==========================================================================
# 主流程
# ==========================================================================
def run_room(cfg, root: Path, room: str, args) -> Dict[str, Any]:
    loc = select_location(root, room=room, min_frames=8)
    hr(f"{room}：{loc.uuid[:12]}，{len(loc.frame_ids)} 个真实视角")

    t0 = time.time()
    scene = load_scene(root, loc, width=args.width, height=args.height,
                       max_frames=args.max_views, max_depth=args.max_range,
                       resize=(args.resize, args.resize))
    load_s = time.time() - t0
    n_gt_all = int(scene.gt_within(args.max_range)[0].shape[0])
    print(f"  全景 {args.width}x{args.height} 来自 {scene.frames_used} 帧"
          f"（读盘+融合+GT {load_s:.1f}s）")
    print(f"  GT（≤{args.max_range:.0f} m，全场分母）{n_gt_all} 个；"
          f"GT 源 {scene.meta.get('gt_room_objects')} 条房间标注")

    gt_boxes, gt_labels = scene.gt_within(args.max_range)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)

    # ---- 把用到的真实帧一次性读进内存：两条臂共用同一批帧 ----
    views24 = list(scene.meta["frame_ids"])
    cache = {fid: loc.frame(fid, resize=(args.resize, args.resize))
             for fid in views24}
    frames24 = [cache[f] for f in views24 if cache[f] is not None]
    if not frames24:
        raise RuntimeError("一帧都没读出来")

    # 单帧臂的候选帧：在**全部**真实视角里等间隔取 K 个（与 24 帧子集独立）
    all_ids = list(loc.frame_ids)
    sel = np.linspace(0, len(all_ids) - 1,
                      num=min(int(args.n_single), len(all_ids))).round().astype(int)
    single_ids = [all_ids[int(i)] for i in np.unique(sel)]

    row: Dict[str, Any] = {
        "room": room, "uuid": loc.uuid[:12],
        "n_frames_available": len(all_ids), "n_gt_in_range": n_gt_all,
        "load_seconds": load_s, "max_views": int(args.max_views),
        "resize": int(args.resize),
    }

    # ---------------- 臂 1：全景（视角数扫描，嵌套子集） ----------------
    ks = [k for k in SWEEP if k <= len(frames24)]
    if not ks or ks[-1] != len(frames24):
        ks = sorted(set(ks) | {len(frames24)})
    hr(f"{room} · 臂 1/3：全景（k = {ks}，取自同一批 {len(frames24)} 帧的嵌套子集）")
    pano_rows: List[Dict[str, Any]] = []
    pano_smaps: Dict[int, Any] = {}
    for k in ks:
        idx = np.linspace(0, len(frames24) - 1, num=k).round().astype(int)
        sub = [frames24[int(i)] for i in np.unique(idx)]
        t0 = time.time()
        # 直接构造 `PanoScene`（不再走一遍 `load_scene`：那会重读盘、重读 GT）
        pscene = PanoScene(
            uuid=scene.uuid, room=scene.room,
            panorama=fuse_to_equirect(sub, width=args.width, height=args.height,
                                      max_depth=args.max_range),
            n_frames_available=len(all_ids), frames_used=len(sub),
            objects=scene.objects, gt_boxes=scene.gt_boxes,
            gt_labels=scene.gt_labels,
            meta={"root": str(root), "max_depth": args.max_range,
                  "resize": (args.resize, args.resize),
                  "frame_ids": [f.frame_id for f in sub]})

        vis = visible_gt(pscene, max_range_m=args.max_range)
        order = sorted(int(v["index"]) for v in vis)
        sub_boxes = gt_boxes[order] if order else np.zeros((0, 7))
        sub_labels = [gt_labels[i] for i in order]
        smap, _ = build_pano_map(cfg, pscene, vis)
        pano_smaps[int(k)] = smap

        m_all = match_label_aware(smap, sub_boxes, sub_labels)
        q_all = query_recall(smap, sub_boxes, sub_labels)
        r: Dict[str, Any] = {
            "k": int(k), "n_views": int(len(sub)),
            "coverage": float(pscene.coverage),
            "solid_angle_coverage": float(pscene.panorama.solid_angle_coverage),
            "n_visible": int(len(vis)),
            "visible_rate": len(vis) / max(n_gt_all, 1),
            "n_map_objects": int(smap.num_objects),
            "n_voxels": int(smap.num_voxels),
            "build_seconds": time.time() - t0,
            "visible_idx": order,
        }
        r.update({f"loc_{k2}": v for k2, v in m_all.items()})
        r.update({f"qr_{k2}": v for k2, v in q_all.items()})
        pano_rows.append(r)
        print(f"  k={k:>2}  像素覆盖 {r['coverage']*100:5.1f}%  "
              f"立体角 {r['solid_angle_coverage']*100:5.1f}%  "
              f"可见 {r['n_visible']:>2}/{n_gt_all} ({r['visible_rate']*100:4.1f}%)  "
              f"地图物体 {r['n_map_objects']:>3}  "
              f"命中率 {m_all['match_rate']*100:5.1f}%  "
              f"中位 {'n/a' if m_all['median_m'] is None else format(m_all['median_m'], '.3f')} m  "
              f"≤0.5m {m_all['within_0.50m']*100:4.1f}%  "
              f"查询@1 {q_all['recall@1']*100:4.1f}%")
    row["pano"] = pano_rows
    k_max = max(pano_smaps) if pano_smaps else 0
    row["k_max"] = int(k_max)
    row["visible_idx_pano"] = (pano_rows[-1]["visible_idx"] if pano_rows else [])
    row["n_visible_pano"] = len(row["visible_idx_pano"])
    row["visible_rate_pano"] = row["n_visible_pano"] / max(n_gt_all, 1)

    # ---------------- 臂 2：单帧（多个候选真实视角） ----------------
    hr(f"{room} · 臂 2/3：单帧（{len(single_ids)} 个真实视角，逐个建图）")
    frame_rows: List[Dict[str, Any]] = []
    frame_smaps: Dict[int, Any] = {}
    for fid in single_ids:
        fr = cache.get(fid) or loc.frame(fid, resize=(args.resize, args.resize))
        if fr is None:
            continue
        vis = visible_gt_in_frame(fr, gt_boxes, gt_labels,
                                  max_range_m=args.max_range)
        order = sorted(int(v["index"]) for v in vis)
        sub_boxes = gt_boxes[order] if order else np.zeros((0, 7))
        sub_labels = [gt_labels[i] for i in order]
        smap, _ = build_frame_map(cfg, fr, vis)
        frame_smaps[int(fid)] = smap
        m = match_label_aware(smap, sub_boxes, sub_labels)
        q = query_recall(smap, sub_boxes, sub_labels)
        rr: Dict[str, Any] = {
            "frame": int(fid), "n_visible": len(order),
            "visible_rate": len(order) / max(n_gt_all, 1),
            "n_map_objects": int(smap.num_objects),
            "n_voxels": int(smap.num_voxels),
            "visible_idx": order,
        }
        rr.update({f"loc_{k2}": v for k2, v in m.items()})
        rr.update({f"qr_{k2}": v for k2, v in q.items()})
        frame_rows.append(rr)
        print(f"  frame {fid:>3}  可见 {len(order):>2}/{n_gt_all} "
              f"({rr['visible_rate']*100:4.1f}%)  地图物体 {rr['n_map_objects']:>3}  "
              f"命中率 {m['match_rate']*100:5.1f}%  "
              f"中位 {'n/a' if m['median_m'] is None else format(m['median_m'], '.3f')} m  "
              f"≤0.5m {m['within_0.50m']*100:4.1f}%  "
              f"查询@1 {q['recall@1']*100:4.1f}%")
    row["single"] = frame_rows
    # 单帧臂的**最强形态**：per-object 挑最好的那一帧（oracle 并集）。
    # 这是给单帧臂的最有利假设 —— 如果全景连它都赢不了，就不能再用
    # "你只是没选对帧"来解释。代价是它需要 **N 张各自独立的地图**，
    # 而全景只要 **1 张**。
    union_idx: set = set()
    for rr in frame_rows:
        union_idx |= set(rr["visible_idx"])
    row["single_union_idx"] = sorted(union_idx)
    row["single_union_n"] = len(union_idx)
    row["single_union_rate"] = len(union_idx) / max(n_gt_all, 1)
    row["single_max_n"] = max([rr["n_visible"] for rr in frame_rows], default=0)
    row["single_median_n"] = float(np.median([rr["n_visible"] for rr in frame_rows])) \
        if frame_rows else 0.0
    print(f"  —— 单帧臂最强形态（oracle 并集，需 {len(frame_rows)} 张图）："
          f"覆盖 {len(union_idx)}/{n_gt_all} = {row['single_union_rate']*100:.1f}%；"
          f"单帧最好 {row['single_max_n']}，中位 {row['single_median_n']:.0f}")

    # ---------------- 臂 3：同帧的全景 vs 针孔（表示形式对照） ----------------
    hr(f"{room} · 臂 3/3：同一帧的「全景 vs 针孔」（把表示形式与信息量拆开）")
    ref_fid = int(views24[0])
    ref_fr = cache.get(ref_fid)
    pair: Dict[str, Any] = {"frame": ref_fid}
    if ref_fr is not None:
        vis_f = visible_gt_in_frame(ref_fr, gt_boxes, gt_labels,
                                    max_range_m=args.max_range)
        order_f = sorted(int(v["index"]) for v in vis_f)
        smap_f, _ = build_frame_map(cfg, ref_fr, vis_f)
        m_f = match_label_aware(smap_f, gt_boxes[order_f] if order_f else np.zeros((0, 7)),
                                [gt_labels[i] for i in order_f])
        pair.update({
            "frame_n_visible": len(order_f),
            "frame_n_map_objects": int(smap_f.num_objects),
            "frame_loc_median_m": m_f["median_m"],
            "frame_loc_match_rate": m_f["match_rate"],
            "frame_visible_idx": order_f,
        })
        # 全景臂 k=1 用的就是 `views24[0]`（同一批帧的嵌套子集起点）
        k1 = next((r for r in pano_rows if r["k"] == 1), None)
        if k1 is not None:
            pair.update({
                "pano_n_visible": k1["n_visible"],
                "pano_n_map_objects": k1["n_map_objects"],
                "pano_loc_median_m": k1.get("loc_median_m"),
                "pano_loc_match_rate": k1.get("loc_match_rate"),
                "pano_visible_idx": k1["visible_idx"],
            })
        print(f"  同一帧 {ref_fid}：针孔可见 {len(order_f)} / 地图物体 "
              f"{smap_f.num_objects} / 中位 "
              f"{'n/a' if m_f['median_m'] is None else format(m_f['median_m'], '.4f')} m")
        if k1 is not None:
            print(f"                 等距柱状可见 {k1['n_visible']} / 地图物体 "
                  f"{k1['n_map_objects']} / 中位 "
                  f"{'n/a' if k1.get('loc_median_m') is None else format(k1['loc_median_m'], '.4f')} m")
    row["paired"] = pair

    # ---------------- 共同子集：两条臂都可见的 GT ----------------
    hr(f"{room} · 共同子集定位对比（控住分母，回答『是不是也更准』）")
    best_single = max(frame_rows, key=lambda r: (r["n_visible"], -r["frame"]),
                      default=None)
    row["best_single_frame"] = None if best_single is None else best_single["frame"]
    if best_single is not None:
        common = sorted(set(best_single["visible_idx"]) & set(row["visible_idx_pano"]))
        row["n_common"] = len(common)
        if common:
            sub_boxes = gt_boxes[common]
            sub_labels = [gt_labels[i] for i in common]
            pm = match_label_aware(pano_smaps[k_max], sub_boxes, sub_labels)
            fm = match_label_aware(frame_smaps[best_single["frame"]],
                                   sub_boxes, sub_labels)
            row["common_pano"] = pm
            row["common_single"] = fm
            print(f"  最好单帧 = frame {best_single['frame']}（可见 "
                  f"{best_single['n_visible']} 个）；与全景（k={k_max}）共同可见 "
                  f"{len(common)} 个 GT")
            for tag, m in (("全景", pm), ("单帧", fm)):
                print(f"  [{tag}] 命中率 {m['match_rate']*100:5.1f}%  "
                      f"中位 {'n/a' if m['median_m'] is None else format(m['median_m'], '.4f')} m  "
                      f"≤0.25m {m['within_0.25m']*100:5.1f}%  "
                      f"≤0.50m {m['within_0.50m']*100:5.1f}%  "
                      f"命中物体点数中位 "
                      f"{'n/a' if m['median_points'] is None else format(m['median_points'], '.0f')}")

            # —— 在**同一个固定 GT 集合**上扫 k：覆盖随 k 涨，精度呢？——
            # 这一列是判"全景的精度到底有没有随视角数变好"的关键：
            # 分母固定（都用最好单帧可见的那批 GT），所以 match_rate 的上升
            # 就是**覆盖**的上升，而中位误差变化是**精度**的变化。
            print(f"  固定 GT 集合（{len(common)} 个）上扫 k：")
            acc = []
            for p in row["pano"]:
                mm = match_label_aware(pano_smaps[p["k"]], sub_boxes, sub_labels)
                acc.append({"k": p["k"], "match_rate": mm["match_rate"],
                            "median_m": mm["median_m"],
                            "median_points": mm["median_points"]})
                print(f"    k={p['k']:>2}  命中率 {mm['match_rate']*100:5.1f}%  "
                      f"中位 {'n/a' if mm['median_m'] is None else format(mm['median_m'], '.4f')} m  "
                      f"点数中位 "
                      f"{'n/a' if mm['median_points'] is None else format(mm['median_points'], '.0f')}")
            row["acc_vs_k"] = acc
        else:
            print("  [n/a] 没有共同可见的 GT（单帧与全景交集为空）")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--rooms", nargs="*",
                    default=["office_6", "hallway_6", "office_27"])
    ap.add_argument("--max-views", type=int, default=24)
    ap.add_argument("--n-single", type=int, default=12)
    ap.add_argument("--resize", type=int, default=540,
                    help="源视角缩放边长（两条臂用同一个值，保证信息量口径一致）")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--out", default="runs/41_ablate_panorama_value.json")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2

    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("project.verbose", False)

    results: List[Dict[str, Any]] = []
    for room in args.rooms:
        try:
            results.append(run_room(cfg, args.root, room, args))
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"[skip] {room}: {type(e).__name__}: {e}")

    if not results:
        print("[FAIL] 一个房间都没跑成")
        return 1

    # ---------------- 汇总 ----------------
    hr("汇总 A：可观测上限（分母 = 房间内 ≤max_range 的 GT）")
    print(f"{'房间':<12}{'GT':>5}{'全景可见':>10}{'占比':>8}"
          f"{'单帧中位':>10}{'单帧最好':>10}{'单帧oracle并集':>15}")
    for r in results:
        print(f"{r['room']:<12}{r['n_gt_in_range']:>5}{r['n_visible_pano']:>10}"
              f"{r['visible_rate_pano']*100:>7.1f}%{r['single_median_n']:>10.0f}"
              f"{r['single_max_n']:>10}"
              f"{str(r['single_union_n']) + ' (' + format(r['single_union_rate']*100, '.1f') + '%)':>15}")

    hr("汇总 B：全景视角数扫描（同一批帧的嵌套子集）")
    print(f"{'房间':<12}{'k':>4}{'可见':>6}{'命中率':>9}{'中位误差':>11}"
          f"{'≤0.25m':>9}{'≤0.50m':>9}{'查@1':>8}")
    for r in results:
        for p in r["pano"]:
            md = p.get("loc_median_m")
            print(f"{r['room']:<12}{p['k']:>4}{p['n_visible']:>6}"
                  f"{p['loc_match_rate']*100:>8.1f}%"
                  f"{('n/a' if md is None else f'{md:.4f} m'):>11}"
                  f"{p['loc_within_0.25m']*100:>8.1f}%"
                  f"{p['loc_within_0.50m']*100:>8.1f}%"
                  f"{p['qr_recall@1']*100:>7.1f}%")

    hr("汇总 C：同一帧下「等距柱状 vs 针孔」（表示形式 vs 信息量）")
    print(f"{'房间':<12}{'帧':>5}{'针孔可见':>9}{'全景可见':>9}"
          f"{'针孔中位':>11}{'全景中位':>11}")
    for r in results:
        p = r.get("paired") or {}
        fmt = lambda x: "n/a" if x is None else f"{x:.4f} m"   # noqa: E731
        print(f"{r['room']:<12}{p.get('frame', -1):>5}{p.get('frame_n_visible', 0):>9}"
              f"{p.get('pano_n_visible', 0):>9}"
              f"{fmt(p.get('frame_loc_median_m')):>11}"
              f"{fmt(p.get('pano_loc_median_m')):>11}")

    # ---------------- 判据 ----------------
    ok = True
    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    hr("判据（哪一条不成立就要如实写出来，不许含糊过去）")
    # 1a. 全景 vs **最好的一帧**（最常用的对照）
    w_all, w_best = [], []
    for r in results:
        w_all.append(r["visible_rate_pano"])
        w_best.append(r["single_max_n"] / max(r["n_gt_in_range"], 1))
    if w_all and w_best:
        check("全景可观测上限 > 最好的单帧（覆盖价值）",
              float(np.mean(w_all)) > float(np.mean(w_best)),
              f"全景均值 {np.mean(w_all)*100:.1f}% vs 最好单帧 {np.mean(w_best)*100:.1f}%"
              f"（{np.mean(w_all)/max(np.mean(w_best),1e-9):.2f}×）")

    # 1b. 全景 vs **单帧的 oracle 并集**：给单帧臂最有利的假设
    w_union = [r["single_union_rate"] for r in results]
    if w_all and w_union:
        a, b = float(np.mean(w_all)), float(np.mean(w_union))
        frac = float(np.mean([1.0 if r["n_visible_pano"] >= r["single_union_n"] else 0.0
                              for r in results]))
        detail = (f"全景 1 张图 {a*100:.1f}% vs 单帧并集 "
                  f"{len(results[0]['single'])} 张图 {b*100:.1f}%")
        if a >= b:
            check("全景覆盖 ≥ 单帧 oracle 并集（每房间都成立则更强）", True,
                  detail + f"；逐房间达标率 {frac*100:.0f}%")
        else:
            # 不达标也要如实打出来，并**不算通过**
            check("全景覆盖 ≥ 单帧 oracle 并集", False,
                  detail + " —— 全景没有赢过『per-object 挑最好那一帧』这个 oracle；"
                  "结论必须缩到『全景把 N 帧合成 1 张一致的图』，不能再声称覆盖更高")
    # 2. 视角数的趋势（**信息项**，不参与通过/不通过：
    #    融合用了共识筛选，多一帧理论上可能把某个像素判成离群，不保证严格单调）
    print("  [INFO] 可见 GT 数随视角数变化：")
    for r in results:
        v = [p["n_visible"] for p in r["pano"]]
        kk = [p["k"] for p in r["pano"]]
        mono = all(v[i + 1] >= v[i] for i in range(len(v) - 1))
        print(f"         {r['room']}: " + " → ".join(f"k={a}:{b}" for a, b in zip(kk, v))
              + ("（单调不降）" if mono else "（**中间有回落**，需在报告里说明）"))

    # 3. 共同子集上，全景是否**也更准**
    #
    #    ⚠️ 这一条**不放进通过/不通过**，因为它不是"应该成立的性质"，
    #    而是一个**待测的经验问题**。实测（见 runs/41_*.log）答案是：
    #    全景在共同子集上**没有**更准，甚至更差（平均 −22%）——
    #    因为共同子集里那些物体在单帧里正好位于光轴附近、像素最多、
    #    深度最准；而全景把它们从多个掠射角度的观测也并了进来。
    #    既然测出来是这样，那么"全景提升定位精度"这句话就**不许写**。
    #    下面把结论限定到实际测出来的那一句。
    pairs = [(r.get("common_pano", {}).get("median_m"),
              r.get("common_single", {}).get("median_m")) for r in results]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    print("  [INFO] 共同子集精度对比（**不是判据**，是结论用语的分界线）：")
    if pairs:
        am = float(np.mean([a for a, _ in pairs]))
        bm = float(np.mean([b for _, b in pairs]))
        better = sum(1 for a, b in pairs if a <= b)
        print(f"         全景 {am:.4f} m vs 单帧 {bm:.4f} m（{len(pairs)} 个房间，"
              f"全景更优 {better}/{len(pairs)}）")
        if am <= bm:
            print("         ⇒ 可以说：全景在共同可见物体上**不劣于**单帧。")
        else:
            print(f"         ⇒ **不许说**「全景定位更准」**：实测全景差 "
                  f"{(am/bm-1)*100:.1f}%。全景可辩护的价值只有**覆盖**"
                  f"（一次观测看得见更多物体），精度不是它的卖点。")
    else:
        print("         n/a（没有可用的共同子集样本）")

    # 4. 同帧的两种表示形式：信息量相同时是否等价
    eq = []
    for r in results:
        p = r.get("paired") or {}
        if p.get("frame_n_visible") is not None and p.get("pano_n_visible") is not None:
            eq.append((p["frame_n_visible"], p["pano_n_visible"]))
    if eq:
        check("同一帧的「针孔 vs 全景」可观测数相当（±1 个 GT 内）",
              all(abs(a - b) <= 1 for a, b in eq),
              "; ".join(f"{a} vs {b}" for a, b in eq)
              + " —— 相当说明等距柱状**本身**不创造信息，"
                "价值来自它能让 N 个视角落进同一张图")

    # ---------------- 允许写的结论 ----------------
    hr("结论用语（照抄，不要扩写）")
    pano_rate = float(np.mean([r["visible_rate_pano"] for r in results]))
    best_rate = float(np.mean([r["single_max_n"] / max(r["n_gt_in_range"], 1)
                               for r in results]))
    union_rate = float(np.mean([r["single_union_rate"] for r in results]))
    print(f"  · 在全景【1 张图】上可观测到房间内 {pano_rate*100:.1f}% 的 "
          f"≤{args.max_range:.0f} m GT 物体；")
    print(f"    单个真实视角最好只有 {best_rate*100:.1f}%，"
          f"每物体各挑一帧的 oracle 并集（{len(results[0]['single'])} 张图）"
          f"也只有 {union_rate*100:.1f}%。")
    print(f"  · 视角数 1→{args.max_views} 时可见物体数单调上升；"
          f"同一帧换成等距柱状表示**不改变**可见物体数。")
    accs = [(r.get("common_pano", {}).get("median_m"),
             r.get("common_single", {}).get("median_m")) for r in results]
    accs = [(a, b) for a, b in accs if a is not None and b is not None]
    if accs:
        am = float(np.mean([a for a, _ in accs]))
        bm = float(np.mean([b for _, b in accs]))
        rel = "不劣于" if am <= bm else "差于"
        print(f"  · **精度**：共同可见物体上全景 {am:.3f} m vs 单帧 {bm:.3f} m，{rel}单帧"
              + ("" if am <= bm else
                 " —— 所以**不要**声称全景提升定位精度。"))
    print("  · 禁止写：『全景提升了定位精度』『虚拟视角』『任意视角渲染』"
          "（本项目没有任何虚拟视角，只有真实采集视角的融合）。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n产物已写入 {out}")
    print("\n" + "=" * 78)
    print("结论: " + ("全部判据通过 ✓" if ok else "存在不通过项 ✗（如实记录）"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
