# -*- coding: utf-8 -*-
"""消融：实例关联策略到底值多少？（批评 #9）

问题是什么
==========
现在地图里的物体是这么来的：把每个 2D 检测反投影成 3D 点云，
再按"**标签兼容 + 中心距离 < 0.6 m 或 3D IoU > 0.1 的贪心最近邻**"
把多帧观测归并成"一条轨迹 = 一个物体"（`MapBuilder._associate`）。

批评 #9 说得对：这套东西**没有新意** —— 它就是 SORT/ByteTrack 那一类
朴素贪心关联。但"没有新意"不等于"没有用"：真正该回答的是
**换掉它，地图会变差多少**。如果换成最优分配（Hungarian）、
或者干脆不要跨帧关联（体素聚类），结果几乎不变，那么这块就是装饰；
如果差很多，那它至少是**承重结构**，值得写清楚。

实验怎么做到"只换关联、别的都不动"
================================
★ 关键手法：**一次建图，多次重新分组**。

先用默认参数跑一遍 `MapBuilder.build_from_frames()`，拿到
`observations`（每个检测的 3D 点云）与 `VoxelGrid`（体素场）。
体素场是**逐帧独立**累加的，与关联无关；观测更是原始量。
于是后面每一条臂都只是"把**同一批观测**重新分成若干组"，
再调用同一个 `finalize()` 产出物体 ——
几何、感知、体素、特征全部逐位相同，唯一的自变量就是**分组规则**。

这样的好处是：不存在"换了实现导致数值漂移"的解释空间。
另外 `greedy_label` 臂是**离线重写**默认规则，它必须与
`lib_default`（库自己的轨迹）**完全一致** —— 脚本会断言这一点，
不一致就说明离线复现写错了，后面的结论全部作废。

臂
==
    lib_default     库默认（金标准，只用来校验离线复现）
    greedy_label    离线复现的默认规则（应该与上一条逐一相同）
    greedy_nolabel  去掉"标签兼容"约束（同名才算同一物体 → 只按几何）
    radius_only     去掉 3D IoU 兜底，只留中心距离
    nearest_any     去掉距离门限：永远并给最近的兼容轨迹（会过度合并）
    hungarian       逐帧**全局最优**分配（scipy linear_sum_assignment），
                    打分函数与贪心完全相同，只是不做"先到先得"
    hungarian_nolabel 同上但不要标签约束
    clustering      不做跨帧关联，直接对体素做 DBSCAN（降级路径）

指标
====
    n_true        K 帧里**任一帧看得见**的 GT 物体数（并集，= 应该得到的物体数）
    n_objects     地图里实际产出的物体数 → `fragmentation = n_objects/n_true`
    match_rate    并集 GT 里有多少个拿到了"标签相同 + 2 m 内有物体"的命中
    median_m      命中物体的中心误差中位
    claims_1/2+   每个 GT 被**几个**地图物体认领：=1 正常，≥2 是**过分割**
    spurious      地图物体里"谁也没认领"的比例（幽灵物体）

跑法：
    python scripts/42_ablate_association.py --rooms office_6 hallway_6 office_27
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
from roboground.data.pano_scene import select_location  # noqa: E402

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")
#: 全部臂（顺序即打印顺序）
STRATEGIES = ("lib_default", "greedy_label", "greedy_nolabel", "radius_only",
              "nearest_any", "hungarian", "hungarian_nolabel",
              "hungarian_radius_only", "clustering")
#: 判定"认领"的距离门限（米）
CLAIM_DIST_M = 2.0


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ==========================================================================
# 打分：**与库内完全相同的规则**，只是把它抽出来复用
# ==========================================================================
def pair_score(b, obs, track, *, radius: float, iou_thr: float,
               require_label: bool, no_radius_limit: bool = False,
               stats: Optional[Dict[str, int]] = None) -> float:
    """观测 `obs` 与轨迹 `track` 的关联得分；`-1` 表示不允许关联。

    规则照抄 `MapBuilder._associate`：
    · 标签不兼容 → 不允许（`assoc_require_label` 为真时）
    · 中心距离 ≤ radius → `1 − d/radius`（越大越好）
    · 否则若 3D IoU ≥ iou_thr → 用 IoU 当分数

    `stats` 用来数**每个分支被选中了几次** —— 只看结果差多少，
    不知道差在哪一步，就等于没解释。实测（见 log）IoU 兜底被选中时
    两团点云的中心距离经常在 0.6 m 以上，正是精度的损失来源。
    """
    from roboground.mapping.builder import MapBuilder

    if require_label and not b._labels_compatible(obs.label, track.label):
        return -1.0
    center = obs.centroid
    if center is None:
        center = np.zeros(3)
    dist = float(np.linalg.norm(center - track.center))
    if no_radius_limit:
        # `nearest_any`：不给门限，直接按距离给分（越近越好）
        return 1.0 / (1.0 + dist)
    if dist <= radius:
        if stats is not None:
            stats["by_radius"] = stats.get("by_radius", 0) + 1
        return 1.0 - dist / max(radius, 1e-6)
    if track.points_count > 0:
        iou = float(MapBuilder._track_observation_iou(track, obs))
        if iou >= iou_thr:
            if stats is not None:
                stats["by_iou"] = stats.get("by_iou", 0) + 1
                stats.setdefault("iou_far_dists", []).append(dist)   # type: ignore[arg-type]
            return iou
    return -1.0


def _new_track(b, track_id: int):
    from roboground.mapping.builder import _ObjectTrack

    return _ObjectTrack(track_id, b.feature_dim, b.object_max_points)


def assign_sequential(b, observations: Sequence[Any], *, require_label: bool,
                      radius: float, iou_thr: float,
                      no_radius_limit: bool = False,
                      stats: Optional[Dict[str, Any]] = None) -> List[Any]:
    """逐观测贪心（库内算法的**离线重写**，必须与 `lib_default` 一致）。"""
    tracks: List[Any] = []
    for obs in observations:
        best, best_score = None, -1.0
        best_stats: Optional[Dict[str, int]] = {} if stats is not None else None
        for tr in tracks:
            s_stats: Optional[Dict[str, int]] = {} if stats is not None else None
            s = pair_score(b, obs, tr, radius=radius, iou_thr=iou_thr,
                           require_label=require_label,
                           no_radius_limit=no_radius_limit, stats=s_stats)
            # ★ 用严格大于：与库内一致，分数相同时保留**先建的那条**轨迹
            if s > best_score:
                best, best_score, best_stats = tr, s, s_stats
        if best is None:
            best = _new_track(b, len(tracks))
            tracks.append(best)
            if stats is not None:
                stats["new_track"] = stats.get("new_track", 0) + 1
        elif stats is not None and best_stats:
            # 只累计**最终被选中**的那次决策走的是哪个分支
            for k, v in best_stats.items():
                if k == "iou_far_dists":
                    stats.setdefault(k, []).extend(v)      # type: ignore[arg-type]
                else:
                    stats[k] = stats.get(k, 0) + int(v)
        best.add(obs)
    return tracks


def assign_hungarian(b, observations: Sequence[Any], *, require_label: bool,
                     radius: float, iou_thr: float) -> List[Any]:
    """逐帧**全局最优**分配：同一帧的观测一起和已有轨迹做一对一匹配。

    与贪心的区别只有一点：贪心"先到的观测先挑"，可能把某条轨迹
    抢给了一个不该拿它的观测；最优分配会让**整帧总得分**最大。

    实现上把"不分配"也做成一个选项：给每个观测补一列分数为 0 的哑列，
    不允许的配对给 `−1e6`，于是"空着"永远优于"硬塞一个不合法配对"。
    """
    from scipy.optimize import linear_sum_assignment

    tracks: List[Any] = []
    # 按首次出现顺序分组（`b.observations` 本来就是帧序 + 检测序）
    order: List[str] = []
    groups: Dict[str, List[Any]] = {}
    for obs in observations:
        fid = str(getattr(obs, "frame_id", "frame"))
        if fid not in groups:
            groups[fid] = []
            order.append(fid)
        groups[fid].append(obs)

    for fid in order:
        grp = groups[fid]
        n_o, n_t = len(grp), len(tracks)
        # 每一行末尾有 n_o 个哑列（自己一个，保证"空着"总是可行）
        M = np.zeros((n_o, n_t + n_o), dtype=np.float64)
        M[:, n_t:] = 0.0
        for i, obs in enumerate(grp):
            for j, tr in enumerate(tracks):
                s = pair_score(b, obs, tr, radius=radius, iou_thr=iou_thr,
                               require_label=require_label)
                # ★ 合法但得分为 0 的配对（距离正好等于门限）必须**保留**：
                #   库内是用 `score > best_score`（初始 −1）比较的，
                #   0 分是合法的。这里若写成 `s > 0` 会把它误判成不合法。
                M[i, j] = s if s >= 0.0 else -1e6
        rows, cols = linear_sum_assignment(-M)
        for i, j in zip(rows.tolist(), cols.tolist()):
            obs = grp[i]
            if j < n_t and M[i, j] > -1e5:
                tracks[j].add(obs)
            else:
                tr = _new_track(b, len(tracks))
                tracks.append(tr)
                tr.add(obs)
    return tracks


# ==========================================================================
# 指标：GT ↔ 地图物体 的**认领关系**
# ==========================================================================
def instance_quality(smap, gt_boxes: np.ndarray, gt_labels: Sequence[str], *,
                     max_dist: float = CLAIM_DIST_M) -> Dict[str, Any]:
    """数"每个 GT 被几个地图物体认领"，而不是只数一对一命中。

    为什么要单独写：一对一贪心匹配**看不见过分割** ——
    一个被拆成 3 块的椅子在贪心里只算 1 次命中，
    但机器人看到地图里 3 把椅子就是错的。所以这里直接数认领数。
    """
    objs = list(getattr(smap, "objects", []) or [])
    boxes = np.asarray(gt_boxes, dtype=np.float64)
    n_gt = int(boxes.shape[0]) if boxes.size else 0
    out: Dict[str, Any] = {
        "n_true": n_gt, "n_objects": int(len(objs)),
        "fragmentation": (len(objs) / n_gt) if n_gt else None,
        "n_matched": 0, "match_rate": 0.0, "median_m": None,
        "claims_1": 0, "claims_ge2": 0, "claims_0": 0,
        "frac_claims_1": 0.0, "frac_claims_ge2": 0.0,
        "n_spurious": int(len(objs)), "frac_spurious": 1.0 if objs else 0.0,
    }
    if n_gt == 0 or not objs:
        return out

    lab_gt = [str(s).strip().lower() for s in gt_labels]
    obj_lab = [str(getattr(o, "label", "")).strip().lower() for o in objs]
    obj_c = np.stack([np.asarray(o.center, dtype=np.float64).reshape(3)
                      for o in objs], axis=0)

    claimed = np.zeros(len(objs), dtype=bool)
    claims: List[int] = []
    dists: List[float] = []
    for gi in range(n_gt):
        d = np.linalg.norm(obj_c - boxes[gi, :3][None, :], axis=1)
        same = np.array([obj_lab[k] == lab_gt[gi] for k in range(len(objs))],
                        dtype=bool)
        hit = np.flatnonzero(same & (d <= max_dist))
        claims.append(int(hit.size))
        if hit.size:
            claimed[hit] = True
            dists.append(float(d[hit].min()))

    claims_a = np.asarray(claims, dtype=np.int64)
    n_matched = int((claims_a > 0).sum())
    out.update({
        "n_matched": n_matched,
        "match_rate": float((claims_a > 0).mean()),
        "median_m": float(np.median(dists)) if dists else None,
        "claims_1": int((claims_a == 1).sum()),
        "claims_ge2": int((claims_a >= 2).sum()),
        "claims_0": int((claims_a == 0).sum()),
        "frac_claims_1": float((claims_a == 1).mean()),
        "frac_claims_ge2": float((claims_a >= 2).mean()),
        "n_spurious": int((~claimed).sum()),
        "frac_spurious": float((~claimed).mean()) if len(objs) else 0.0,
        # ⚠️ `claims_ge2` **不是**过分割的可靠指标：同一个房间里有两扇门、
        # 两张桌子时，它们本来就互相在 2 m 内，于是每个都会被 ≥2 个物体"认领"。
        # 真正能读的是这两个：`over_seg`（多出来的物体数）与 `miss`（漏掉的 GT 数）。
        "over_seg": int(len(objs) - n_matched),
        "miss": int(n_gt - n_matched),
        "n_true": n_gt,
    })
    return out


# ==========================================================================
# 主流程
# ==========================================================================
def run_room(cfg, root: Path, room: str, args, loc=None) -> Dict[str, Any]:
    if loc is None:
        loc = select_location(root, room=room, min_frames=8)
    hr(f"{room}：{loc.uuid[:12]}，{len(loc.frame_ids)} 个真实视角")

    # ---- 输入：K 个真实视角 + 每帧的 GT 注入检测（与 41 号脚本同一手法）----
    ids = list(loc.frame_ids)
    sel = np.linspace(0, len(ids) - 1,
                      num=min(int(args.n_frames), len(ids))).round().astype(int)
    fids = [ids[int(i)] for i in np.unique(sel)]
    frames = [loc.frame(f) for f in fids]
    frames = [f for f in frames if f is not None]

    # GT：用 `load_scene` 的正规入口读（同一套"房间匹配"逻辑），
    # 只给它 1 帧、极小分辨率 —— 这里只要 GT，不要全景。
    from roboground.data.pano_scene import load_scene

    scene = load_scene(root, loc, width=256, height=128, max_frames=1,
                       max_depth=args.max_range)
    gt_boxes, gt_labels = scene.gt_within(args.max_range)
    print(f"  房间 GT（≤{args.max_range:.0f} m）{gt_boxes.shape[0]} 个"
          f"（房间标注共 {len(scene.objects)} 条）")

    # `n_true` 的口径：**这 K 帧里至少被一帧看见**的 GT（并集），
    # 而不是房间全部 GT —— 看不见的物体不该算作"关联没做出来"。
    union: set = set()
    dets_by_frame: Dict[str, List[Dict[str, Any]]] = {}
    for fr in frames:
        vis = visible_gt_in_frame(fr, gt_boxes, gt_labels,
                                  max_range_m=args.max_range)
        dets_by_frame[fr.frame_id] = vis
        union |= {int(v["index"]) for v in vis}
    union_idx = sorted(union)
    sub_boxes = (gt_boxes[union_idx] if union_idx else np.zeros((0, 7))).astype(np.float64)
    sub_labels = [gt_labels[i] for i in union_idx]

    from roboground.types import Detection2D
    from roboground.mapping import MapBuilder

    prompts = sorted({str(v["label"]) for vs in dets_by_frame.values() for v in vs})
    b = MapBuilder(cfg, prompts=prompts or ["object"])
    per_frame: Dict[str, List[Detection2D]] = {}
    for fr in frames:
        ds = []
        for v in dets_by_frame[fr.frame_id]:
            for (u0, v0, u1, v1) in v.get("uv_parts") or [v["uv"]]:
                ds.append(Detection2D(
                    label=str(v["label"]), score=1.0,
                    bbox=np.array([u0, v0, u1, v1], dtype=np.float64),
                    prompt=str(v["label"])))
        per_frame[fr.frame_id] = ds

    def fake_run(frame, prompts=None):
        """注入 GT 检测，替换掉感知流水线（其余链路完全不变）。"""
        return per_frame.get(frame.frame_id, [])

    b.pipeline.run = fake_run                                    # type: ignore[assignment]
    t0 = time.time()
    smap_lib = b.build_from_frames(frames)
    print(f"  输入 {len(frames)} 帧 → 观测 {len(b.observations)} 个、"
          f"体素 {b.grid.num_voxels} 个（建图 {time.time()-t0:.1f}s）")
    print(f"  并集 GT（K 帧里至少被看见一次）{len(union_idx)} 个；"
          f"库默认产出 {smap_lib.num_objects} 个物体")

    radius = float(b.assoc_radius)
    iou_thr = float(b.assoc_iou) if b.assoc_iou is not None else float("inf")
    lib_strategy = b.assoc_strategy
    # 保存库默认状态（后面每条臂都要改 `b.tracks` / `b.object_mode`）
    lib_tracks = list(b.tracks)
    lib_mode = b.object_mode
    observations = list(b.observations)

    rows: List[Dict[str, Any]] = []
    for name in STRATEGIES:
        t0 = time.time()
        branch_stats: Dict[str, Any] = {}
        if name == "lib_default":
            b.tracks = list(lib_tracks)
            b.object_mode = "association"
            smap = b.finalize()
        elif name == "clustering":
            b.tracks = []
            b.object_mode = "clustering"
            smap = b.finalize()
        elif name.startswith("greedy"):
            b.object_mode = "association"
            st: Dict[str, Any] = {}
            b.tracks = assign_sequential(
                b, observations,
                require_label=("nolabel" not in name),
                radius=radius, iou_thr=iou_thr, stats=st)
            smap = b.finalize()
            branch_stats = st
        elif name == "radius_only":
            b.object_mode = "association"
            # `iou_thr=inf` 等价于关掉 IoU 兜底
            b.tracks = assign_sequential(b, observations, require_label=True,
                                         radius=radius, iou_thr=float("inf"))
            smap = b.finalize()
        elif name == "nearest_any":
            b.object_mode = "association"
            b.tracks = assign_sequential(b, observations, require_label=True,
                                         radius=radius, iou_thr=iou_thr,
                                         no_radius_limit=True)
            smap = b.finalize()
        elif name.startswith("hungarian"):
            b.object_mode = "association"
            b.tracks = assign_hungarian(b, observations,
                                        require_label=("nolabel" not in name),
                                        radius=radius,
                                        # `hungarian_radius_only`：两种修法叠加
                                        iou_thr=(float("inf") if "radius_only" in name
                                                 else iou_thr))
            smap = b.finalize()
        else:
            raise ValueError(name)

        q = instance_quality(smap, sub_boxes, sub_labels)
        far = branch_stats.get("iou_far_dists") or []
        r: Dict[str, Any] = {"arm": name, "seconds": time.time() - t0,
                             "n_tracks": int(len(b.tracks)), **q,
                             "assoc_by_radius": int(branch_stats.get("by_radius", 0)),
                             "assoc_by_iou": int(branch_stats.get("by_iou", 0)),
                             "assoc_new_track": int(branch_stats.get("new_track", 0)),
                             "iou_far_median_m": (float(np.median(far)) if far else None),
                             "iou_far_max_m": (float(np.max(far)) if far else None)}
        rows.append(r)
        extra = ""
        if r["assoc_by_iou"] or r["assoc_new_track"]:
            extra = (f"  关联：距离 {r['assoc_by_radius']} / IoU兜底 {r['assoc_by_iou']}"
                     f" / 新建 {r['assoc_new_track']}")
            if r["iou_far_median_m"] is not None:
                extra += (f"（IoU 兜底时的中心距离中位 "
                          f"{r['iou_far_median_m']:.2f} m，"
                          f"max {r['iou_far_max_m']:.2f} m）")
        print(f"  [{name:<17}] 轨迹 {len(b.tracks):>3}  物体 {q['n_objects']:>3}"
              f"（碎裂率 {('n/a' if q['fragmentation'] is None else format(q['fragmentation'], '.2f'))}）"
              f"  命中率 {q['match_rate']*100:5.1f}%  "
              f"中位 {('n/a' if q['median_m'] is None else format(q['median_m'], '.4f'))} m  "
              f"幽灵 {q['frac_spurious']*100:5.1f}%" + extra)

    # 恢复默认状态，避免影响别的调用（虽然本进程到此为止）
    b.tracks = lib_tracks
    b.object_mode = lib_mode

    return {
        "room": room, "uuid": loc.uuid[:12],
        "n_frames": len(frames), "n_observations": len(observations),
        "n_voxels": int(b.grid.num_voxels),
        "n_true": len(union_idx), "union_idx": union_idx,
        "assoc_radius": radius, "assoc_iou": iou_thr,
        "assoc_strategy": str(lib_strategy),
        "arms": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--rooms", nargs="*", default=None,
                    help="指定房间名；不给则用 --n-rooms 在全区等间隔抽点")
    ap.add_argument("--n-rooms", type=int, default=0,
                    help="在全部 186 个采集点里等间隔抽多少个（覆盖不同房间类型）")
    ap.add_argument("--n-frames", type=int, default=12,
                    help="每个采集点用多少个真实视角（同一批观测喂给所有臂）")
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--out", default="runs/42_ablate_association.json")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2

    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("project.verbose", False)

    # 选点：显式房间名，或"等间隔抽 n 个采集点"（避免只测同一种房间）
    targets: List[Tuple[str, Any]] = []
    if args.n_rooms:
        from roboground.data.stanford2d3d import list_locations

        locs = sorted(list_locations(args.root), key=lambda l: -len(l.frame_ids))
        locs = [l for l in locs if len(l.frame_ids) >= max(8, args.n_frames)]
        idx = np.linspace(0, len(locs) - 1,
                          num=min(int(args.n_rooms), len(locs))).round().astype(int)
        for i in np.unique(idx):
            targets.append((str(locs[int(i)].room), locs[int(i)]))
    else:
        for room in (args.rooms or ["office_6", "hallway_6", "office_27"]):
            targets.append((room, None))

    results: List[Dict[str, Any]] = []
    for room, loc in targets:
        try:
            results.append(run_room(cfg, args.root, room, args, loc=loc))
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"[skip] {room}: {type(e).__name__}: {e}")
    if not results:
        print("[FAIL] 一个采集点都没跑成")
        return 1

    hr("汇总（每格为各房间的均值；碎裂率 = 地图物体数 / 应得物体数，越接近 1 越好）")
    print(f"{'臂':<19}{'应得':>6}{'物体数':>8}{'碎裂率':>9}{'命中率':>9}{'中位误差':>11}"
          f"{'多出':>7}{'漏掉':>7}{'幽灵':>8}")
    for name in STRATEGIES:
        rs = [next((a for a in r["arms"] if a["arm"] == name), None) for r in results]
        rs = [x for x in rs if x is not None]
        if not rs:
            continue
        def mean(key: str) -> Optional[float]:
            v = [x[key] for x in rs if x.get(key) is not None]
            return float(np.mean(v)) if v else None
        n_true = mean("n_true")
        n_obj = mean("n_objects")
        frag = mean("fragmentation")
        mr = mean("match_rate")
        md = mean("median_m")
        oseg = mean("over_seg")
        miss = mean("miss")
        fs = mean("frac_spurious")
        print(f"{name:<19}{n_true:>6.1f}{n_obj:>8.1f}{frag:>9.2f}{mr*100:>8.1f}%"
              f"{('n/a' if md is None else format(md, '.4f') + ' m'):>11}"
              f"{oseg:>7.1f}{miss:>7.1f}{fs*100:>7.1f}%")

    ok = True
    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    hr("判据")
    # 1) 离线复现必须与**当前默认策略**逐字段一致，否则后面所有对比都不可信
    #
    #    ⚠️ 这里比的是"库默认"与"离线重写"的对应臂，**不是**固定比 greedy：
    #    `configs/default.yaml` 里的 `assoc_strategy` 换了以后，
    #    这个自检要跟着换，否则会拿一个已经不是默认的臂去对。
    same = True
    detail = []
    for r in results:
        strat = str(r.get("assoc_strategy", "hungarian"))
        ref_name = "hungarian" if strat == "hungarian" else "greedy_label"
        lib = next(a for a in r["arms"] if a["arm"] == "lib_default")
        own = next(a for a in r["arms"] if a["arm"] == ref_name)
        keys = ("n_objects", "n_matched", "claims_1", "claims_ge2",
                "n_spurious", "median_m")
        diff = [k for k in keys if lib.get(k) != own.get(k)]
        same = same and not diff
        detail.append(f"{r['room']}:{strat}"
                      + ("一致" if not diff else f"不一致{diff}"))
    check("离线复现 == 库默认策略（同一批观测、同一打分函数）", same,
          "; ".join(detail))

    # 2) 有没有哪条臂明显更差 —— 有，说明关联是承重的
    def arm_mean(key: str, name: str) -> Optional[float]:
        v = [a[key] for r in results for a in r["arms"]
             if a["arm"] == name and a.get(key) is not None]
        return float(np.mean(v)) if v else None

    base = arm_mean("match_rate", "lib_default")
    worse = {n: arm_mean("match_rate", n) for n in STRATEGIES
             if n not in ("lib_default", "greedy_label", "hungarian")}
    worse = {k: v for k, v in worse.items() if v is not None and base is not None
             and v < base - 0.02}
    check("存在明显更差的关联策略（证明这块是承重的，不是装饰）",
          len(worse) > 0,
          "；".join(f"{k} {v*100:.1f}% vs 默认 {base*100:.1f}%"
                   for k, v in sorted(worse.items())) or "所有变体都不差于默认")

    # 3) 有没有哪条臂**明显更好** —— 有，说明当前默认选错了
    better = {n: arm_mean("match_rate", n) for n in STRATEGIES
              if n not in ("lib_default", "greedy_label", "hungarian")}
    better = {k: v for k, v in better.items() if v is not None and base is not None
              and v > base + 0.02}
    if better:
        print("  [NOTE] 有变体好于默认，应当考虑改默认："
              + "；".join(f"{k} {v*100:.1f}% vs {base*100:.1f}%"
                          for k, v in sorted(better.items())))
    else:
        print("  [INFO] 没有任何变体明显好于当前默认（默认不是被随手选的）")

    # 4) 旧默认（逐观测贪心）现在必须在表里，且必须**差于**当前默认
    gra = arm_mean("match_rate", "greedy_label")
    if base is not None and gra is not None:
        check("旧默认（greedy）确实差于当前默认（换默认这个决定有依据）",
              gra < base, f"greedy {gra*100:.1f}% vs 默认 {base*100:.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n产物已写入 {out}")
    print("\n" + "=" * 78)
    print("结论: " + ("判据通过 ✓" if ok else "存在不通过项 ✗（如实记录）"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
