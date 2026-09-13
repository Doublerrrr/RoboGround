# -*- coding: utf-8 -*-
"""全景管线的**可复现统计**：把报告里的数字都变成脚本产物。

为什么要单独一个脚本
==================
项目规矩是「报告里每个数字都要能追到一个 `runs/` 产物」。
`docs/多视角全景融合报告.md` 里有一批统计量（覆盖随视角数的单调性、
`max_depth` 前后对照、3,300 帧光轴仰角分布），它们之前只在临时脚本里
跑过一次 —— 临时脚本删掉之后，那些数字就**不可复现**了。这个脚本把它们固化。

跑法：
    python scripts/36_pano_dataset_stats.py --room office_6 --views 6 12 24 48
    python scripts/36_pano_dataset_stats.py --elevation-locations 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.pano_scene import load_scene, select_location  # noqa: E402
from roboground.data.stanford2d3d import list_locations  # noqa: E402

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")


def hr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ==========================================================================
# 1) 覆盖随视角数单调增长（"真实多视角确实带来信息增量"的证据）
# ==========================================================================
def coverage_vs_views(root: Path, loc, view_counts: List[int],
                      width: int, height: int) -> List[Dict[str, Any]]:
    n_avail = len(loc.frame_ids)
    rows: List[Dict[str, Any]] = []
    print(f"采集点 {loc.uuid[:12]} / {loc.room}，可用视角 {n_avail}")
    print(f"{'视角数':>6}{'像素覆盖':>10}{'立体角覆盖':>12}{'仰角范围':>18}{'耗时':>9}")
    for nv in view_counts:
        nv = min(int(nv), n_avail)
        t0 = time.time()
        sc = load_scene(root, loc, width=width, height=height,
                        max_frames=nv, with_gt=False)
        dt = time.time() - t0
        lo, hi = sc.panorama.elevation_span_deg()
        dv = sc.panorama.depth_m[sc.panorama.depth_m > 0]
        rows.append({
            "views_requested": int(nv), "views_used": int(sc.frames_used),
            "pixel_coverage": sc.coverage,
            "solid_angle_coverage": sc.panorama.solid_angle_coverage,
            "elev_lo_deg": lo, "elev_hi_deg": hi,
            "range_min_m": float(dv.min()) if dv.size else 0.0,
            "range_max_m": float(dv.max()) if dv.size else 0.0,
            "seconds": dt,
        })
        print(f"{sc.frames_used:>6}{sc.coverage*100:>9.2f}%"
              f"{sc.panorama.solid_angle_coverage*100:>11.2f}%"
              f"{lo:>9.1f}..{hi:<8.1f}{dt:>8.1f}s")
    return rows


# ==========================================================================
# 2) max_depth 现在作用在**斜距**上（修 bug 前后对照）
# ==========================================================================
def max_depth_semantics(root: Path, loc, depths: List[float],
                        width: int, height: int) -> List[Dict[str, Any]]:
    print("max_depth 应当直接卡住**存下来的斜距**。")
    print("（修复前它作用在 z 深度上，于是设 8.0 m 却融合出 9.91 m —— "
          "因为边缘像素 z→斜距要除以 d_cam.z，最低 0.35 倍。）")
    print(f"\n{'max_depth':>10}{'实测最大斜距':>14}{'像素覆盖':>10}")
    rows = []
    for md in depths:
        sc = load_scene(root, loc, width=width, height=height,
                        max_frames=len(loc.frame_ids), max_depth=md,
                        with_gt=False)
        dv = sc.panorama.depth_m[sc.panorama.depth_m > 0]
        mx = float(dv.max()) if dv.size else 0.0
        rows.append({"max_depth": float(md), "range_max_m": mx,
                     "pixel_coverage": sc.coverage})
        print(f"{md:>10.1f}{mx:>13.2f}m{sc.coverage*100:>9.2f}%")
    return rows


# ==========================================================================
# 3) 光轴仰角分布（"水平环带而非全球面"这一结论的证据）
# ==========================================================================
def elevation_stats(root: Path, n_locations: int,
                    min_frames: int = 8) -> Dict[str, Any]:
    locs = [l for l in list_locations(root) if len(l.frame_ids) >= min_frames]
    if not locs:
        raise ValueError("没有满足条件的采集点")
    idx = np.linspace(0, len(locs) - 1, num=min(n_locations, len(locs))).round().astype(int)
    locs = [locs[int(i)] for i in np.unique(idx)]

    els: List[float] = []
    per_loc: List[Dict[str, Any]] = []
    for loc in locs:
        e = []
        for f in loc.frame_ids:
            R = np.asarray(loc.pose(f).R, dtype=np.float64).reshape(3, 3)
            fwd = R.T @ np.array([0.0, 0.0, 1.0])
            e.append(float(np.degrees(np.arcsin(np.clip(fwd[2], -1.0, 1.0)))))
        e = np.asarray(e)
        els.append(e)
        per_loc.append({"room": loc.room, "uuid": loc.uuid[:12],
                        "n_frames": int(e.size),
                        "el_min": float(e.min()), "el_max": float(e.max()),
                        "el_median": float(np.median(e))})
    E = np.concatenate(els)

    out = {
        "n_locations": len(locs), "n_frames": int(E.size),
        "el_min": float(E.min()), "el_max": float(E.max()),
        "el_median": float(np.median(E)),
        "frames_beyond_pm30": int(((E < -30) | (E > 30)).sum()),
        "locations_all_within_pm25": int(sum(
            1 for r in per_loc if r["el_min"] >= -25 and r["el_max"] <= 25)),
        "per_location": per_loc,
    }
    print(f"统计 {out['n_locations']} 个采集点 / {out['n_frames']} 个真实视角：")
    print(f"  光轴仰角 min {out['el_min']:+.1f}°  max {out['el_max']:+.1f}°  "
          f"中位 {out['el_median']:+.1f}°")
    print(f"  超出 ±30° 的视角数: **{out['frames_beyond_pm30']}**"
          "  ← 为 0 说明采集是**水平环带**，天顶/天底从未被观测")
    print(f"  光轴全部落在 ±25° 内的采集点: {out['locations_all_within_pm25']}"
          f"/{out['n_locations']}")
    return out


# ==========================================================================
# 4) 同一光心：融合成立的前提
# ==========================================================================
def center_spread_stats(root: Path, n_locations: int) -> Dict[str, Any]:
    locs = [l for l in list_locations(root) if len(l.frame_ids) >= 8]
    spreads = []
    for loc in locs[:n_locations]:
        C = np.array([np.asarray(loc.pose(f).camera_center(), dtype=np.float64)
                      for f in loc.frame_ids])
        spreads.append(float(np.max(np.linalg.norm(C - C.mean(axis=0), axis=1))))
    s = np.asarray(spreads)
    out = {"n_locations": int(s.size), "spread_min_m": float(s.min()),
           "spread_median_m": float(np.median(s)), "spread_max_m": float(s.max())}
    print(f"同一采集点内 N 个视角的**光心离散度**（{out['n_locations']} 个采集点）：")
    print(f"  min {out['spread_min_m']:.3e} m   中位 {out['spread_median_m']:.3e} m   "
          f"max {out['spread_max_m']:.3e} m")
    print("  → 远小于 0.05 m 的告警门限，说明这些视角确实是**原地转一圈**，"
          "融合成一张全景在几何上成立。")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--room", default="office_6")
    ap.add_argument("--views", type=int, nargs="+", default=[6, 12, 24, 48])
    ap.add_argument("--depths", type=float, nargs="+", default=[4.0, 8.0, 20.0])
    ap.add_argument("--elevation-locations", type=int, default=60)
    ap.add_argument("--center-locations", type=int, default=40)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--out", default="runs/36_pano_dataset_stats.json")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2

    result: Dict[str, Any] = {"root": str(args.root)}
    loc = select_location(args.root, room=args.room, min_frames=8)

    hr(f"1) 覆盖随视角数增长（{loc.room}，{args.width}x{args.height}）")
    result["coverage_vs_views"] = coverage_vs_views(
        args.root, loc, args.views, args.width, args.height)

    hr("2) max_depth 语义（应卡住斜距）")
    result["max_depth"] = max_depth_semantics(
        args.root, loc, args.depths, args.width, args.height)

    hr("3) 光轴仰角分布 → 水平环带（天顶/天底未观测）")
    result["elevation"] = elevation_stats(args.root, args.elevation_locations)

    hr("4) 同一采集点的光心一致性")
    result["center_spread"] = center_spread_stats(args.root, args.center_locations)

    hr("判据")
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    covs = [r["pixel_coverage"] for r in result["coverage_vs_views"]]
    check("覆盖随视角数单调不减", all(b >= a - 1e-9 for a, b in zip(covs, covs[1:])),
          " → ".join(f"{c*100:.1f}%" for c in covs))
    check("更多视角确实带来增益（末项 > 首项）",
          covs[-1] > covs[0] * 1.05, f"{covs[0]*100:.1f}% → {covs[-1]*100:.1f}%")

    for r in result["max_depth"]:
        check(f"max_depth={r['max_depth']:.0f} m 时斜距未超限",
              r["range_max_m"] <= r["max_depth"] + 1e-3,
              f"实测最大 {r['range_max_m']:.2f} m")

    e = result["elevation"]
    check("光轴仰角不超过 ±30°（水平环带）", e["frames_beyond_pm30"] == 0,
          f"{e['n_frames']} 帧中 {e['frames_beyond_pm30']} 帧超限；"
          f"范围 {e['el_min']:+.1f}°~{e['el_max']:+.1f}°")
    check("光心离散度远小于 0.05 m 门限",
          result["center_spread"]["spread_max_m"] < 0.001,
          f"最大 {result['center_spread']['spread_max_m']:.3e} m")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物已写入 {out}")

    print("\n" + "=" * 74)
    print("结论: " + ("全部通过 ✓" if ok else "存在不通过项 ✗"))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
