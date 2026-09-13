# -*- coding: utf-8 -*-
"""在**真实** 2D-3D-S 全景场景上跑通下游链路，并做一次关键消融。

这个脚本回答两件事
================
1. **"后续的操作和现在的保持差不多"到底成不成立？**
   全景被包成普通 `RGBDFrame` 后，`MapBuilder → SemanticMap → 查询`
   这条链路**一行都不用改**就能跑。这里就用真实数据跑一遍。

2. **精确等距柱状反投影值不值得？**（消融）
   全景不是针孔相机。如果用"等效针孔内参"去反投影 360° 全景，
   远离光轴的像素会系统性错位。脚本用**同一份全景、同一份检测**，
   只切换反投影模型（`equirect` vs `pinhole`），比较地图级定位误差 ——
   把"几何上正确"这件事**量化**成一个可以引用的差值。

检测从哪来：**用 GT 框当检测**（不是真的检测器）
=============================================
把 GT 轴对齐框投到全景图上得到 2D 框，直接当作 `Detection2D` 注入。
这样评的是 **2D→3D→地图** 这一段（正是我们改过的部分），
而**不是**检测器本身。报告里必须写清这一点，否则会被误读成
"目标检测指标"。项目里 ROS2 延迟测试也用同样的注入手法。

跑法：
    python scripts/37_eval_pano_grounding.py --rooms office_6 hallway_6 --max-views 48
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config  # noqa: E402
from roboground.data.pano_scene import (  # noqa: E402
    PanoScene,
    load_scene,
    select_location,
    visible_gt,
)
from roboground.data.stanford2d3d import list_locations  # noqa: E402

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")
#: 本项目其余部分（RL / 查询 / ROS2）用的提示词集合，保持同一种写法
PROMPTS = ["chair", "table", "door", "bookcase", "sofa", "board",
           "window", "column", "beam", "stairs"]


def hr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def make_detections(scene: PanoScene, items: List[Dict[str, Any]],
                    label_map: Optional[Dict[str, str]] = None):
    """把投到全景图上的 GT 框变成 `Detection2D`（**当作检测器用**）。

    ⚠️ 用 `uv_parts` 而不是 `uv`：跨 ±180° 接缝的物体会被拆成两段，
    一个 bbox 不能越出图像边界。只用 `uv`（= 第一段）会把它切成一半。
    """
    from roboground.types import Detection2D

    dets = []
    for it in items:
        label = it["label"]
        if label_map:
            label = label_map.get(label, label)
        for (u0, v0, u1, v1) in it.get("uv_parts") or [it["uv"]]:
            dets.append(Detection2D(
                label=str(label), score=1.0,
                bbox=np.array([u0, v0, u1, v1], dtype=np.float64),
                prompt=str(label)))
    return dets


def build_map(cfg, scene: PanoScene, dets, *, projection: str,
              label_map: Optional[Dict[str, str]] = None):
    """用给定反投影模型建一张语义地图。

    `projection="pinhole"` 是**消融对照**：强行让几何层走等效针孔通路。
    """
    from roboground.mapping import MapBuilder

    frame = scene.frame()
    frame.meta["projection"] = projection
    if projection != "equirect":
        # 对照臂：抹掉光心元数据，退回普通针孔语义
        frame.meta.pop("center", None)

    prompts = sorted({d.label for d in dets})
    b = MapBuilder(cfg, prompts=prompts or PROMPTS)
    # 注入 GT 检测（替换掉感知流水线，其余链路完全不变）
    b.pipeline.run = lambda f, prompts=None: dets          # type: ignore[assignment]
    smap = b.build_from_frames([frame])
    return smap, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--rooms", nargs="*",
                    default=["office_6", "hallway_6", "office_27"])
    ap.add_argument("--max-views", type=int, default=48)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--out", default="runs/37_eval_pano_grounding.json")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2

    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("project.verbose", False)

    from roboground.eval.benchmark import map_level_localization

    results: List[Dict[str, Any]] = []
    for room in args.rooms:
        try:
            loc = select_location(args.root, room=room, min_frames=8)
        except (KeyError, ValueError) as e:
            print(f"[skip] {room}: {e}")
            continue

        hr(f"{room}：{loc.uuid[:12]}，{len(loc.frame_ids)} 个真实视角")
        t0 = time.time()
        scene = load_scene(args.root, loc, width=args.width, height=args.height,
                           max_frames=args.max_views, max_depth=args.max_range)
        build_s = time.time() - t0
        print(f"  全景: {scene.panorama.rgb.shape[1]}x{scene.panorama.rgb.shape[0]}  "
              f"像素覆盖 {scene.coverage*100:.1f}% / 立体角 "
              f"{scene.panorama.solid_angle_coverage*100:.1f}%  用时 {build_s:.1f}s")

        # ---- 只拿"真的看得见"的 GT 当检测 ----
        vis = visible_gt(scene, max_range_m=args.max_range)
        n_gt_all = scene.gt_within(args.max_range)[0].shape[0]
        print(f"  GT(≤{args.max_range:.0f} m) {n_gt_all} 个，其中**可见** {len(vis)} 个")
        if not vis:
            print("  [skip] 没有可见 GT，跳过")
            continue

        dets = make_detections(scene, vis)
        gt_boxes = np.stack([v["box"] for v in vis], axis=0)

        row: Dict[str, Any] = {
            "room": room, "uuid": scene.uuid[:12],
            "frames_used": scene.frames_used,
            "coverage": scene.coverage,
            "solid_angle_coverage": scene.panorama.solid_angle_coverage,
            "n_gt_in_range": int(n_gt_all), "n_gt_visible": int(len(vis)),
            "build_seconds": build_s,
        }

        # ---- 两条臂：精确等距柱状 vs 等效针孔 ----
        for arm in ("equirect", "pinhole"):
            try:
                smap, b = build_map(cfg, scene, dets, projection=arm)
            except Exception as e:                        # noqa: BLE001
                print(f"  [FAIL] {arm} 建图异常：{type(e).__name__}: {e}")
                row[f"{arm}_error"] = f"{type(e).__name__}: {e}"
                continue
            loc_m = map_level_localization(smap, [gt_boxes])
            row[f"{arm}_map_objects"] = int(smap.num_objects)
            row[f"{arm}_voxels"] = int(smap.num_voxels)
            for k, v in loc_m.items():
                if isinstance(v, (int, float, np.floating)):
                    row[f"{arm}_{k}"] = float(v)
            # ★ 键名要从**实际返回**里取，不能凭猜。
            #   第一版我猜了 mean_distance/median_error 之类，结果全落空，
            #   汇总表一路打 "n/a" —— 而逐臂那行又打出了真实数值，
            #   很容易被误读成"指标缺失"。真实键名是 median_m / mean_m / p90_m …，
            #   这里按优先级找，找不到就明确写 None（不编数字）。
            err = None
            for key in ("median_m", "mean_m", "median_distance", "mean_distance",
                        "median_error", "mean_error", "distance_mean"):
                if key in loc_m and loc_m[key] is not None:
                    err = float(loc_m[key])
                    row[f"{arm}_loc_err_key"] = key
                    break
            row[f"{arm}_loc_err_m"] = err
            print(f"  [{arm:<8}] 地图物体 {smap.num_objects:>3}  体素 {smap.num_voxels:>7}  "
                  f"定位误差"
                  + ("n/a" if err is None else f" {err:.4f} m ({row.get(f'{arm}_loc_err_key')})")
                  + f"  匹配率 {loc_m.get('match_rate', float('nan')):.4f}"
                  + f"  ≤0.5m {loc_m.get('within_0.50m', float('nan'))*100:.1f}%")
        results.append(row)

    hr("汇总")
    print(f"{'房间':<14}{'视角':>5}{'可见GT':>7}{'匹配率':>9}{'≤0.5m':>9}"
          f"{'中位误差':>11}{'针孔中位误差':>14}{'改善':>9}")
    for r in results:
        a, b_ = r.get("equirect_loc_err_m"), r.get("pinhole_loc_err_m")
        mr = r.get("equirect_match_rate")
        w5 = r.get("equirect_within_0.50m")
        def f(x, fmt="{:.4f}"):
            return "n/a" if x is None else fmt.format(x)
        imp = ("n/a" if (a is None or b_ in (None, 0)) else f"{(1 - a / b_) * 100:+.1f}%")
        print(f"{r['room']:<14}{r['frames_used']:>5}{r['n_gt_visible']:>7}"
              f"{f(mr):>9}{f(w5, '{:.1%}'):>9}{f(a):>11}{f(b_):>14}{imp:>9}")

    ok = True
    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    hr("判据")
    check("下游链路在全景上跑通（每条臂都产出了地图）",
          all(r.get("equirect_map_objects", 0) > 0 for r in results),
          "; ".join(f"{r['room']}:{r.get('equirect_map_objects')}" for r in results))
    check("等距柱状臂的定位误差键名可识别（不出现 n/a）",
          all(r.get("equirect_loc_err_m") is not None for r in results),
          "; ".join(str(r.get("equirect_loc_err_key")) for r in results))
    pairs = [(r.get("equirect_loc_err_m"), r.get("pinhole_loc_err_m"))
             for r in results]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if pairs:
        a_m = float(np.mean([a for a, _ in pairs]))
        b_m = float(np.mean([b for _, b in pairs]))
        check("精确等距柱状反投影优于等效针孔（消融）", a_m <= b_m,
              f"均值 {a_m:.4f} m vs {b_m:.4f} m"
              + (f"（改善 {(1-a_m/b_m)*100:+.1f}%）" if b_m else ""))
        check("等距柱状臂的匹配率明显更高",
              all((r.get("equirect_match_rate") or 0)
                  > (r.get("pinhole_match_rate") or 0)
                  for r in results if r.get("pinhole_match_rate") is not None),
              "; ".join(f"{r['room']}: "
                        f"{(r.get('equirect_match_rate') or 0):.4f} vs "
                        f"{(r.get('pinhole_match_rate') or 0):.4f}" for r in results))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物已写入 {out}")
    print("\n" + "=" * 74)
    print("结论: " + ("通过 ✓" if ok else "存在不通过项 ✗"))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
