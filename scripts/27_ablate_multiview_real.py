#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""27 · 严谨版消融：多视角融合的收益，究竟来自"新信息"还是"平均降方差"？

要回答的质疑
==========
项目文档里写着一句"真实单帧 0.553 m → 多视角 0.262 m（−52%）"。
这句话有两个问题（都是本脚本要修的）：

1. **数字复现不出来** —— `0.553` 在任何 `runs/` 产物里都找不到，
   同一条命令重跑现在是 `0.381 m`；而 `0.262` 来自一份**只有 2 个类别、
   19 个物体**的旧产物，与单帧那次的场景集并不相同。
2. **机制说不清** —— SUN RGB-D 每个场景只有**一张**图，"多视角"是把这张图
   反投影成点云后**重渲染**出来的。所以融合 N 个视角 = 对**同一份观测**
   采样 N 次。那 −52%/−32% 到底是"多视角信息增益"还是"平均降方差"？

本脚本用一个**判定性对照**把它分开（三臂，同一场景集、同一份代码）：

    arm A  single    ：只用那 1 张真实图               ← 基线
    arm B  dupN      ：把同一张真实图**复制 N 份**      ← 纯平均（零新信息）
    arm C  virtN     ：把同一张图反投影后重渲染 N 个视角 ← 项目声称的"多视角"

判据：
  · 若 **B ≈ C** ⇒ 收益几乎全部来自"平均降方差"，**不是**多视角信息增益
    （因为 B 完全没有新视角，却拿到了同样的改善）；
  · 若 **C 明显优于 B** ⇒ 重渲染确实带来了一点新东西（视角相关的采样差异）。

产出：`runs/ablation_multiview.json` + 控制台表格

用法::

    python scripts/27_ablate_multiview_real.py --scenes 5 --views 2 4 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config                          # noqa: E402
from roboground.utils.io import save_json                            # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity       # noqa: E402

log = get_logger("ablate.multiview")

#: 与 `scripts/05_benchmark.py` 保持一致，保证可比
PROMPTS = ["chair", "table", "desk", "monitor", "door", "window",
           "bookshelf", "trash can", "box", "bottle", "sofa", "bed"]


def pick_scenes(cfg, n_scenes: int, max_side: int = 320):
    """按"GT 框最多的场景优先"选，与 `05_benchmark.py` 的选法一致。

    返回 [(scene_idx, scene), ...] —— **一次选定，三个臂共用**，
    这是"同场景集"的保证。
    """
    from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene

    index_path = str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz")
    index = load_scene_index(index_path)
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))
    wanted = {p.lower() for p in PROMPTS}

    out = []
    for pos in np.argsort(-counts):
        i = int(pos)
        if len(out) >= n_scenes:
            break
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = {str(x).lower() for x in np.asarray(index["label_flat"][off0:off1]).ravel()}
        if not (labels & wanted):
            continue
        scene = load_sunrgbd_scene(index, i, max_depth=8.0)
        if scene is None or scene.boxes_3d.shape[0] == 0:
            continue
        if max_side and max(scene.shape) > max_side:
            h, w = scene.shape
            s = max_side / float(max(h, w))
            scene = load_sunrgbd_scene(index, i, max_depth=8.0,
                                       resize=(int(round(w * s)), int(round(h * s))))
        out.append((i, scene))
    return out


def build_and_eval(cfg, frames: List, scene, prompts: List[str]) -> Dict:
    """建图 + 地图级定位评测（按标签贪心一对一匹配）。"""
    from roboground.eval.benchmark import map_level_localization
    from roboground.mapping import MapBuilder

    b = MapBuilder(cfg, prompts=prompts)
    smap = b.build_from_frames(frames)
    loc = map_level_localization(smap, [scene.boxes_3d])
    return {
        "n_map_objects": int(smap.num_objects),
        "n_voxels": int(smap.num_voxels),
        **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
           for k, v in loc.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--views", type=int, nargs="+", default=[2, 4, 6],
                    help="要对比的视角数")
    ap.add_argument("--radius", type=float, default=1.8)
    ap.add_argument("--max-side", type=int, default=320)
    ap.add_argument("--out", default="runs/ablation_multiview.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("perception.prompts", PROMPTS)
    cfg.set("project.verbose", False)

    from roboground.data.virtual_camera import scene_sequence_from_cloud

    scenes = pick_scenes(cfg, args.scenes, args.max_side)
    if not scenes:
        log.error("没选到场景，检查 SUN RGB-D 索引")
        return 2

    print("=" * 90)
    print("多视角融合消融：收益来自「新信息」还是「平均降方差」？")
    print("=" * 90)
    print(f"场景集（三个臂共用）：{ [i for i, _ in scenes] }")
    for i, s in scenes:
        print(f"  #{i}  {s.sequence.split('/')[-1][:52]}  "
              f"图像 {s.color.shape[:2]}  GT {len(s.labels)} 个")
    print(f"参数：radius={args.radius} m，max_side={args.max_side}，"
          f"detector=stub（用 GT 框，隔离几何链路）")

    results: Dict[str, List[Dict]] = {}

    def run_arm(name: str, frames_for_scene):
        per_scene = []
        for idx, scene in scenes:
            prompts = sorted({str(x).lower() for x in scene.labels})
            frames = frames_for_scene(scene)
            try:
                per_scene.append(build_and_eval(cfg, frames, scene, prompts))
            except Exception as exc:                     # pragma: no cover
                log.warn(f"{name} 场景 #{idx} 失败：{exc}")
        results[name] = per_scene
        med = [r["median_m"] for r in per_scene if np.isfinite(r.get("median_m", np.nan))]
        n = sum(int(r.get("n", 0)) for r in per_scene)
        print(f"\n  {name:<10} 场景 {len(per_scene)}  匹配物体 n={n}  "
              f"中位误差 {np.median(med):.4f} m（各场景 {[round(m, 3) for m in med]}）")

    # ---- arm A：单帧真实 ----
    run_arm("single", lambda s: [s.to_frame()])

    for k in args.views:
        # ---- arm B：同一张图复制 k 份（零新信息，纯平均）----
        def dup(s, k=k):
            f0 = s.to_frame()
            out = []
            for j in range(k):
                g = f0.copy() if hasattr(f0, "copy") else f0
                g.meta = dict(f0.meta)
                g.frame_id = f"{f0.frame_id}_dup{j}"
                out.append(g)
            return out

        # ---- arm C：反投影后重渲染 k 个视角 ----
        def virt(s, k=k):
            return scene_sequence_from_cloud(s, num_frames=k, radius=args.radius,
                                            max_points=120_000, splat=2)

        run_arm(f"dup{k}", dup)
        run_arm(f"virt{k}", virt)

    # ---------------- 汇总 ----------------
    def summarize(name: str) -> Dict:
        rs = results[name]
        med = [r["median_m"] for r in rs if np.isfinite(r.get("median_m", np.nan))]
        return {
            "n": int(sum(int(r.get("n", 0)) for r in rs)),
            "median_m": float(np.median(med)) if med else float("nan"),
            "mean_m": float(np.mean([r["mean_m"] for r in rs if np.isfinite(r.get("mean_m", np.nan))])) if med else float("nan"),
            "within_0.25m": float(np.mean([r["within_0.25m"] for r in rs])),
            "objects": int(sum(r["n_map_objects"] for r in rs)),
        }

    summary = {k: summarize(k) for k in results}
    base = summary["single"]["median_m"]

    print("\n" + "=" * 90)
    print("汇总（同场景集 / 同代码 / 同 prompts）")
    print("=" * 90)
    print(f"{'臂':<10}{'匹配n':>7}{'地图物体':>9}{'中位误差(m)':>12}{'vs 单帧':>10}{'≤0.25m':>9}")
    for k, v in summary.items():
        delta = f"{(v['median_m'] / base - 1) * 100:+.1f}%" if base else "-"
        print(f"{k:<10}{v['n']:>7}{v['objects']:>9}{v['median_m']:>12.4f}{delta:>10}"
              f"{v['within_0.25m']:>9.3f}")

    # ---------------- 判定 ----------------
    print("\n" + "=" * 90)
    print("判定")
    print("=" * 90)
    verdict = {}
    for k in args.views:
        b, c = summary[f"dup{k}"], summary[f"virt{k}"]
        gain_dup = (b["median_m"] / base - 1) * 100
        gain_virt = (c["median_m"] / base - 1) * 100
        extra = (c["median_m"] / b["median_m"] - 1) * 100
        verdict[k] = {"gain_dup_pct": gain_dup, "gain_virt_pct": gain_virt,
                      "virt_vs_dup_pct": extra}
        print(f"  {k} 视角：复制 {gain_dup:+.1f}%　重渲染 {gain_virt:+.1f}%　"
              f"重渲染相对复制 {extra:+.1f}%")
    print("""
  读法：
    · "复制 k 份"用的是**同一张图**，零新信息。它若也能大幅改善，
      说明改善主要来自**对同一观测的多次采样平均（降方差）**；
    · "重渲染 k 视角"相对"复制"的**增量**才是"多视角"真正贡献的部分。
""")

    save_json({"config": vars(args), "scenes": [i for i, _ in scenes],
               "summary": summary, "verdict": verdict,
               "per_scene": {k: v for k, v in results.items()}},
              Path(args.out))
    log.ok(f"已保存 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
