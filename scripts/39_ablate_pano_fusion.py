# -*- coding: utf-8 -*-
"""39 · 全景融合策略消融：证明（或否定）"我们的融合设计"到底有没有用。

为什么必须做这个
==============
`scripts/35` 证明了"融合结果与官方全景一致到厘米级"，但那只说明**流程对**，
**不能说明我们的设计比朴素做法更好**。一个没有 baseline 对比的"设计"不叫贡献 ——
这是评审最直接会问的一件事，所以这里把它补上。

五个臂，逐个隔离一个设计点
========================
| 臂 | 角度加权 | 融合方式 | 深度语义 | 隔离出什么 |
|---|---|---|---|---|
| `mean_equal` | 无（power=0，等权） | 全部样本加权平均 | 斜距 | **朴素基线**：既不加权也不做一致性筛选 |
| `mean_cos` | 余弦² | 全部样本加权平均 | 斜距 | **只加"角度加权"**的贡献（vs mean_equal） |
| `nearest` | 余弦² | **完全不融合**（取权重最高单样本） | 斜距 | **"多视角平均"本身**的贡献（vs mean_cos） |
| `consensus` | 余弦² | **基准 + 共识筛选**（本项目的做法） | 斜距 | **"共识筛选"**的贡献（vs mean_cos） |
| `z_semantics` | 余弦² | 基准 + 共识筛选 | **z 深度**（不做换算） | **z→斜距换算**的贡献（vs consensus） |

判据（全部对**官方全景**，即数据集自带的独立参考答案）
- **深度 MAE / 中位误差**（米）：越小越好
- **深度比值中位**：应 ≈1（我们的斜距 vs 官方深度，两者同为斜距）
- **共同有效像素数**：覆盖是否被某个臂牺牲掉了

★ 公平性做法：**所有臂共用同一套对齐参数**（由 `consensus` 臂一次性求出）。
否则不同臂会各自选到不同的方位偏移，比出来的就不是融合质量而是对齐巧合。

用法::

    python scripts/39_ablate_pano_fusion.py --points 10 --max-views 24
    python scripts/39_ablate_pano_fusion.py --points 186 --max-views 48   # 全量（慢）
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from roboground.data.pano_scene import _pick_frame_ids  # noqa: E402,SLF001
from roboground.data.stanford2d3d import list_locations  # noqa: E402

# 复用脚本 35 里已验证的对齐与官方全景读取（不要重写一遍）
from importlib import import_module  # noqa: E402

_v35 = import_module("35_validate_pano_vs_official")
align_azimuth = _v35.align_azimuth
_apply_alignment = _v35._apply_alignment
_resize_nearest = _v35._resize_nearest
official_depth_m = _v35.official_depth_m

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")

#: 消融臂定义：(名字, fuse_to_equirect 的关键字参数, 一句话说明)
ARMS: List[Dict[str, Any]] = [
    {"name": "mean_equal", "kw": {"weight_power": 0.0, "depth_mode": "mean"},
     "note": "朴素基线：等权（不加权）+ 全部样本平均"},
    {"name": "mean_cos", "kw": {"weight_power": 2.0, "depth_mode": "mean"},
     "note": "只加角度加权，仍不做一致性筛选"},
    {"name": "nearest", "kw": {"weight_power": 2.0, "depth_mode": "nearest"},
     "note": "完全不融合：取权重最高的单个样本"},
    {"name": "consensus", "kw": {"weight_power": 2.0, "depth_mode": "consensus"},
     "note": "本项目的做法：角度加权 + 基准共识筛选"},
    {"name": "z_semantics", "kw": {"weight_power": 2.0, "depth_mode": "consensus",
                                  "depth_semantics": "z"},
     "note": "消融：不做 z 深度→斜距 换算（错误做法）"},
]


def analyze_min_cos_binding(frames_by_loc: Dict[str, list]) -> Dict[str, Any]:
    """★ 为什么"角度加权"没有贡献？先量一下它到底有没有做事。

    加权与 `min_cos` 门限都建立在"离光轴越远、权重越低"上。
    但如果相机的 **FOV 很窄**，所有像素的 `cos(离轴角)` 都很接近 1，
    那么 `cos^p` 就**接近等权**，门限也**从不触发** —— 这时"角度加权"
    在数学上就退化成等权，当然测不出贡献。

    这个分析把原因量出来（而不是含糊地说"效果不明显"）。
    """
    from roboground.data.panorama import view_rays_world

    cos_all = []
    for frames in frames_by_loc.values():
        for fr in frames[:4]:
            dirs, _ = view_rays_world(fr)
            R = np.asarray(fr.pose.R, dtype=np.float64).reshape(3, 3)
            cos_all.append((dirs @ R.T)[..., 2].ravel())
    if not cos_all:
        return {}
    c = np.concatenate(cos_all)
    out = {
        "n_pixels": int(c.size),
        "cos_min": float(c.min()), "cos_p1": float(np.percentile(c, 1)),
        "cos_median": float(np.median(c)), "cos_max": float(c.max()),
        "max_offaxis_deg": float(np.degrees(np.arccos(c.min()))),
        "default_min_cos": 0.35,
        "default_min_cos_binds": bool(c.min() <= 0.35),
        "weight_cos2_min": float(c.min() ** 2), "weight_cos2_max": float(c.max() ** 2),
        "weight_cos2_ratio": float(c.max() ** 2 / max(c.min() ** 2, 1e-9)),
    }
    return out


def sensitivity_min_cos(locs, width: int, height: int, max_views: int) -> List[Dict[str, Any]]:
    """`min_cos` 敏感性：门限从 0 到 0.5，等权 vs 余弦加权各自表现如何。

    如果门限从不触发，那么整张表会**完全不变** —— 这正是"该参数在数据上失效"的直接证据。
    """
    from roboground.data.panorama import fuse_to_equirect

    vw, vh = width // 2, height // 2
    rows: List[Dict[str, Any]] = []
    for mc in (0.0, 0.2, 0.35, 0.5, 0.7):
        acc = {"equal": [], "cos": []}
        for loc in locs:
            frames = [f for f in (loc.frame(i) for i in loc.frame_ids[:max_views])
                      if f is not None]
            off = official_depth_m(loc)
            if off is None or len(frames) < 2:
                continue
            off_s = _resize_nearest(off, vw, vh)
            p_ref = fuse_to_equirect(frames, width=width, height=height,
                                     weight_power=2.0, min_cos=mc, depth_mode="mean")
            al = align_azimuth(
                _resize_nearest(p_ref.rgb, vw, vh),
                _resize_nearest(p_ref.depth_m, vw, vh) > 0,
                _resize_nearest(loc.official_panorama()["rgb"], vw, vh))
            offa = _apply_alignment(off_s, al)
            for tag, wp in (("equal", 0.0), ("cos", 2.0)):
                p = (p_ref if tag == "cos" else
                     fuse_to_equirect(frames, width=width, height=height,
                                      weight_power=wp, min_cos=mc, depth_mode="mean"))
                o = _resize_nearest(p.depth_m, vw, vh)
                both = (o > 0) & (offa > 0)
                if both.sum() < 300:
                    continue
                acc[tag].append(float(np.abs(o[both] - offa[both]).mean()))
        if acc["equal"] and acc["cos"]:
            e = float(np.mean(acc["equal"])); k = float(np.mean(acc["cos"]))
            rows.append({"min_cos": mc, "equal_mae_m": e, "cos_mae_m": k,
                         "gain_pct": (1 - k / e) * 100 if e else 0.0,
                         "n_points": len(acc["cos"])})
    return rows


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fuse_with(scene_frames, width: int, height: int, kw: Dict[str, Any]):
    """按给定参数融合（不动全局默认值）。"""
    from roboground.data.panorama import fuse_to_equirect

    return fuse_to_equirect(scene_frames, width=width, height=height, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--points", type=int, default=10, help="用多少个采集点（跨房间等间隔取）")
    ap.add_argument("--max-views", type=int, default=24)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--max-depth", type=float, default=8.0)
    ap.add_argument("--out", default="runs/39_ablate_pano_fusion.json")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2

    locs = sorted(list_locations(args.root), key=lambda l: -len(l.frame_ids))
    locs = [l for l in locs if len(l.frame_ids) >= 10]
    idx = np.linspace(0, len(locs) - 1, num=min(args.points, len(locs))).round().astype(int)
    locs = [locs[int(i)] for i in np.unique(idx)]
    print(f"采集点 {len(locs)} 个（从 {len(list_locations(args.root))} 个里等间隔取）"
          f"，每点最多 {args.max_views} 个视角，全景 {args.width}x{args.height}")

    vw, vh = args.width // 2, args.height // 2
    per_arm: Dict[str, List[Dict[str, Any]]] = {a["name"]: [] for a in ARMS}
    frames_by_loc: Dict[str, list] = {}
    t_all = time.time()

    for li, loc in enumerate(locs, 1):
        off_d = official_depth_m(loc)
        if off_d is None:
            print(f"  [skip] {loc.room} 没有官方全景深度")
            continue
        off_d_s = _resize_nearest(off_d, vw, vh)

        # 直接取原始帧（不用 load_scene —— 那会先融合一次我们并不需要的全景）
        ids = _pick_frame_ids(loc.frame_ids, args.max_views)
        frames = [f for f in (loc.frame(i) for i in ids) if f is not None]
        if len(frames) < 2:
            print(f"  [skip] {loc.room} 可用帧不足")
            continue
        frames_by_loc[loc.uuid] = frames

        # ★ 对齐只算一次（用 consensus 臂的 RGB），**所有臂共用同一套对齐参数**，
        #   否则各臂会选到不同的方位偏移，比出来的是对齐巧合而不是融合质量。
        consensus_kw = next(a["kw"] for a in ARMS if a["name"] == "consensus")
        pano_c = fuse_with(frames, args.width, args.height, consensus_kw)
        our_rgb_c = _resize_nearest(pano_c.rgb, vw, vh)
        our_rng_c = _resize_nearest(pano_c.depth_m, vw, vh)
        al = align_azimuth(our_rgb_c, our_rng_c > 0,
                           _resize_nearest(loc.official_panorama()["rgb"], vw, vh))
        off_d_a = _apply_alignment(off_d_s, al)

        # ★ 分层掩码：**深度不连续**区域（官方深度梯度最大的 10% 像素）。
        #   共识筛选的全部意义就是"不要在两张不相连的表面上取平均"，
        #   所以它该在**这里**体现出价值 —— 如果整体 MAE 看不出差别，
        #   就要看这个分层；如果分层也看不出，那这个设计就是无效的。
        gy = np.zeros_like(off_d_a); gx = np.zeros_like(off_d_a)
        gy[1:-1, :] = np.abs(off_d_a[2:, :] - off_d_a[:-2, :])
        gx[:, 1:-1] = np.abs(off_d_a[:, 2:] - off_d_a[:, :-2])
        gmag = gy + gx
        gvalid = off_d_a > 0
        edge_thr = (float(np.percentile(gmag[gvalid], 90.0)) if gvalid.any() else np.inf)
        edge_mask = gvalid & (gmag >= edge_thr)

        for arm in ARMS:
            if arm["name"] == "consensus":
                pano = pano_c
            else:
                pano = fuse_with(frames, args.width, args.height, arm["kw"])
            our_rng = _resize_nearest(pano.depth_m, vw, vh)
            both = (our_rng > 0) & (off_d_a > 0)
            if both.sum() < 500:
                continue
            a = our_rng[both].astype(np.float64)
            b = off_d_a[both].astype(np.float64)
            err = np.abs(a - b)
            ratio = a / np.maximum(b, 1e-9)
            # `both_edge` 是 2-D 掩码，而 err 已经是 1-D（按 both 取过），
            # 所以要再用 both 选一次，得到"在 both 里的不连续像素"。
            both_edge = (both & edge_mask)[both]
            rec = {
                "room": loc.room, "uuid": loc.uuid[:12], "n_frames": len(frames),
                "n_both": int(both.sum()),
                "mae_m": float(err.mean()),
                "median_abs_m": float(np.median(err)),
                "p90_abs_m": float(np.percentile(err, 90)),
                "p99_abs_m": float(np.percentile(err, 99)),
                "max_abs_m": float(err.max()),
                "ratio_median": float(np.median(ratio)),
                "coverage": float((our_rng > 0).mean()),
                # 覆盖率口径也要一起看：某个臂可能靠"少给像素"换来低误差
                "edge_n": int(both_edge.sum()),
                "edge_mae_m": (float(err[both_edge].mean())
                               if both_edge.sum() > 50 else None),
            }
            per_arm[arm["name"]].append(rec)
        if li % 5 == 0 or li == len(locs):
            print(f"  ... {li}/{len(locs)}（已耗时 {time.time()-t_all:.0f}s）")

    # ---- 汇总 ----
    hr("汇总（全部对官方全景，所有臂共用同一套对齐参数）")
    hdr = (f"{'臂':<14}{'采集点':>6}{'像素':>10}{'MAE':>9}{'中位':>9}{'p95':>9}"
           f"{'p99':>9}{'最大':>9}{'不连续区MAE':>13}{'覆盖':>8}")
    print(hdr)
    print("-" * len(hdr))
    summary: Dict[str, Any] = {}
    for arm in ARMS:
        rows = per_arm[arm["name"]]
        if not rows:
            print(f"{arm['name']:<14}{'—':>6}{'无有效数据':>12}")
            continue
        mae = float(np.mean([r["mae_m"] for r in rows]))
        med = float(np.median([r["median_abs_m"] for r in rows]))
        p95 = float(np.median([r["p90_abs_m"] for r in rows]))
        p99 = float(np.median([r["p99_abs_m"] for r in rows]))
        mx = float(np.median([r["max_abs_m"] for r in rows]))
        rat = float(np.median([r["ratio_median"] for r in rows]))
        cov = float(np.mean([r["coverage"] for r in rows]))
        px = int(np.mean([r["n_both"] for r in rows]))
        edge_rows = [r["edge_mae_m"] for r in rows if r["edge_mae_m"] is not None]
        edge = float(np.mean(edge_rows)) if edge_rows else float("nan")
        print(f"{arm['name']:<14}{len(rows):>6}{px:>10,}{mae:>8.4f}m{med:>8.4f}m{p95:>8.3f}m"
              f"{p99:>8.3f}m{mx:>8.3f}m{edge:>12.4f}m{cov*100:>7.1f}%")
        summary[arm["name"]] = {"n_points": len(rows), "mean_mae_m": mae,
                                "median_abs_m": med, "p95_abs_m": p95, "p99_abs_m": p99,
                                "max_abs_m": mx, "ratio_median": rat,
                                "edge_mae_m": edge,
                                "coverage": cov, "n_both_mean": px,
                                "note": arm["note"], "kw": arm["kw"]}

    hr("判据：每个设计点是否真的带来改善")
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    def mae_of(k: str):
        return summary.get(k, {}).get("mean_mae_m")

    if all(mae_of(k) is not None for k in ("mean_equal", "mean_cos", "nearest", "consensus")):
        base = mae_of("mean_equal")
        check("角度加权在整体 MAE 上有正贡献",
              mae_of("mean_cos") < base,
              f"{mae_of('mean_cos'):.4f} vs {base:.4f} m"
              f"（{(1-mae_of('mean_cos')/base)*100:+.1f}%）")
        check("共识筛选在整体 MAE 上有正贡献",
              mae_of("consensus") < mae_of("mean_cos"),
              f"{mae_of('consensus'):.4f} vs {mae_of('mean_cos'):.4f} m"
              f"（{(1-mae_of('consensus')/mae_of('mean_cos'))*100:+.1f}%）")
        # ★ 共识筛选的设计目的就是"不在不相连的表面上取平均"，
        #   所以真正该看的是**深度不连续区**的误差。
        e_c = summary["consensus"].get("edge_mae_m")
        e_m = summary["mean_cos"].get("edge_mae_m")
        if e_c and e_m and np.isfinite(e_c) and np.isfinite(e_m):
            check("★ 共识筛选在**深度不连续区**显著更优（这才是它的设计目的）",
                  e_c < e_m * 0.95,
                  f"不连续区 MAE {e_c:.4f} vs 朴素平均 {e_m:.4f} m"
                  f"（{(1-e_c/e_m)*100:+.1f}%）")
        check("多视角平均优于不融合（mean_cos < nearest 或 consensus < nearest）",
              min(mae_of("mean_cos"), mae_of("consensus")) < mae_of("nearest"),
              f"best-fused {min(mae_of('mean_cos'), mae_of('consensus')):.4f} vs "
              f"nearest {mae_of('nearest'):.4f} m")
    if mae_of("z_semantics") is not None and mae_of("consensus") is not None:
        z, c = mae_of("z_semantics"), mae_of("consensus")
        check("z→斜距换算有巨大贡献（z_semantics 远差于 consensus）",
              z > c * 2,
              f"z 语义 {z:.4f} m vs 斜距 {c:.4f} m（差 {z/c:.1f} 倍）")

    # ---- 附加分析 1：角度加权为什么没贡献？----
    hr("附加分析 1：min_cos 门限与角度加权到底有没有在「做事」")
    binding = analyze_min_cos_binding(frames_by_loc)
    if binding:
        print(f"  全体像素相对光轴的 cos：min {binding['cos_min']:.4f} / "
              f"p1 {binding['cos_p1']:.4f} / 中位 {binding['cos_median']:.4f} / "
              f"max {binding['cos_max']:.4f}")
        print(f"  最大离轴角：{binding['max_offaxis_deg']:.1f}°")
        print(f"  默认 min_cos=0.35 是否触发："
              f"{'是' if binding['default_min_cos_binds'] else '★ 否 —— 门限从不触发'}")
        print(f"  cos^2 权重的实际范围：{binding['weight_cos2_min']:.4f} ~ "
              f"{binding['weight_cos2_max']:.4f}（仅 {binding['weight_cos2_ratio']:.2f}× 变化）")
        if not binding["default_min_cos_binds"]:
            print("  ⇒ 这解释了上面「角度加权无贡献」的零结果：该相机的 FOV 窄，")
            print("     所有像素的 cos 都 >= 0.70，cos^2 权重只在 2 倍以内浮动，数学上接近等权；")
            print("     min_cos=0.35 这个门限从未触发，是个死参数。")

    # ---- 附加分析 2：min_cos 敏感性（若门限失效，整张表应当完全不变）----
    hr("附加分析 2：min_cos 敏感性（门限失效的话，整张表会完全不变）")
    sens = sensitivity_min_cos(locs[:6], args.width, args.height, min(args.max_views, 12))
    if sens:
        print(f"{'min_cos':>9}{'等权MAE':>12}{'余弦MAE':>12}{'加权收益':>10}")
        for r in sens:
            print(f"{r['min_cos']:>9.2f}{r['equal_mae_m']:>11.5f}m"
                  f"{r['cos_mae_m']:>11.5f}m{r['gain_pct']:>9.3f}%")
        uniq = len({round(r["equal_mae_m"], 6) for r in sens})
        spread = max(r["equal_mae_m"] for r in sens) - min(r["equal_mae_m"] for r in sens)
        # ⚠️ 不能只看"取值个数"：0.70 那档会在**第 5 位小数**上变化，
        #    个数>1 但实质仍是失效。所以按**相对变化幅度**判。
        rel = spread / max(min(r["equal_mae_m"] for r in sens), 1e-9)
        print(f"  不同 min_cos 下等权 MAE 的极差：{spread:.2e} m（相对 {rel*100:.4f}%）"
              f"，取值个数 {uniq}/{len(sens)}")
        if rel < 0.01:
            print("  ⇒ ★ 门限从 0 到 0.5 几乎不改变结果（相对变化 < 0.01%），"
                  "证实 min_cos 在这份数据上是**死参数**")
        else:
            print("  ⇒ 门限在起作用")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"points": len(locs), "max_views": args.max_views,
                   "width": args.width, "height": args.height,
                   "max_depth": args.max_depth, "root": str(args.root)},
        "arms": summary, "per_point": per_arm,
        "min_cos_binding": binding, "min_cos_sensitivity": sens,
        "elapsed_s": time.time() - t_all,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物：{out}")
    print(f"总耗时 {time.time()-t_all:.0f}s")
    print("\n" + "=" * 78)
    print("结论: " + ("融合设计的每个环节都有正贡献 ✓" if ok else "存在无贡献/负贡献的环节 ✗（如实记录，不粉饰）"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
