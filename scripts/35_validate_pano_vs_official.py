# -*- coding: utf-8 -*-
"""把「我们的多视角融合全景」和「官方全景」对齐后做定量比对。

为什么需要这个脚本
================
`tests/test_panorama.py` 用的是**合成**几何（已知答案），它证明的是
"融合算法按约定是对的"。但真实数据上还有两件合成数据证明不了的事：

  1. **融合结果真的对得上场景吗？** —— 需要一个**独立**的参考答案。
     2D-3D-S 官方在同一批真实位姿上给出了 `/pano/` 等距柱状全景，
     它的光心与我们的采集点光心**实测只差 1.6e-06 m**（同一个物理位置），
     所以它是一份合格的参考答案。
  2. **我们的实现有没有系统性错误？** 官方全景与我们全景的**方位角起点不同**
     （官方有 `camera_rt_matrix` / `final_camera_rotation`），
     必须先对齐再比，否则比出来的是"转了多少度"而不是"融合得准不准"。

对齐怎么做：**实测**，不猜。
用列剖面做互相关，搜索方位偏移 —— 脚本会打印"最佳偏移处的相关性"与
"零偏移处的相关性"，两者差距本身就是"确实存在一个真实旋转"的证据。

深度语义：官方 `/pano/depth` 也是 16-bit、`/512`、`65535` 为无数据。
对齐后我们比的是**斜距**（我们全景的 `depth_m`）与官方深度值 ——
如果官方是"沿某轴的 z 深度"，比对结果会随仰角呈现 `1/cos(el)` 的系统性偏差，
正好可以据此判定，脚本会把该诊断打出来。

用法
====
    python scripts/35_validate_pano_vs_official.py --room office_6
    python scripts/35_validate_pano_vs_official.py --all --n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.pano_scene import (  # noqa: E402
    load_scene,
    select_location,
)
from roboground.data.stanford2d3d import INVALID_RAW, list_locations  # noqa: E402

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")


def hr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ==========================================================================
# 对齐
# ==========================================================================
def _grad_mag(rgb: np.ndarray) -> np.ndarray:
    """梯度幅值（只看结构，天然抗曝光/色调差异）。"""
    g = np.asarray(rgb, dtype=np.float64).mean(axis=2)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gy[1:-1, :] = np.abs(g[2:, :] - g[:-2, :])
    gx[:, 1:-1] = np.abs(g[:, 2:] - g[:, :-2])
    return gx + gy


def align_azimuth(our_rgb: np.ndarray, our_valid: np.ndarray,
                  off_rgb: np.ndarray) -> Dict[str, Any]:
    """求官方全景与我们全景的对齐关系。

    ★ 这里有两个**踩过的坑**，都写下来免得再犯：

    **坑 1：用列剖面互相关会给出"看着很像"的错误答案。**
    我第一版用列亮度剖面做互相关，得到"最佳偏移 118.83°、相关性 0.497"，
    看着挺像回事 —— 但那只说明**低频剖面**像，逐像素根本没对上
    （梯度相关性只有 0.011，纯属噪声）。后来改用**梯度幅值相关**，
    并且**只在有覆盖的行上**算，才得到真实答案。

    **坑 2：官方全景的方位角方向是反的。**
    只允许"滚一圈"时，最好的相关性也只有 0.14；一旦把官方图的**列镜像**
    一下，相关性跳到 **0.569**。也就是说官方等距柱状的方位角**递增方向与我们相反**。
    这种"镜像"错误**不可能靠平移修正**，必须显式测试。
    所以这里把两种方位约定都试一遍，让数据选。

    仰角方向则**不需要**翻转，而且最佳仰角偏移正好是 **0 行** ——
    说明两边都以世界 z 轴（重力方向）为极轴，这一点是一致的。
    """
    H = our_rgb.shape[0]
    rows = np.flatnonzero(our_valid.any(axis=1))
    r0, r1 = int(rows.min()), int(rows.max())
    gour = _grad_mag(our_rgb)[r0:r1 + 1]

    out: Dict[str, Any] = {}
    for az_flip in (False, True):
        cand = off_rgb[:, ::-1] if az_flip else off_rgb
        goff = _grad_mag(cand)
        # 先粗扫方位（步长 2 列），再在邻域细化
        cors = [(float(np.corrcoef(gour.ravel(),
                                   np.roll(goff, s, axis=1)[r0:r1 + 1].ravel())[0, 1]), s)
                for s in range(0, cand.shape[1], 2)]
        _, s0 = max(cors)
        cors = [(float(np.corrcoef(gour.ravel(),
                                   np.roll(goff, s, axis=1)[r0:r1 + 1].ravel())[0, 1]), s)
                for s in range(s0 - 3, s0 + 4)]
        c_az, roll = max(cors)
        # 再找仰角偏移
        best = (c_az, roll, 0)
        for dy in range(-40, 41, 2):
            o = np.roll(np.roll(goff, roll, axis=1), dy, axis=0)[r0:r1 + 1]
            c = float(np.corrcoef(gour.ravel(), o.ravel())[0, 1])
            if c > best[0]:
                best = (c, roll, dy)
        out["az_flip" if az_flip else "no_flip"] = best

    key = max(out, key=lambda k: out[k][0])
    corr, roll, dy = out[key]
    W = our_rgb.shape[1]
    return {
        "az_flip": (key == "az_flip"),
        "roll_cols": int(roll),
        "roll_deg": float(roll / W * 360.0),
        "el_shift_rows": int(dy),
        "el_shift_deg": float(dy / H * 180.0),
        "corr_best": float(corr),
        "corr_other_convention": float(out["no_flip" if key == "az_flip" else "az_flip"][0]),
        "row_span": (r0, r1),
    }


def _apply_alignment(img: np.ndarray, al: Dict[str, Any]) -> np.ndarray:
    """把官方图按对齐结果变换到我们的坐标系。"""
    out = img[:, ::-1] if al["az_flip"] else img
    out = np.roll(out, al["roll_cols"], axis=1)
    out = np.roll(out, al["el_shift_rows"], axis=0)
    return out


def _resize_nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """最近邻缩放（纯 numpy，避免依赖顺序问题；深度图必须用最近邻）。"""
    H, W = img.shape[:2]
    yi = np.clip((np.arange(h) + 0.5) * H / h, 0, H - 1).astype(int)
    xi = np.clip((np.arange(w) + 0.5) * W / w, 0, W - 1).astype(int)
    return img[yi][:, xi]


def official_depth_m(loc) -> Optional[np.ndarray]:
    """官方全景深度 → 米，并把 `65535` 记为无效（0）。"""
    po = loc.official_panorama()
    if po is None or "depth_m" not in po:
        return None
    from PIL import Image

    p = loc._path("depth", -1, pano=True)          # noqa: SLF001
    if p is None:
        return None
    raw = np.asarray(Image.open(p)).astype(np.float32)
    d = raw / 512.0
    d[(raw <= 0) | (raw >= INVALID_RAW)] = 0.0
    return d


# ==========================================================================
# 单个场景的比对
# ==========================================================================
def compare_scene(scene, loc, *, width: int, height: int,
                  n_grid: int = 1024) -> Dict[str, Any]:
    """把我们的全景与官方全景在**同一分辨率**上比对。

    `width/height` 是**比对**用的尺寸（通常取全景的一半，够用且快）。
    两边的图都先缩到该尺寸，再做对齐与统计 —— 保证逐像素一一对应。
    """
    ours_rgb = _resize_nearest(scene.panorama.rgb, width, height)
    ours_rng = _resize_nearest(scene.panorama.depth_m, width, height)
    off_rgb_full = loc.official_panorama()["rgb"]
    off_d_full = official_depth_m(loc)
    if off_d_full is None:
        raise RuntimeError("官方全景没有深度图")

    off_rgb = _resize_nearest(off_rgb_full, width, height)
    off_d = _resize_nearest(off_d_full, width, height)

    # ---- 对齐（自动选方位约定 + 求偏移）----
    al = align_azimuth(ours_rgb, ours_rng > 0, off_rgb)
    off_rgb_a = _apply_alignment(off_rgb, al)
    off_d_a = _apply_alignment(off_d, al)

    # ---- RGB 比对：只比"两边都有覆盖"的像素 ----
    both = (ours_rng > 0) & (off_d_a > 0)
    if both.sum() > 100:
        rgb_mae = float(np.abs(ours_rgb.astype(np.float64)
                               - off_rgb_a.astype(np.float64))[both].mean())
        # 对照：不翻转方位（即"只滚一圈"的错误做法）能到多少
        al_bad = dict(al, az_flip=not al["az_flip"])
        off_bad = _apply_alignment(off_rgb, al_bad)
        corr_bad = float(np.corrcoef(_grad_mag(ours_rgb)[both].ravel(),
                                     _grad_mag(off_bad)[both].ravel())[0, 1])
        grad_corr = float(np.corrcoef(_grad_mag(ours_rgb)[both].ravel(),
                                      _grad_mag(off_rgb_a)[both].ravel())[0, 1])
    else:
        rgb_mae = corr_bad = grad_corr = float("nan")

    # ---- 深度比对：我们的**斜距** vs 官方深度值 ----
    dep: Dict[str, Any] = {}
    if both.sum() > 100:
        a = ours_rng[both].astype(np.float64)        # 我们：斜距
        b = off_d_a[both].astype(np.float64)         # 官方：实测也是斜距
        ratio = a / np.maximum(b, 1e-9)
        dep = {
            "n": int(both.sum()),
            "mae_m": float(np.abs(a - b).mean()),
            "median_ratio": float(np.median(ratio)),
            "p10_ratio": float(np.percentile(ratio, 10)),
            "p90_ratio": float(np.percentile(ratio, 90)),
        }
        # 诊断：比值是否随仰角漂移（若官方是 z 深度，会呈 1/cos(el)）
        rows = np.nonzero(both)[0]
        H = ours_rng.shape[0]
        el = np.pi / 2.0 - (rows + 0.5) / H * np.pi
        for lo, hi, name in ((-60, -30, "低仰角"), (-20, 20, "赤道"),
                             (30, 60, "高仰角")):
            m = (np.degrees(el) >= lo) & (np.degrees(el) < hi)
            if m.sum() > 50:
                dep[f"median_ratio_{name}"] = float(np.median(ratio[m]))
                dep[f"inv_cos_{name}"] = float(np.median(1.0 / np.cos(el[m])))

    return {
        "room": scene.room, "uuid": scene.uuid[:12],
        "align": al,
        "our_coverage": float((ours_rng > 0).mean()),
        "our_solid": scene.panorama.solid_angle_coverage,
        "off_coverage": float((off_d > 0).mean()),
        "rgb_mae": rgb_mae, "grad_corr": grad_corr,
        "grad_corr_wrong_conv": corr_bad,
        "depth": dep,
        "our_range_max": float(ours_rng[ours_rng > 0].max()) if (ours_rng > 0).any() else 0.0,
        "off_range_max": float(off_d[off_d > 0].max()) if (off_d > 0).any() else 0.0,
    }


def print_report(r: Dict[str, Any]) -> None:
    al = r["align"]
    print(f"\n--- {r['room']} ({r['uuid']}) ---")
    print(f"  对齐: 方位约定={'列镜像(官方方位递增方向与我们相反)' if al['az_flip'] else '同向'}"
          f"  滚动 {al['roll_deg']:+.2f}°  仰角偏移 {al['el_shift_deg']:+.2f}°"
          f"（{al['el_shift_rows']} 行）")
    print(f"        梯度相关性 best={al['corr_best']:.4f}   "
          f"（另一种方位约定只有 {al['corr_other_convention']:.4f}）")
    print(f"  覆盖: 我们(像素) {r['our_coverage']*100:.1f}% / "
          f"(立体角) {r['our_solid']*100:.1f}%   官方(像素) {r['off_coverage']*100:.1f}%")
    print(f"  RGB : MAE {r['rgb_mae']:.2f}/255   梯度相关 {r['grad_corr']:.4f}"
          f"（错误方位约定下 {r['grad_corr_wrong_conv']:.4f}）")
    d = r["depth"]
    if d:
        print(f"  深度: 共同有效 {d['n']:,} px   MAE {d['mae_m']:.4f} m   "
              f"比值中位 {d['median_ratio']:.4f} "
              f"(p10 {d['p10_ratio']:.3f} / p90 {d['p90_ratio']:.3f})")
        for k in ("低仰角", "赤道", "高仰角"):
            kk = f"median_ratio_{k}"
            if kk in d:
                print(f"        {k:<4}: 比值 {d[kk]:.3f}   "
                      f"(若官方是 z 深度，期望 1/cos≈{d[f'inv_cos_{k}']:.3f})")
    print(f"  最远距离: 我们 {r['our_range_max']:.2f} m（受 max_depth 限制）  "
          f"官方 {r['off_range_max']:.2f} m")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--room", default="office_6")
    ap.add_argument("--all", action="store_true", help="跑多个房间")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-depth", type=float, default=20.0,
                    help="融合时的斜距上限（默认放宽到 20 m，便于和官方比远场）")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2
    vw, vh = args.width // 2, args.height // 2      # 比对在小尺寸上做，够用且快

    if args.all:
        locs = sorted(list_locations(args.root), key=lambda l: -len(l.frame_ids))
        locs = [l for l in locs if len(l.frame_ids) >= 20]
        idx = np.linspace(0, len(locs) - 1, num=min(args.n, len(locs))).round().astype(int)
        targets = [locs[int(i)] for i in np.unique(idx)]
    else:
        targets = [select_location(args.root, room=args.room, min_frames=8)]

    results: List[Dict[str, Any]] = []
    for loc in targets:
        hr(f"构建全景并比对：{loc.room} ({loc.uuid[:12]})，{len(loc.frame_ids)} 个真实视角")
        sc = load_scene(args.root, loc, width=args.width, height=args.height,
                        max_frames=args.max_frames, max_depth=args.max_depth,
                        with_gt=False)
        r = compare_scene(sc, loc, width=vw, height=vh,
                          n_grid=min(1024, vw))
        print_report(r)
        results.append(r)

    hr("汇总与判据")
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    # 1) ★ 必须选对**方位约定**：正确约定的梯度相关性要远高于另一种。
    #    这一步是关键 —— 只允许平移时，官方全景的对不上（镜像错误无法靠平移修）。
    for r in results:
        al = r["align"]
        check(f"{r['room']} 方位约定判定明确",
              al["corr_best"] > al["corr_other_convention"] * 1.5,
              f"{al['corr_best']:.4f} vs 另一种 {al['corr_other_convention']:.4f}")
        check(f"{r['room']} 仰角无需翻转（两边都以世界 z 为极轴）",
              abs(al["el_shift_rows"]) <= 6,
              f"{al['el_shift_deg']:+.2f}° ({al['el_shift_rows']} 行)")

    # 2) 结构必须真的对上（梯度相关，抗曝光差异）
    check("对齐后结构一致（梯度相关 > 0.4）",
          all(r["grad_corr"] > 0.4 for r in results),
          "; ".join(f"{r['grad_corr']:.4f}" for r in results))

    # 3) 深度必须**逐像素**吻合 —— 这是"融合几何正确"最硬的证据
    for r in results:
        d = r["depth"]
        if not d:
            continue
        check(f"{r['room']} 深度 MAE < 0.1 m", d["mae_m"] < 0.1,
              f"{d['mae_m']:.4f} m")
        check(f"{r['room']} 深度比值中位 ≈ 1",
              abs(d["median_ratio"] - 1.0) < 0.02,
              f"{d['median_ratio']:.4f}")
        check(f"{r['room']} 深度比值分布紧（p90−p10 < 0.1）",
              (d["p90_ratio"] - d["p10_ratio"]) < 0.1,
              f"p10 {d['p10_ratio']:.3f} / p90 {d['p90_ratio']:.3f}")
        # 4) 深度语义：比值不随仰角漂移（z 深度会漂 1/cos）
        drift = abs(d.get("median_ratio_高仰角", 1.0) - d.get("median_ratio_赤道", 1.0))
        check(f"{r['room']} 深度语义是斜距（不随仰角漂移）",
              drift < 0.05,
              f"高仰角与赤道之差 {drift:.4f}；若是 z 深度，期望漂移 "
              f"{abs(d.get('inv_cos_高仰角', 1.0) - d.get('inv_cos_赤道', 1.0)):.3f}")

    # 5) RGB 也应吻合
    check("对齐后 RGB MAE < 20/255",
          all(r["rgb_mae"] < 20 for r in results if np.isfinite(r["rgb_mae"])),
          "; ".join(f"{r['rgb_mae']:.2f}" for r in results))

    print("\n" + "=" * 74)
    print("结论: " + ("与官方全景一致 ✓" if ok else "存在不通过项 ✗"))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
