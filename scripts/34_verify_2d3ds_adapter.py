# -*- coding: utf-8 -*-
"""在**真实** 2D-3D-S 数据上验收适配器（`roboground.data.stanford2d3d`）。

为什么需要这个脚本：`tests/test_stanford2d3d.py` 守的是"代码里的约定"，
但它**碰不到真数据**（13.8 GB 不进 CI）。而本项目踩过的最大一个坑恰恰是
约定本身搞错了 —— 我曾经用 `%.1f%%` 打印缺失率，把 0.88% 显示成 "0.0%"，
于是在报告里写了"深度完全稠密"。约定对了、代码对了，**真数据仍可能打脸**。

所以这个脚本的角色是：拿真实的 `area_1` 把适配器的输出和
`scripts/32_probe_2d3ds_fields.py` 独立测出来的数字**对一遍**。

用法：
    python scripts/34_verify_2d3ds_adapter.py --root G:\\2d3ds\\area_1\\area_1
    python scripts/34_verify_2d3ds_adapter.py --limit 40            # 少采样，快
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.stanford2d3d import (  # noqa: E402
    DEPTH_SCALE,
    INVALID_RAW,
    describe,
    list_locations,
    load_objects,
    objects_as_boxes,
)

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")


def hr(t: str) -> None:
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=40,
                    help="采样多少个采集点做逐像素统计")
    ap.add_argument("--objects", action="store_true",
                    help="顺带跑一遍 3d/pointcloud.mat 的 GT（要 1 GB 内存）")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}（先跑 scripts/31_extract_2d3ds.py）")
        return 2

    hr("0) 目录概览")
    info = describe(args.root)
    for k, v in info.items():
        print(f"  {k}: {v}")

    hr("1) 采集点（同一 uuid = 同一光心、不同朝向 = 多视角融合的前提）")
    locs = list_locations(args.root)
    nf = [len(l.frame_ids) for l in locs]
    print(f"  采集点数: {len(locs)}")
    if nf:
        print(f"  每点帧数: min={min(nf)} median={int(np.median(nf))} max={max(nf)}")
    sample = [l for l in locs if len(l.frame_ids) >= 4][: args.limit]
    print(f"  用于统计的采集点: {len(sample)} 个"
          f"（共 {sum(len(l.frame_ids) for l in sample)} 帧）")

    hr("2) 逐像素深度统计 —— 检验 raw==65535 是否真的是'无数据'")
    # 这里刻意**同时**读原始 PNG 和适配器输出，因为这是两个不同的问题：
    #   Q1 原始数据里有多少像素质标记为无数据？   → 读 raw
    #   Q2 适配器有没有把它们变成 127.998 m？      → 读 depth_m()
    # 第一版脚本把两者混在一个计数器里，得到"无效 = 0 像素"这种自相矛盾的输出。
    from PIL import Image

    tot = valid = 0
    n_raw65535 = n_raw0 = 0
    n_leak = 0
    n_zero_mismatch = 0
    below5 = 0
    dmax = 0.0
    vmin = np.inf
    n_sentinel_images = 0
    for loc in sample:
        for fid in loc.frame_ids:
            p = loc._path("depth", fid)          # noqa: SLF001（核验脚本，故意用私有）
            d = loc.depth_m(fid)
            if p is None or d is None:
                continue
            raw = np.asarray(Image.open(p))
            inv = (raw <= 0) | (raw >= INVALID_RAW)
            n_raw65535 += int((raw >= INVALID_RAW).sum())
            n_raw0 += int((raw <= 0).sum())
            if inv.any():
                n_sentinel_images += 1
            # Q2a：适配器输出的零集必须**恰好**等于原始无效集
            if int((d <= 0).sum()) != int(inv.sum()):
                n_zero_mismatch += 1
            # Q2b：不能有任何像素等于 65535/512
            n_leak += int((d == INVALID_RAW / DEPTH_SCALE).sum())

            tot += d.size
            v = d[d > 0]
            valid += int(v.size)
            if v.size:
                below5 += int((v < 5.0).sum())
                dmax = max(dmax, float(v.max()))
                vmin = min(vmin, float(v.min()))

    n_invalid = n_raw65535 + n_raw0
    print(f"  总像素: {tot:,}")
    print(f"  有效  : {valid:,}  ({valid / tot * 100:.4f}%)")
    print(f"  无效  : {n_invalid:,}  ({n_invalid / tot * 100:.4f}%)")
    print(f"     其中 raw=={INVALID_RAW}（哨兵）: {n_raw65535:,}")
    print(f"     其中 raw==0（空洞）        : {n_raw0:,}")
    print(f"  含无效像素的图: {n_sentinel_images} 张")
    print(f"  适配器输出中等于 {INVALID_RAW / DEPTH_SCALE:.4f} 的像素: {n_leak}")
    print(f"  零集与原始无效集不一致的图: {n_zero_mismatch} 张")
    print(f"  有效深度范围: {vmin:.4f} ~ {dmax:.4f} m")
    print(f"  有效像素中 < 5 m 的占比: {below5 / valid * 100:.4f}%")

    hr("3) 判据")
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    frac_invalid = n_invalid / tot * 100 if tot else 0.0
    frac_lt5 = below5 / valid * 100 if valid else 0.0
    # 独立测量（scripts/32）在两个不同样本上得到 0.8797%（1 个采集点/40 张）
    # 与 0.6275%（40 个采集点/2158 帧）—— 缺失率随房间而变，所以给区间不给点值。
    check("无效深度占比 ∈ (0.2%, 1.5%)", 0.2 < frac_invalid < 1.5,
          f"实测 {frac_invalid:.4f}%")
    # ★ 关键回归：适配器**不能**把 65535 变成 127.998 m。这是精确判据，
    #   不是"看起来不大"—— 第一版我写成 `max < 10 m`，结果被真实的
    #   51.6 m 长走廊误判为失败。
    check("无 65535/512 = 127.998 m 泄漏", n_leak == 0, f"{n_leak} 像素命中")
    check("适配器零集 == 原始无效集", n_zero_mismatch == 0,
          f"{n_zero_mismatch} 张图不一致")
    check("无效值确实存在（不是 0%）", n_invalid > 0,
          f"{n_invalid:,} 像素（若为 0 说明又回到 '%.1f%%' 那种误判）")
    # ★ 深度长尾：实测 94.9% 的像素 < 5 m，最远 51.6 m 且集中在
    #   office_28（同一房间多帧稳定复现 47.6~51.6 m）→ 是**真实长视距**，
    #   不是脏数据。所以这里断言的是"分布形状"，不是"上限很小"。
    check("绝大多数有效像素 < 5 m", frac_lt5 > 94.0,
          f"实测 {frac_lt5:.4f}%")
    check("最远有效深度 ∈ [10, 60] m",
          10.0 < dmax < 60.0,
          f"实测 {dmax:.4f} m（真实长走廊；非 128 m 伪值）")
    check("深度下限合理（>0.3 m）", vmin > 0.3, f"实测 {vmin:.4f} m")

    hr("4) 位姿：同一采集点内光心一致、朝向不同")
    loc = sample[0]
    cs, rots = [], []
    for fid in loc.frame_ids[:8]:
        p = loc.pose(fid)
        cs.append(-np.asarray(p.R).T @ np.asarray(p.t))
        rots.append(np.asarray(p.R))
    spread = float(np.max(np.linalg.norm(np.array(cs) - np.mean(cs, axis=0), axis=1)))
    rot_spread = max(float(np.linalg.norm(rots[0] - r)) for r in rots[1:])
    print(f"  采集点 {loc.uuid[:12]} / {loc.room}：{len(loc.frame_ids)} 帧")
    print(f"  光心离散度: {spread:.3e} m")
    print(f"  旋转差异  : {rot_spread:.4f}（应远大于 0）")
    print(f"  内参 fx 范围: "
          f"{min(loc.intrinsics(f).fx for f in loc.frame_ids[:8]):.1f} ~ "
          f"{max(loc.intrinsics(f).fx for f in loc.frame_ids[:8]):.1f}")
    check("同采集点光心一致", spread < 1e-4, f"{spread:.3e} m")
    check("同采集点朝向不同", rot_spread > 1e-3, f"{rot_spread:.4f}")

    if args.objects:
        hr("5) GT（3d/pointcloud.mat）")
        mp = args.root / "3d" / "pointcloud.mat"
        t0 = time.time()
        objs = load_objects(mp)
        boxes, labels = objects_as_boxes(objs)
        cls = {}
        for c in labels:
            cls[c] = cls.get(c, 0) + 1
        print(f"  物体 {len(objs)}，成框 {boxes.shape[0]}，"
              f"类别 {len(cls)}（{time.time() - t0:.1f}s）")
        print(f"  yaw 是否全 0: {bool(np.allclose(boxes[:, 6], 0.0))}")
        print(f"  被丢弃的物体: {len(objs) - boxes.shape[0]} 个"
              "（字段缺失/含 NaN/尺寸为零）")
        for k, v in sorted(cls.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<14} {v:>5}")
        check("GT 是轴对齐盒（yaw 全 0）",
              bool(np.allclose(boxes[:, 6], 0.0)),
              "→ 评估不能用有向 3D IoU")

    print("\n" + ("=" * 72))
    print("结论: " + ("适配器在真数据上通过 ✓" if ok else "存在不通过项 ✗"))
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
