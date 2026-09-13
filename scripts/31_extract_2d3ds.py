#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""31 · 从 2D-3D-S 的 tar 里**只抽取需要的部分**（带体积核算）。

为什么要选择性抽取
================
`area_1_no_xyz.tar` 解压是 30.44 GB，但其中一大半我们不需要：

| 目录 | 大小 | 我们要吗 | 原因 |
|---|---|---|---|
| `pano/global_xyz` | **11.4 GB** | ❌ | 逐像素世界坐标（.exr），本项目用不上 |
| `data/normal` | 781 MB | ❌ | 表面法线，本项目用不上 |
| `data/rgb` | 9.3 GB | ✅ | 主输入（真实采集位姿上渲染的高清 RGB） |
| `data/depth` | 2.0 GB | ✅ | 主输入（16-bit 深度） |
| `data/pose` | 8.9 MB | ✅ | **位姿，最关键** |
| `data/semantic` | 309 MB | ✅ | 逐像素实例标签（可做 GT/验证） |
| `pano/rgb` + `pano/depth` + `pano/pose` | 1.28 GB | ✅ | **官方全景 = 我们融合结果的参考答案** |
| `3d/pointcloud.mat` | 1.07 GB | ✅ | **物体级 GT（AABB + 实例标签）** |
| `raw/` | 4.2 GB | ⚠️ 可选 | 真实传感器原图（做 `/raw` vs `/data` 对照用） |
| `3d/*.obj` + 纹理 | 1.2 GB | ❌ | 网格与纹理，建图用不上 |

默认抽 **~14 GB**（不含 raw）；加 `--with-raw` 是 ~18.2 GB。

用法::

    python scripts/31_extract_2d3ds.py --tar G:\\2d3ds\\area_1_no_xyz.tar \\
        --out G:\\2d3ds\\area_1 --dry-run      # 先看要抽多少
    python scripts/31_extract_2d3ds.py --tar ... --out ...   # 真抽
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: 需要的路径片段（任一命中即保留）
INCLUDE = (
    "/data/pose/",
    "/data/rgb/",
    "/data/depth/",
    "/data/semantic/",          # 不含 semantic_pretty 的路径也会命中，下面再排除
    "/pano/rgb/",
    "/pano/depth/",
    "/pano/pose/",
    "/pano/semantic/",
    "pointcloud.mat",
    "camera_to_room.json",
    "LICENSE.pdf",
)

#: 明确排除（即使命中了 INCLUDE）
EXCLUDE = (
    "/data/semantic_pretty/",   # 可视化版，学习要用 semantic
    "/pano/global_xyz/",        # 11.4 GB，用不上
    "/pano/normal/",
    "/pano/semantic_pretty/",
)

#: 只有加了 --with-raw 才要
RAW = ("/raw/",)


def wanted(name: str, *, with_raw: bool) -> bool:
    if any(x in name for x in EXCLUDE):
        return False
    if any(x in name for x in RAW):
        return with_raw
    return any(k in name for k in INCLUDE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-raw", action="store_true",
                    help="连 raw/ 一起抽（+4.2 GB，做 /raw vs /data 对照时用）")
    ap.add_argument("--dry-run", action="store_true", help="只核算体积，不写盘")
    args = ap.parse_args()

    tar_path = Path(args.tar)
    if not tar_path.exists():
        print(f"[FAIL] 找不到 {tar_path}")
        return 2
    out = Path(args.out)

    print("=" * 84)
    print(f"{'核算' if args.dry_run else '抽取'} {tar_path.name} → {out}")
    print("=" * 84)

    keep, drop = [], []
    drop_by_dir = defaultdict(lambda: [0, 0])
    t0 = time.time()
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            if wanted(m.name, with_raw=args.with_raw):
                keep.append(m)
            else:
                drop.append(m)
                parts = m.name.split("/")
                key = "/".join(parts[:3]) if len(parts) >= 3 else m.name
                drop_by_dir[key][0] += 1
                drop_by_dir[key][1] += m.size
    scan_s = time.time() - t0

    keep_bytes = sum(m.size for m in keep)
    drop_bytes = sum(m.size for m in drop)
    print(f"\n扫描耗时 {scan_s:.1f}s")
    print(f"  保留 {len(keep):>6} 个文件  {keep_bytes / (1 << 30):7.2f} GB")
    print(f"  跳过 {len(drop):>6} 个文件  {drop_bytes / (1 << 30):7.2f} GB")
    print(f"  合计 {len(keep) + len(drop)} 个 / "
          f"{(keep_bytes + drop_bytes) / (1 << 30):.2f} GB")

    print(f"\n  跳过的内容（按体积，确认没跳错东西）：")
    for k, (n, s) in sorted(drop_by_dir.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"    {s / (1 << 20):9.1f} MB  {n:>6} 个   {k}")

    if args.dry_run:
        print(f"\n  [dry-run] 没有写盘。要真抽就去掉 --dry-run。")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    print(f"\n开始抽取到 {out} …")
    t0 = time.time()
    done = 0
    written = 0
    try:
        with tarfile.open(tar_path, "r") as tf:
            for m in keep:
                f = tf.extractfile(m)
                if f is None:
                    continue
                dst = out / m.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with dst.open("wb") as fh:
                    while True:
                        b = f.read(4 << 20)
                        if not b:
                            break
                        fh.write(b)
                written += m.size
                done += 1
                if done % 2000 == 0:
                    el = time.time() - t0
                    print(f"\r    {done}/{len(keep)} 个  "
                          f"{written / (1 << 30):.2f} GB  "
                          f"{written / max(el, 1e-9) / (1 << 20):.0f} MB/s",
                          end="", flush=True)
    except KeyboardInterrupt:
        print("\n  [中断] 已抽出的文件保留，重跑会覆盖——不影响正确性")
        return 130
    el = time.time() - t0
    print(f"\r    完成：{done} 个文件 / {written / (1 << 30):.2f} GB / "
          f"耗时 {el:.1f}s / 平均 {written / max(el, 1e-9) / (1 << 20):.0f} MB/s")
    print(f"\n  下一步：python scripts/29_verify_2d3ds.py --tar {tar_path}  "
          f"（核验位姿/深度/GT 的真实字段）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
