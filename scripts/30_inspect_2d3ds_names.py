#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""30 · 列出 2D-3D-S 各目录的**真实文件名样本**（写适配器前必须先看）。

背景：核验脚本（29）发现实际情况与官方 README 的描述有出入：
  · `data/rgb` 有 10,327 个文件，而 `pano/pose` 只有 191 个 —— 每点约 54 张，不是 18
  · 「18 视角」是 `/raw` 的属性（3 俯仰 × 6 方位）
  · 没找到 `PointCloud.mat`（我的名字假设可能不对）
  · 命名约定也不是 README 里写的 `camera_{uuid}__{room}_...`

所以本脚本把**真实文件名**打出来，适配器按它写 —— 不按文档猜。
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--samples", type=int, default=4, help="每个目录打印几个样本")
    args = ap.parse_args()

    tar_path = Path(args.tar)
    print(f"扫描 {tar_path.name} …\n")
    names = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if m.isfile():
                names.append((m.name, m.size))
    print(f"共 {len(names)} 个文件\n")

    # ---- 按目录分组 ----
    by_dir = defaultdict(list)
    for n, s in names:
        d = str(Path(n).parent).replace("\\", "/")
        by_dir[d].append((Path(n).name, s))

    print("=" * 88)
    print("各目录的文件名样本")
    print("=" * 88)
    for d in sorted(by_dir):
        files = by_dir[d]
        print(f"\n[{d}]  {len(files)} 个文件")
        for fname, size in sorted(files)[: args.samples]:
            print(f"    {size / 1024:9.1f} KB  {fname}")
        if len(files) > args.samples:
            print(f"    … 另外 {len(files) - args.samples} 个")

    # ---- 3d/ 目录完整列出（GT 在这里）----
    for d in sorted(by_dir):
        if d.endswith("/3d") or d.endswith("/3d/rgb_textures"):
            if d.endswith("rgb_textures"):
                print(f"\n[{d}] 纹理，跳过逐个列出（{len(by_dir[d])} 个）")
                continue
            print(f"\n{'=' * 88}\n[3d] 全部 {len(by_dir[d])} 个文件（GT 来源）\n{'=' * 88}")
            for fname, size in sorted(by_dir[d]):
                print(f"    {size / (1 << 20):9.1f} MB  {fname}")

    # ---- 推断"每个采集点的视角数" ----
    print(f"\n{'=' * 88}\n结构推断\n{'=' * 88}")
    for d, pat, label in (
        ("area_1/raw", r"^([0-9a-f]{32})_i(\d)_(\d)\.jpg$", "raw RGB  {uuid}_i{pitch}_{yaw}.jpg"),
        ("area_1/raw", r"^([0-9a-f]{32})_d(\d)_(\d)\.(png|jpg)$", "raw 深度 {uuid}_d{pitch}_{yaw}"),
        ("area_1/raw", r"^([0-9a-f]{32})_pose_(\d)_(\d)\.txt$", "raw 位姿 {uuid}_pose_{pitch}_{yaw}.txt"),
        ("area_1/raw", r"^([0-9a-f]{32})_intrinsics_(\d)\.txt$", "raw 内参 {uuid}_intrinsics_{pitch}.txt"),
    ):
        rx = re.compile(pat)
        uuids, pitches, yaws = set(), set(), set()
        depths_ext = Counter()
        for fname, _ in by_dir.get(d, []):
            m = rx.match(fname)
            if m:
                uuids.add(m.group(1))
                pitches.add(m.group(2))
                yaws.add(m.group(3))
                if m.lastindex and m.lastindex >= 4 and m.group(4):
                    depths_ext[m.group(4)] += 1
        if uuids:
            per = len([1 for f, _ in by_dir[d] if rx.match(f)]) / max(len(uuids), 1)
            extra = f"  扩展名分布 {dict(depths_ext)}" if depths_ext else ""
            print(f"\n  {label}")
            print(f"    采集点 {len(uuids)} 个   俯仰 {sorted(pitches)}   方位 {sorted(yaws)}"
                  f"   → 每点 {per:.1f} 个{extra}")

    # ---- data/ 的每点文件数 ----
    for d in ("area_1/data/rgb", "area_1/data/depth", "area_1/data/pose"):
        files = by_dir.get(d, [])
        if not files:
            continue
        # 从文件名里提取"采集点标识"（取前两段下划线之前的部分）
        keys = Counter()
        for fname, _ in files:
            key = fname.split("_frame_")[0] if "_frame_" in fname else fname.split("_")[0]
            keys[key] += 1
        n_keys = len(keys)
        print(f"\n  [{d}]  {len(files)} 个文件 / {n_keys} 个采集点 "
              f"= 每点 {len(files) / max(n_keys, 1):.1f} 个")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
