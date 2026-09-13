#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""33 · 钻进 pointcloud.mat（MAT v7.3 / HDF5）拿到物体级 GT。

MATLAB v7.3 的存储方式有两个坑（都是实测出来的，不踩不知道）
========================================================
1. **struct array 是"按字段存成引用数组"**：
   `Area_1/Disjoint_Space` 是一个 Group，里面有 `name` / `object` / `color` /
   `AlignmentAngle` **四个 datasets，各是 (44,1) 的引用数组**。
   所以第 i 个空间的名字是 `h[ h[...]['name'][i,0] ]`，
   而不是 `h[...]['name'][i]` —— 后者拿到的是引用对象本身。
2. **字符串是 MATLAB char 数组（uint16 码点）**，不是 HDF5 字符串。
   直接 `ravel()[0]` 会得到**第一个字符的码点**（例如 'c' = 99），
   看起来像"名字是 99"这种莫名其妙的数字。必须按 char 数组解码。

用法::

    python scripts/33_probe_2d3ds_gt.py --mat G:\\2d3ds\\area_1\\area_1\\3d\\pointcloud.mat
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def decode(obj) -> str:
    """把 MATLAB char 数组 / 字符串 / 数字解码成 Python 字符串。"""
    a = np.asarray(obj)
    if a.dtype.kind in ("U", "S"):
        return str(a.ravel()[0])
    if a.dtype == np.uint16 or a.dtype.kind in ("i", "u"):
        try:
            return "".join(chr(int(c)) for c in a.ravel() if int(c) != 0)
        except Exception:
            return str(a.ravel()[0])
    return str(a)


def deref(h, node):
    """把单个引用解成实际对象（不是引用数组）。"""
    import h5py

    if isinstance(node, h5py.h5r.Reference) or np.asarray(node).dtype == h5py.ref_dtype:
        arr = np.asarray(node)
        if arr.dtype == h5py.ref_dtype:
            r = arr.ravel()[0]
            return h[r] if r else None
    return node


def field_names(node) -> list:
    import h5py

    return list(node.keys()) if isinstance(node, h5py.Group) else []


def get_field(h, struct, name: str):
    """按大小写不敏感取 struct 的字段。"""
    if not hasattr(struct, "keys"):
        return None
    for k in struct.keys():
        if k.lower() == name.lower():
            return deref(h, struct[k])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mat", required=True)
    ap.add_argument("--show-rooms", type=int, default=3)
    ap.add_argument("--max-objects-per-room", type=int, default=6)
    args = ap.parse_args()

    m = Path(args.mat)
    if not m.exists():
        print(f"[FAIL] 找不到 {m}")
        return 2
    import h5py

    with h5py.File(m, "r") as h:
        print("=" * 84)
        print(f"{m.name}  ({m.stat().st_size / (1 << 30):.2f} GB)  MAT v7.3 / HDF5")
        print("=" * 84)

        area_key = next(k for k in h.keys() if not k.startswith("#"))
        area = h[area_key]
        print(f"\n顶层：{list(h.keys())}")
        print(f"[{area_key}] 字段：{field_names(area)}")

        ds = None
        for k in area.keys():
            if "disjoint" in k.lower():
                ds = area[k]
                break
        if ds is None:
            print(f"⚠️ 没找到 Disjoint_Space")
            return 0

        print(f"[Disjoint_Space] 字段：{field_names(ds)}")
        for k in ds.keys():
            a = np.asarray(ds[k])
            print(f"    {k:<22} shape={a.shape} dtype="
                  f"{'引用数组' if a.dtype == h5py.ref_dtype else a.dtype}")

        room_names_ds = ds["name"]
        room_obj_ds = next(ds[k] for k in ds.keys() if k.lower() == "object")
        n_rooms = room_names_ds.shape[0]
        print(f"\n共 {n_rooms} 个空间")

        def objects_of_room(i: int):
            """取第 i 个空间的物体列表。

            ⚠️ 第二层嵌套的坑：`object[i,0]` 解引用出来是一个 **Group**
            （代表"结构体数组"），它的每个字段（`name`/`Bbox`/`points`…）
            又是 **(N,1) 的引用数组**。所以要按字段取第 j 个元素，
            不能把 Group 当成引用数组直接索引。
            """
            grp = deref(h, room_obj_ds[i, 0])
            if not hasattr(grp, "keys"):
                return []
            name_ds = next((grp[k] for k in grp.keys() if k.lower() == "name"), None)
            bbox_ds = next((grp[k] for k in grp.keys() if k.lower() == "bbox"), None)
            if name_ds is None:
                return []
            n = np.asarray(name_ds).shape[0]
            out = []
            for j in range(n):
                nm = decode(deref(h, np.asarray(name_ds)[j, 0]))
                bb = np.zeros(0)
                if bbox_ds is not None:
                    bb = np.asarray(deref(h, np.asarray(bbox_ds)[j, 0]),
                                    dtype=float).ravel()
                out.append((nm, bb))
            return out

        # ---------------- 逐空间统计 ----------------
        print(f"\n{'=' * 84}\n逐个空间：名字 + 物体（前 {args.show_rooms} 个详列）\n{'=' * 84}")
        all_names = []
        rooms_report = []
        for i in range(n_rooms):
            rname = decode(deref(h, room_names_ds[i, 0]))
            objs = objects_of_room(i)
            rooms_report.append((rname, len(objs)))
            all_names.extend(nm for nm, _ in objs)
            if i < args.show_rooms:
                print(f"\n  [{i}] 空间 {rname!r}  物体 {len(objs)} 个")
                for nm, bb in objs[:args.max_objects_per_room]:
                    s = ", ".join(f"{v:9.3f}" for v in bb) if bb.size else "?"
                    print(f"      {nm:<18} Bbox=[{s}]")

        # ---------------- 汇总 ----------------
        print(f"\n{'=' * 84}\n汇总\n{'=' * 84}")
        print(f"  空间数 {len(rooms_report)}   物体总数 {len(all_names)}")
        obj_per_room = [n for _, n in rooms_report]
        if obj_per_room:
            print(f"  每空间物体数：min {min(obj_per_room)}  "
                  f"中位 {int(np.median(obj_per_room))}  max {max(obj_per_room)}")
        cats = Counter(re.sub(r"[-_]?\d+$", "", n) for n in all_names if n)
        print(f"\n  类别数 {len(cats)}，前 25 个：")
        for c, n in cats.most_common(25):
            print(f"    {n:>5} 个  {c or '(空)'}")
        print("\n  → 适配器的 GT 来源：空间名 + 物体名 + Bbox"
              "[Xmin Ymin Zmin Xmax Ymax Zmax]（世界系，与 pose 同一坐标系）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
