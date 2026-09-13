#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""32 · 探明 2D-3D-S 的三个关键字段（适配器的全部依赖都在这）。

为什么必须实测而不是照文档写
==========================
官方 README 说：`camera_rt_matrix` 是 "The 4x3 camera RT matrix"，
但**没说方向**。而这个方向搞错会导致**静默的错误位姿** ——
本项目在 SUN RGB-D 上就踩过同类坑（坐标系约定错 → 地图重影，且不报错）。

所以这里用一个**可判定的验算**来确定方向：

    json 里同时给了 `camera_location`（相机光心的世界坐标）和 `camera_rt_matrix`。
    设 M[:3,:3] = R，M[3] = t（4×3 的最后一"行"），则只有两种可能：
      · 若 p_cam = R·p_world + t   （world→camera）→ 光心 C = −Rᵀ·t
      · 若 p_world = R·p_cam + t   （camera→world）→ 光心 C = t
    把两种都算出来跟 `camera_location` 比，**谁对上就是哪种**。

用法::

    python scripts/32_probe_2d3ds_fields.py --dir G:\\2d3ds\\area_1\\area_1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.logging import get_logger  # noqa: E402

log = get_logger("probe.2d3ds")


# ==========================================================================
def probe_pose(root: Path) -> None:
    """★ 判定 camera_rt_matrix 的方向（决定性的一步）。"""
    pdir = root / "data" / "pose"
    jsons = sorted(pdir.glob("*.json"))
    if not jsons:
        print(f"\n[1] 位姿：{pdir} 下没有 json")
        return
    print(f"\n[1] 位姿（{len(jsons)} 个 json，抽 2 个看）")
    for jp in jsons[:2]:
        d = json.loads(jp.read_text(encoding="utf-8"))
        print(f"\n  文件 {jp.name}")
        for k, v in d.items():
            s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            print(f"    {k:34s} {s[:120]}")

        loc = d.get("camera_location")
        rt = d.get("camera_rt_matrix")
        if loc is None or rt is None:
            continue
        loc = np.asarray(loc, dtype=np.float64).reshape(3)
        M = np.asarray(rt, dtype=np.float64)
        print(f"    camera_rt_matrix 形状 {M.shape}")
        # ⚠️ 官方 README 说这是 "4x3"，但**实测是 (3, 4)** ——
        #    所以两种布局都要判：(3,4) → R=M[:, :3], t=M[:, 3]；
        #    (4,3) → R=M[:3, :], t=M[3, :]。
        if M.shape == (3, 4):
            R, t = M[:, :3], M[:, 3]
            print("    布局判定：形状 (3,4) → 前 3 列是旋转、第 4 列是平移"
                  "（**与官方 README 的 4x3 说法不符**）")
        elif M.shape == (4, 3):
            R, t = M[:3, :], M[3, :]
            print("    布局判定：形状 (4,3) → 前 3 行是旋转、第 4 行是平移")
        else:
            print(f"    ⚠️ 未知形状 {M.shape}，跳过方向判定")
            continue
        det = float(np.linalg.det(R))
        C_wc = -R.T @ t          # 假设 p_cam = R p_world + t（world→camera）
        C_cw = t                 # 假设 p_world = R p_cam + t（camera→world）
        e_wc = float(np.linalg.norm(C_wc - loc))
        e_cw = float(np.linalg.norm(C_cw - loc))
        print(f"    det(R) = {det:+.6f}（±1 说明是合法旋转）")
        print(f"    若 world→camera：C = −Rᵀ·t = {np.round(C_wc, 4).tolist()}  "
              f"误差 {e_wc:.6f} m")
        print(f"    若 camera→world：C = t     = {np.round(C_cw, 4).tolist()}  "
              f"误差 {e_cw:.6f} m")
        if min(e_wc, e_cw) < 1e-3:
            verdict = ("world→camera  p_cam = R·p_world + t" if e_wc < e_cw
                       else "camera→world  p_world = R·p_cam + t")
            print(f"    ★★ 判定成立（误差 {min(e_wc, e_cw):.2e} m）：**{verdict}**")
        else:
            print("    ⚠️ 两种假设都对不上，需要人工看（可能还差一步轴置换）")


def probe_depth(root: Path) -> None:
    """核验深度图的位深、缺失值与换算。"""
    ddir = root / "data" / "depth"
    pngs = sorted(ddir.glob("*.png"))
    if not pngs:
        print(f"\n[2] 深度：{ddir} 下没有 png")
        return
    try:
        from PIL import Image
    except ImportError:
        print("\n[2] 深度：没有 PIL，跳过")
        return
    print(f"\n[2] 深度（{len(pngs)} 个，抽 3 个看）")
    for p in pngs[:3]:
        im = Image.open(p)
        a = np.asarray(im)
        print(f"\n  {p.name}")
        print(f"    尺寸 {a.shape}  dtype {a.dtype}  mode {im.mode}")
        if a.dtype == np.uint16:
            for miss_val, name in ((65535, "2^16-1"), (0, "0")):
                miss = int((a == miss_val).sum())
                print(f"    取值 {name:>7} 的占比 {miss / a.size * 100:5.1f}%")
            valid = a[(a != 65535) & (a != 0)]
            if valid.size:
                for scale, label in ((512.0, "1/512 m（官方说明）"), (1000.0, "1/1000 m（毫米）")):
                    m = valid.astype(np.float64) / scale
                    print(f"    按 {label:22s}：min {m.min():7.2f}  "
                          f"max {m.max():8.2f}  中位 {np.median(m):6.2f} m")
        else:
            print(f"    ⚠️ 不是 uint16")


def probe_pointcloud(root: Path) -> None:
    """探明 GT：pointcloud.mat 的变量结构与物体框字段。"""
    m = root / "3d" / "pointcloud.mat"
    if not m.exists():
        cands = list((root / "3d").glob("*.mat")) if (root / "3d").is_dir() else []
        print(f"\n[3] GT：没找到 pointcloud.mat；3d/ 下有 {[c.name for c in cands]}")
        return
    size_gb = m.stat().st_size / (1 << 30)
    print(f"\n[3] GT：{m.name}  {size_gb:.2f} GB")

    # 先看文件头，决定用 scipy(v5) 还是 h5py(v7.3)
    with m.open("rb") as fh:
        head = fh.read(2048)
    # ⚠️ HDF5 允许在文件最前面带 512/1024/2048 字节的 user block，
    #    所以 `head.startswith(HDF5 签名)` 会**漏判**。
    #    实测：pointcloud.mat 就是这种情况 —— 前 128 字节不像 HDF5，
    #    但 scipy 会报 "Please use HDF reader for matlab v7.3"。
    HDF5_SIG = b"\x89HDF\r\n\x1a\n"
    is_hdf5 = any(head[off:off + 8] == HDF5_SIG for off in (0, 512, 1024, 2048))
    print(f"    文件头：{'HDF5（MAT v7.3）→ 用 h5py 惰性读取' if is_hdf5 else 'MAT v5/v7 → 用 scipy'}")

    if is_hdf5:
        try:
            import h5py
        except ImportError:
            print("    [WARN] 没有 h5py。装它（`pip install h5py`）或用 scipy（内存吃紧）")
            return
        with h5py.File(m, "r") as h:
            print(f"    顶层键：{list(h.keys())}")
            for k in list(h.keys())[:3]:
                print(f"      {k}: {h[k]}")
        return

    try:
        import scipy.io as sio
    except ImportError:
        print("    [WARN] 没有 scipy，跳过")
        return
    if size_gb > 1.5:
        print(f"    ⚠️ 超过 1.5 GB，整个载入有 OOM 风险；改用 variable_names 只取需要的")
    md = sio.loadmat(str(m), struct_as_record=False, squeeze_me=True)
    keys = [k for k in md if not k.startswith("__")]
    print(f"    顶层变量：{keys}")
    for k in keys:
        v = md[k]
        print(f"      {k}: type={type(v).__name__} shape={getattr(v, 'shape', None)}")
        for attr in ("Disjoint_Space", "name", "AlignmentAngle", "object"):
            if hasattr(v, attr):
                print(f"        .{attr} = {type(getattr(v, attr)).__name__}")
    # 钻进 object → Bbox
    for k in keys:
        v = md[k]
        if not hasattr(v, "Disjoint_Space"):
            continue
        spaces = np.atleast_1d(v.Disjoint_Space)
        print(f"\n    {k}.Disjoint_Space：{spaces.size} 个空间")
        sp = spaces[0]
        objs = np.atleast_1d(getattr(sp, "object", []))
        print(f"      空间 {getattr(sp, 'name', '?')}  物体 {objs.size} 个")
        for o in objs[:5]:
            bb = np.asarray(getattr(o, "Bbox", []), dtype=float)
            pts = getattr(o, "points", None)
            n_pts = np.asarray(pts).shape[0] if pts is not None else 0
            print(f"        - {getattr(o, 'name', '?'):<18} "
                  f"Bbox={np.round(bb, 3).tolist() if bb.size else '?'}  "
                  f"点数 {n_pts}")
        # 统计所有空间的物体总数与类别名
        names = []
        for sp2 in spaces:
            for o in np.atleast_1d(getattr(sp2, "object", [])):
                names.append(str(getattr(o, "name", "")))
        import re as _re
        cats = {}
        for n in names:
            c = _re.sub(r"[-_]?\d+$", "", n)        # chair-1 → chair
            cats[c] = cats.get(c, 0) + 1
        print(f"\n    全部空间物体数 {len(names)}；类别（去重后前 15）：")
        for c, n in sorted(cats.items(), key=lambda kv: -kv[1])[:15]:
            print(f"        {n:>4} 个  {c}")
        break


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="解压后的 area 目录（含 data/ pano/ 3d/）")
    args = ap.parse_args()
    root = Path(args.dir)
    if not root.is_dir():
        print(f"[FAIL] 找不到目录 {root}")
        return 2
    print("=" * 84)
    print(f"探明字段：{root}")
    print("=" * 84)
    probe_pose(root)
    probe_depth(root)
    probe_pointcloud(root)
    print("\n" + "=" * 84)
    print("结论用于**写适配器**：位姿方向、深度换算、GT 字段三者都必须按实测来写。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
