#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""29 · 核验 Stanford 2D-3D-S 的真实结构（**在写适配器之前**必须先跑）。

为什么必须先核验
==============
这个项目已经有过一次教训：**照文档的假设写代码，真数据一到就崩**。
所以拿到数据的第一件事不是写适配器，而是**把实际结构打印出来**：

  1. tar 里到底有哪些目录、各多大（决定"要不要全解压"）
  2. 一个采集点是不是真有 **18 个视角**（6 前 + 6 上 + 6 下）
  3. 每个视角是不是都**真的有深度图**（而不是只有 RGB）
  4. 位姿 json 的字段名与含义（`camera_rt_matrix` 是 world→camera 还是反过来）
  5. 深度图的量化规则（16-bit、缺失值、是 z-depth 还是光心距离）
  6. `Area_#_PointCloud.mat` 里物体框与实例标签的实际字段结构
  7. `/pano` 官方全景有哪些模态（这是我们融合结果的**参考答案**）

本脚本**只读不写**，不修改任何数据。

用法::

    python scripts/29_verify_2d3ds.py --tar G:\\2d3ds\\area_1_no_xyz.tar
    python scripts/29_verify_2d3ds.py --tar ... --extract-to G:\\2d3ds\\area_1 --only-needed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.logging import get_logger  # noqa: E402

log = get_logger("verify.2d3ds")

#: 官方说明里"每个采集点 18 个视角 = 6 个朝前 + 6 个朝上 + 6 个朝下"
VIEWS_PER_LOCATION = 18


# ==========================================================================
def scan_tar(tar_path: Path) -> list:
    """扫一遍 tar 的成员清单（不解压），返回 [(name, size), ...]。"""
    members = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if m.isfile():
                members.append((m.name, m.size))
    return members


def report_tree(members, top_n: int = 12) -> None:
    """按目录汇总：文件数与总大小（决定哪些目录值得解压）。"""
    agg = defaultdict(lambda: [0, 0])
    for name, size in members:
        parts = name.split("/")
        # 取到第二层（area_1/data/rgb/... → area_1/data）
        key = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
        agg[key][0] += 1
        agg[key][1] += size
    print(f"\n  共 {len(members)} 个文件")
    print(f"  {'目录':<40}{'文件数':>8}{'大小':>12}")
    for k, (n, s) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top_n]:
        print(f"  {k:<40}{n:>8}{s / (1 << 20):>10.1f} MB")


def report_prefixes(members, depth: int = 3, top_n: int = 30) -> None:
    """列出实际出现的路径前缀，看清命名约定（不靠文档猜）。"""
    agg = defaultdict(lambda: [0, 0])
    for name, size in members:
        parts = name.split("/")
        key = "/".join(parts[:depth])
        agg[key][0] += 1
        agg[key][1] += size
    print(f"\n  路径前缀（深度 {depth}，按文件数排序，最多 {top_n} 条）：")
    for k, (n, s) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top_n]:
        print(f"    {n:>6} 个  {s / (1 << 20):>9.1f} MB   {k}")


def pick_one_location(members) -> dict:
    """从清单里挑出一个采集点，看它有哪些模态。

    ⚠️ **实际命名与官方 README 不一致**（核验出来的）：
      README 写的是 `camera_{uuid}__{room}_{i}_frame_{j}_domain__xxx`（**双**下划线），
      实际文件是   `camera_{uuid}_{room}_{i}_frame_{j}_domain_{modality}.png`（**单**下划线）。
      所以这里用按 uuid 特征（32 位十六进制）匹配的方式，
      而不是照 README 的模板硬套 —— 模板套错就一个文件都匹配不到。
    """
    uuid_re = re.compile(r"camera_([0-9a-f]{32})_")
    raw_uuid_re = re.compile(r"^([0-9a-f]{32})_")
    uuids = defaultdict(set)
    for name, _ in members:
        base = name.rsplit("/", 1)[-1]
        m = uuid_re.match(base) or raw_uuid_re.match(base)
        if not m:
            continue
        parts = name.split("/")
        if len(parts) >= 4:
            uuids[m.group(1)].add("/".join(parts[1:3]))
    if not uuids:
        return {}
    uuid = max(uuids, key=lambda u: len(uuids[u]))
    return {"uuid": uuid, "modalities": sorted(uuids[uuid])}


def report_one_location(members, uuid: str) -> None:
    """打印某个采集点在每种模态下的帧文件（验证是不是 18 个视角）。"""
    by_mod = defaultdict(list)
    for name, size in members:
        base = name.rsplit("/", 1)[-1]
        if uuid in base:
            parts = name.split("/")
            by_mod["/".join(parts[1:3])].append((base, size))
    print(f"\n  采集点 {uuid} 的模态与帧数：")
    for mod, items in sorted(by_mod.items()):
        print(f"    {mod:<28} {len(items):>4} 个文件")
    # `/raw` 才是"18 视角"的出处（3 俯仰 × 6 方位）
    for mod, items in sorted(by_mod.items()):
        if mod.endswith("/raw"):
            rx = {
                "raw RGB": re.compile(r"^[0-9a-f]{32}_i(\d)_(\d)\.(jpg|png)$"),
                "raw 深度": re.compile(r"^[0-9a-f]{32}_d(\d)_(\d)\.(jpg|png)$"),
                "raw 位姿": re.compile(r"^[0-9a-f]{32}_pose_(\d)_(\d)\.txt$"),
                "raw 内参": re.compile(r"^[0-9a-f]{32}_intrinsics_(\d)\.txt$"),
            }
            for label, r in rx.items():
                hits = [(b, s) for b, s in items if r.match(b)]
                pitches = sorted({r.match(b).group(1) for b, _ in hits})
                yaws = sorted({r.match(b).group(2) for b, _ in hits}) \
                    if r.groups >= 2 else []
                if hits:
                    print(f"    → {label:<10} {len(hits):>3} 个   俯仰 {pitches}   "
                          f"方位 {yaws}   合计 {len(pitches) * max(len(yaws), 1)} 视角")
            for b, s in sorted(items)[:3]:
                print(f"        {b}  ({s / 1024:.0f} KB)")
    # data/ 是"常规图像"，每点比 raw 多
    if any(m.endswith("/data/rgb") for m in by_mod):
        n = len(by_mod["area_1/data/rgb"]) if "area_1/data/rgb" in by_mod else 0
        print("    （注：`/data/rgb` 这类「常规」模态每点的帧数**多于** 18，"
              "这是官方在更多方向上渲染的结果）")


# ==========================================================================
def inspect_pose(tar_path: Path, members, uuid: str) -> None:
    """读一个位姿 json，打印字段与含义（关键：相机 pose 的方向约定）。"""
    target = None
    for name, _ in members:
        if uuid in name and name.endswith(".json") and "/pose/" in name:
            target = name
            break
    if not target:
        print("\n  [WARN] 没找到该采集点的 pose json；跳过位姿核验")
        return
    with tarfile.open(tar_path, "r") as tf:
        f = tf.extractfile(target)
        d = json.loads(f.read().decode("utf-8"))
    print(f"\n  位姿 json：{target}")
    for k, v in d.items():
        s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        print(f"    {k:32s} {s[:110]}")
    rt = d.get("camera_rt_matrix")
    if rt is not None:
        M = np.asarray(rt, dtype=float)
        print(f"    → camera_rt_matrix 形状 {M.shape}")
        if M.shape == (4, 3):
            R = M[:3, :]
            print(f"      R 的行列式 {np.linalg.det(R):+.4f}（±1 说明是合法旋转）")
            print(f"      第 4 行 {np.round(M[3], 4).tolist()}  ← 通常是平移")
            print("      ⚠️ 方向（world→camera 还是 camera→world）要看第 4 行与 "
                  "camera_location 的关系，代码里会实测确认")


def inspect_depth(tar_path: Path, members, uuid: str) -> None:
    """读一张深度图，核验：位深、缺失值、深度范围（决定 depth_scale 与无效值）。"""
    target = None
    for name, _ in members:
        if uuid in name and "/depth/" in name and name.endswith(".png"):
            target = name
            break
    if not target:
        print("\n  [WARN] 没找到深度 png；跳过深度核验")
        return
    try:
        from PIL import Image
    except ImportError:
        print("\n  [WARN] 没有 PIL，跳过深度核验")
        return
    import io

    with tarfile.open(tar_path, "r") as tf:
        raw = tf.extractfile(target).read()
    im = Image.open(io.BytesIO(raw))
    arr = np.asarray(im)
    print(f"\n  深度图：{target}")
    print(f"    尺寸 {arr.shape}   dtype {arr.dtype}   mode {im.mode}")
    if arr.dtype == np.uint16:
        miss = int((arr == 65535).sum())
        valid = arr[arr != 65535]
        print(f"    缺失值(65535) 占比 {miss / arr.size * 100:.1f}%")
        if valid.size:
            m = valid.astype(np.float64) / 512.0          # 官方：灵敏度 1/512 m
            print(f"    按 1/512 m 换算：min {m.min():.2f} m  max {m.max():.2f} m  "
                  f"中位 {np.median(m):.2f} m")
        print("    → 官方说明：16-bit、灵敏度 1/512 m、缺失 2^16-1、"
              "**z-depth（沿光轴）**")
    else:
        print(f"    ⚠️ 不是 uint16（实际 {arr.dtype}）—— 换算规则要重新确认")


def inspect_pointcloud(tar_path: Path, members, max_load_gb: float = 1.5) -> None:
    """读 3D 点云 mat，核验物体框与实例标签的实际结构（这是我们的 GT 来源）。

    ⚠️ 内存保护：`Area_#_PointCloud.mat` 可能有好几个 GB，而 scipy 载入时
    还会再放大 2~3 倍。所以超过 `max_load_gb` 就**不整个读进来**，
    只报告大小并给出后续处理建议 —— 宁可晚一步，也不要 OOM。
    """
    target = None
    for name, size in members:
        if name.lower().endswith("pointcloud.mat"):
            target = (name, size)
            break
    if not target:
        print("\n  [WARN] 没找到 PointCloud.mat；跳过 GT 核验")
        return
    name, size = target
    print(f"\n  3D 点云：{name}  ({size / (1 << 30):.2f} GB)")
    if size / (1 << 30) > max_load_gb:
        print(f"    ⚠️ 超过内存保护阈值 {max_load_gb} GB，**不整个载入**。")
        print("       后续处理建议：用 `matfile` 或 h5py 做**变量级**读取，")
        print("       或用 `scipy.io.loadmat(..., variable_names=[...])` 只取需要的变量。")
        return
    try:
        import scipy.io as sio
    except ImportError:
        print("    [WARN] 没有 scipy，跳过")
        return
    import tempfile

    # scipy 只能从文件读，先落一个临时文件（30 GB 的 tar 里只取这一个成员）
    with tarfile.open(tar_path, "r") as tf:
        data = tf.extractfile(name).read()
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as fh:
        fh.write(data)
        tmp = fh.name
    try:
        md = sio.loadmat(tmp, struct_as_record=False, squeeze_me=True)
        keys = [k for k in md if not k.startswith("__")]
        print(f"    顶层变量：{keys}")
        for k in keys[:3]:
            v = md[k]
            print(f"    {k}: type={type(v).__name__} "
                  f"shape={getattr(v, 'shape', None)}")
        # 尝试钻进 Disjoint_Space → object → Bbox
        for k in keys:
            v = md[k]
            for attr in ("Disjoint_Space", "disjoint_space"):
                if hasattr(v, attr):
                    spaces = getattr(v, attr)
                    spaces = np.atleast_1d(spaces)
                    print(f"    {k}.{attr}: {spaces.size} 个空间")
                    sp = spaces[0]
                    print(f"      空间名 {getattr(sp, 'name', '?')}")
                    objs = np.atleast_1d(getattr(sp, "object", []))
                    print(f"      物体数 {objs.size}")
                    for o in objs[:3]:
                        print(f"        - {getattr(o, 'name', '?')}  "
                              f"Bbox={np.round(np.asarray(getattr(o, 'Bbox', [])), 3).tolist()}")
                    break
    finally:
        Path(tmp).unlink(missing_ok=True)


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True, help="area_X_no_xyz.tar 路径")
    ap.add_argument("--extract-to", default=None, help="解压到该目录")
    ap.add_argument("--only-needed", action="store_true",
                    help="只解压需要的部分（data/pose|rgb|depth|semantic、pano、3d/*.mat），"
                         "跳过 mesh 纹理等大头")
    args = ap.parse_args()

    tar_path = Path(args.tar)
    if not tar_path.exists():
        print(f"[FAIL] 找不到 {tar_path}")
        return 2
    print("=" * 84)
    print(f"核验 {tar_path.name}  ({tar_path.stat().st_size / (1 << 30):.2f} GB)")
    print("=" * 84)

    print("\n[1] 扫成员清单（不解压）…")
    members = scan_tar(tar_path)
    report_tree(members)

    print("\n[2] 实际路径前缀（看清命名约定，不靠文档猜）")
    report_prefixes(members)

    print("\n[3] 挑一个采集点看模态")
    info = pick_one_location(members)
    if info:
        print(f"  选中 uuid = {info['uuid']}")
        print(f"  它出现的模态目录：{info['modalities']}")
        report_one_location(members, info["uuid"])
        inspect_pose(tar_path, members, info["uuid"])
        inspect_depth(tar_path, members, info["uuid"])
    else:
        print("  [WARN] 没解析出 uuid 形式的文件名，命名约定与预期不同")

    print("\n[4] GT：3D 点云里的物体框与实例标签")
    inspect_pointcloud(tar_path, members)

    # ---------------- 可选：解压 ----------------
    if args.extract_to:
        out = Path(args.extract_to)
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n[5] 解压到 {out} …")
        need = ("/data/pose/", "/data/rgb/", "/data/depth/", "/data/semantic/",
                "/pano/", "PointCloud.mat")
        kept = skipped = 0
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                if args.only_needed and not any(k in m.name for k in need):
                    skipped += 1
                    continue
                tf.extract(m, path=out)
                kept += 1
                if kept % 2000 == 0:
                    print(f"\r    已解压 {kept} 个文件（跳过 {skipped}）", end="",
                          flush=True)
        print(f"\r    完成：解压 {kept} 个，跳过 {skipped} 个")

    print("\n" + "=" * 84)
    print("核验完成。**下一步才是写适配器** —— 而且适配器要按上面打印出的")
    print("真实字段名/量化规则来写，不能按文档假设来写。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
