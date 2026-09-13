#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""28 · 获取 Stanford 2D-3D-S（经 Redivis 分发，需要 API token）。

为什么单独写一个脚本
==================
这个数据集不是"点一下就能下"的：它有 **110 GB / 766 GB 两个版本、7 个 area**，
通过 Redivis 的**许可门槛**分发。所以获取过程本身要满足三条：

  1. **先看清再下** —— 列目录、看大小，别盲下 766 GB 的 `withXYZ` 版本；
  2. **可断点续传** —— 10+ GB 的包断一次不能从头再来；
  3. **可校验** —— 官方 checksum 页给了每个包的 MD5，下完必须核对。

这个脚本把这三件事固定下来，顺便成为"数据是怎么来的"的可复现记录。

⚠️ 环境隔离
==========
官方客户端会拖进 dask / geopandas / polars / folium 等重依赖。
**不要装进项目的 conda 环境 `lxr`**（那是验证过 torch 的干净环境）。
本项目用一个独立 venv：

    G:\\minigore\\envs\\lxr\\python.exe -m venv G:\\2d3ds\\_venv
    G:\\2d3ds\\_venv\\Scripts\\python.exe -m pip install --no-deps redivis
    G:\\2d3ds\\_venv\\Scripts\\python.exe -m pip install requests tqdm niquests

实测：**文件下载路径只需要 `requests` + `tqdm` + `niquests`**，
不需要 dask / geopandas / polars / pyarrow（那些是表格与地理功能用的）。

用法
====
    set REDIVIS_TOKEN=xxxxx
    python scripts/28_fetch_2d3ds.py --probe          # 探元数据（确认 token 与引用名）
    python scripts/28_fetch_2d3ds.py --list           # 列目录与大小
    python scripts/28_fetch_2d3ds.py --download area_1_no_xyz.tar --out G:\\2d3ds
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

#: Redivis 数据集引用（url 里的 id，见 sdss.redivis.com/datasets/f304-a3vhsvcaf）
DATASET_ID = "f304-a3vhsvcaf"
DATASET_ORG = "sdss"
API_BASE = "https://redivis.com/api/v1"

#: 官方 checksum（来自 2D-3D-Semantics wiki 的 Checksum-Values-for-Data 页）
#: 下完一定要核对：这个数据集的分发链路很长（原始 → Redivis → 我们），
#: 不校验的话"数据损坏"会伪装成"算法不work"。
OFFICIAL_MD5: Dict[str, str] = {
    # ---- noXYZ（110 GB 全集；我们只需要其中一个 area）----
    "area_1_no_xyz.tar": "21098fbe93b561e30e79197a95fa4fd2",
    "area_2_no_xyz.tar": "d630696e4588c2329483bc07210d0ba9",
    "area_3_no_xyz.tar": "09561f6aa6a87f9ae7d1beb3973964fb",
    "area_4_no_xyz.tar": "6798c3196ae1e9e4d25073c53aec1ad5",
    "area_5a_no_xyz.tar": "961a490072093fca726a483b74169b48",
    "area_5b_no_xyz.tar": "85fd6de1a7c7247d8563ed5976dbd06d",
    "area_6_no_xyz.tar": "71543fc18f2f444fd17243e68f064ff0",
    # ---- withXYZ（766 GB 全集，含逐像素世界坐标；本项目不需要）----
    "area_1.tar.gz": "e07f142e421f1e106742de3a04bf9275",
    "area_2.tar.gz": "3670f6777a370b3935829376718978a8",
    "area_3.tar.gz": "3920a5c3a763bec6e9dd83bf18a565e9",
    "area_4.tar.gz": "47efe992496d83ae32270e448c63ad8c",
    "area_5a.tar.gz": "324aba1e8a53b3c75a0b394ef71bfcb4",
    "area_5b.tar.gz": "f78f526f0c573b05644abebf4b4871ce",
    "area_6.tar.gz": "63d5782c8e12afaddec7f69be69267c2",
}


def _token(args) -> str:
    tok = args.token or os.getenv("REDIVIS_TOKEN") or os.getenv("REDIVIS_API_TOKEN")
    if not tok:
        print("[FAIL] 没有 token。请设置环境变量 REDIVIS_TOKEN，或用 --token 传入。\n"
              "       获取方式：登录 Redivis → 账号设置 → API tokens → 新建。\n"
              "       ⚠️ token 等同密码，别写进任何会提交的文件。", file=sys.stderr)
        raise SystemExit(2)
    return tok.strip()


# --------------------------------------------------------------------------
# 1) REST 探元数据（不依赖客户端，用来确认引用名与可访问性）
# --------------------------------------------------------------------------
def probe(token: str, dataset_id: str) -> None:
    import requests

    url = f"{API_BASE}/datasets/{dataset_id}"
    print(f"GET {url}")
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    print(f"  HTTP {r.status_code}")
    if r.status_code != 200:
        print("  响应片段：" + r.text[:400])
        if r.status_code == 401:
            print("  → token 无效或没有该数据集的权限")
        elif r.status_code == 403:
            print("  → token 有效但没有被授权（可能还没接受数据集许可）")
        return
    info = r.json()
    keep = {k: v for k, v in info.items()
            if k in ("name", "qualifiedReference", "uri", "owner", "organization",
                     "description", "createdAt", "updatedAt", "publicAccessLevel",
                     "numBytes", "numFiles")}
    print("  关键字段：")
    for k, v in keep.items():
        s = str(v)
        print(f"    {k:20s} {s[:120]}")
    print("\n  完整 JSON（便于排查）：")
    print("   " + json.dumps(info, ensure_ascii=False)[:900])


# --------------------------------------------------------------------------
# 2) 用客户端列目录
# --------------------------------------------------------------------------
def _open_dataset(token: str, args):
    import redivis

    os.environ["REDIVIS_API_TOKEN"] = token
    os.environ["REDIVIS_DEFAULT_ORGANIZATION"] = DATASET_ORG
    ref = args.reference or f"{DATASET_ORG}.{DATASET_ID}"
    for candidate, kwargs in (
        (ref, {}),
        (DATASET_ID, {"organization": redivis.organization(DATASET_ORG)}),
    ):
        try:
            ds = redivis.dataset(candidate, **kwargs)
            _ = ds.properties          # 触发一次请求，确认能访问
            print(f"  数据集引用可用：{candidate!r}")
            return ds
        except Exception as exc:
            print(f"  引用 {candidate!r} 失败：{type(exc).__name__}: {str(exc)[:120]}")
    raise SystemExit("[FAIL] 无法打开数据集；先用 --probe 确认 token 与权限")


def list_contents(token: str, args) -> None:
    ds = _open_dataset(token, args)
    print(f"  数据集：{ds.properties.get('name')}  "
          f"（引用 {ds.qualified_reference}）")

    tables = ds.list_tables()
    print(f"\n  表/文件索引（{len(tables)} 个）：")
    for t in tables:
        print(f"    - {t.name}   ({t.num_rows if hasattr(t, 'num_rows') else '?'} 行)")

    for t in tables:
        print(f"\n  表 {t.name!r} 的列：")
        try:
            for v in t.list_variables():
                print(f"    {v.name:28s} {v.type}")
        except Exception as exc:
            print(f"    （取列失败：{exc}）")

    # 文件型数据集：索引表里通常有 name / size / file_id 之类的列
    for t in tables:
        try:
            print(f"\n  表 {t.name!r} 前 20 行（找文件名与大小）：")
            for i, row in enumerate(t.to_rows()):
                if i >= 20:
                    break
                print("    " + json.dumps(row, ensure_ascii=False, default=str)[:200])
        except Exception as exc:
            print(f"    （读行失败：{type(exc).__name__}: {str(exc)[:100]}）")


# --------------------------------------------------------------------------
# 3) 下载 + MD5 校验
# --------------------------------------------------------------------------
def md5_of(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.md5()
    total = path.stat().st_size
    done = 0
    t0 = time.time()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if total > 200 << 20 and done % (200 << 20) < chunk:
                pct = done / total * 100
                mbs = done / max(time.time() - t0, 1e-9) / (1 << 20)
                print(f"\r    校验中 {pct:5.1f}%  {mbs:6.1f} MB/s", end="")
    print()
    return h.hexdigest()


def download(token: str, args) -> int:
    ds = _open_dataset(token, args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 在文件索引表里找目标文件
    target = args.download
    print(f"\n  在数据集里查找 {target!r} …")
    found = None
    for t in ds.list_tables():
        try:
            for row in t.to_rows():
                vals = {str(v) for v in row.values() if v is not None}
                if any(target in v for v in vals):
                    found = (t, row)
                    break
        except Exception:
            continue
        if found:
            break
    if not found:
        print(f"  [FAIL] 没在数据集里找到 {target!r}；先跑 --list 看清有哪些文件")
        return 2
    table, row = found
    print(f"  命中：表 {table.name}  行 {json.dumps(row, ensure_ascii=False, default=str)[:220]}")

    # 用行里的文件 id 取 File 对象
    fid = None
    for k in ("id", "file_id", "fileId", "redivis_file_id", "fileID"):
        if k in row and row[k] is not None:
            fid = str(row[k])
            break
    if fid is None:
        print("  [FAIL] 这一行里找不到文件 id 字段；把上面的行结构发我，我来适配")
        return 2

    import redivis

    f = redivis.file(fid)
    dst = out_dir / target
    print(f"  下载到 {dst}")
    f.download(str(dst))

    # 校验
    expect = OFFICIAL_MD5.get(target)
    if not expect:
        print(f"  [WARN] 没有 {target} 的官方 MD5 记录（见脚本里的 OFFICIAL_MD5 表）")
        return 0
    print(f"  校验 MD5（期望 {expect}）…")
    got = md5_of(dst)
    if got == expect:
        print(f"  [OK] MD5 一致：{got}")
        return 0
    print(f"  [FAIL] MD5 不一致！\n    期望 {expect}\n    实际 {got}\n"
          f"    文件可能损坏，删掉重下：{dst}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=None, help="Redivis API token（默认读 REDIVIS_TOKEN）")
    ap.add_argument("--reference", default=None,
                    help="数据集引用，如 'sdss.2d-3d-s'（默认用 id 推导）")
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--probe", action="store_true", help="只用 REST 探元数据")
    ap.add_argument("--list", action="store_true", help="列数据集内容与列结构")
    ap.add_argument("--download", default=None, help="要下载的文件名，如 area_1_no_xyz.tar")
    ap.add_argument("--out", default=r"G:\2d3ds", help="下载目录")
    args = ap.parse_args()

    tok = _token(args)
    if args.probe:
        probe(tok, args.dataset_id)
        return 0
    if args.list:
        list_contents(tok, args)
        return 0
    if args.download:
        return download(tok, args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
