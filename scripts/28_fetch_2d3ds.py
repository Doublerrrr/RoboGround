#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""28 · 获取 Stanford 2D-3D-S（经 Redivis 分发）。

数据来源与访问模型（实测，不是猜的）
==================================
数据集在 Redivis 上，引用形式是 **table 引用**（不是 dataset 引用）：

    sdss_data_repository.stanford_2d_3d_semantics_dataset_2d_3d_s:f304:v1_0.no_xyz:ct1f
    └────── owner ─────┘ └────────────── dataset ──────────────┘ └version┘└table┘└id┘

实测的访问权限分层：
  · **表元数据是公开读的** —— 匿名 `redivis.table(REF)` 能成功拿到表对象；
  · **文件内容需要认证** —— 匿名取文件会 403 `access_denied`，
    客户端会尝试浏览器 OAuth（非交互 shell 里走不完）。

所以下载需要一个 **Redivis API token**，通过环境变量 `REDIVIS_API_TOKEN` 传入
（客户端源码里明确写了这个变量"只应在非交互环境使用"）。

    set REDIVIS_API_TOKEN=xxxxx
    python scripts/28_fetch_2d3ds.py --list
    python scripts/28_fetch_2d3ds.py --download area_1_no_xyz.tar

⚠️ 环境隔离（重要）
==================
官方客户端会拖进 dask / geopandas / polars / folium / shapely。
**不要装进项目的 conda 环境 `lxr`**（那是验证过 torch 的干净环境）。
用独立 venv：

    G:\\minigore\\envs\\lxr\\python.exe -m venv G:\\2d3ds\\_venv
    G:\\2d3ds\\_venv\\Scripts\\python.exe -m pip install --no-deps redivis
    G:\\2d3ds\\_venv\\Scripts\\python.exe -m pip install requests tqdm niquests pyarrow

实测：文件下载只需 `requests` + `tqdm` + `niquests` + `pyarrow`
（`pyarrow` 是 `table.file()` 内部走 `to_directory()` 时用的；
dask / geopandas / polars / folium / shapely **都不需要**）。

为什么要校验 MD5
==============
这个包的分发链路很长（原始采集 → 斯坦福 → Redivis → 本机）。
不校验的话，**"数据损坏"会伪装成"算法不 work"** —— 本项目在别处已经吃过这种亏。
官方 checksum 页给了每个包的 MD5，已内嵌在下面。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

#: Redivis **table** 引用（no_xyz 版本 = 110 GB 全集；我们只取其中一个 area）
TABLE_REF = (
    "sdss_data_repository.stanford_2d_3d_semantics_dataset_2d_3d_s"
    ":f304:v1_0.no_xyz:ct1f"
)
#: withXYZ 版本的 table 引用（766 GB，含逐像素世界坐标；本项目**不需要**）
TABLE_REF_WITHXYZ = (
    "sdss_data_repository.stanford_2d_3d_semantics_dataset_2d_3d_s"
    ":f304:v1_0.with_xyz:ct1f"
)

#: 官方 MD5（来自 2D-3D-Semantics wiki 的 Checksum-Values-for-Data 页）
OFFICIAL_MD5: Dict[str, str] = {
    # ---- noXYZ（110 GB 全集）----
    "area_1_no_xyz.tar": "21098fbe93b561e30e79197a95fa4fd2",
    "area_2_no_xyz.tar": "d630696e4588c2329483bc07210d0ba9",
    "area_3_no_xyz.tar": "09561f6aa6a87f9ae7d1beb3973964fb",
    "area_4_no_xyz.tar": "6798c3196ae1e9e4d25073c53aec1ad5",
    "area_5a_no_xyz.tar": "961a490072093fca726a483b74169b48",
    "area_5b_no_xyz.tar": "85fd6de1a7c7247d8563ed5976dbd06d",
    "area_6_no_xyz.tar": "71543fc18f2f444fd17243e68f064ff0",
    # ---- withXYZ（766 GB 全集）----
    "area_1.tar.gz": "e07f142e421f1e106742de3a04bf9275",
    "area_2.tar.gz": "3670f6777a370b3935829376718978a8",
    "area_3.tar.gz": "3920a5c3a763bec6e9dd83bf18a565e9",
    "area_4.tar.gz": "47efe992496d83ae32270e448c63ad8c",
    "area_5a.tar.gz": "324aba1e8a53b3c75a0b394ef71bfcb4",
    "area_5b.tar.gz": "f78f526f0c573b05644abebf4b4871ce",
    "area_6.tar.gz": "63d5782c8e12afaddec7f69be69267c2",
}


def _require_token(args) -> str:
    tok = args.token or os.getenv("REDIVIS_API_TOKEN")
    if not tok:
        print(
            "[FAIL] 没有 token。\n"
            "   Redivis 的表元数据是公开的，但**文件内容需要认证**"
            "（匿名取文件返回 403 access_denied）。\n"
            "   获取方式：登录 Redivis → 账号设置 → API tokens → 新建，然后：\n"
            "       set REDIVIS_API_TOKEN=xxxxx\n"
            "   ⚠️ token 等同密码，别写进任何会提交的文件。",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return tok.strip()


def _table(args, token: str):
    import redivis

    os.environ["REDIVIS_API_TOKEN"] = token          # 客户端认这个变量
    ref = args.table or (TABLE_REF_WITHXYZ if args.with_xyz else TABLE_REF)
    print(f"  打开 table：{ref}")
    return redivis.table(ref)


def cmd_list(args, token: str) -> int:
    t = _table(args, token)
    print(f"  table 名称：{getattr(t, 'name', '?')}")

    print("\n  文件索引（用 to_directory 遍历，需要 pyarrow）：")
    try:
        d = t.to_directory()
        items = d.list(recursive=True)
    except Exception as exc:
        print(f"  [FAIL] 遍历失败：{type(exc).__name__}: {str(exc)[:200]}")
        return 1

    total = 0
    rows = []
    for it in items:
        name = getattr(it, "name", str(it))
        size = getattr(it, "size", None)
        if size is None:
            size = getattr(it, "num_bytes", None)
        rows.append((name, size))
        if size:
            total += int(size)
    for name, size in sorted(rows):
        s = f"{int(size) / (1 << 30):8.2f} GB" if size else "        ? "
        print(f"    {s}  {name}")
    if total:
        print(f"    {'-' * 40}\n    合计约 {total / (1 << 30):.2f} GB")
    print("\n  提示：本项目只需要 `area_1_no_xyz.tar`（跑通全链路用），"
          "别下 766 GB 的 with_xyz 版本。")
    return 0


def md5_of(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.md5()
    total = path.stat().st_size
    done, t0 = 0, time.time()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if total > 256 << 20 and done % (256 << 20) < chunk:
                pct = done / total * 100
                mbs = done / max(time.time() - t0, 1e-9) / (1 << 20)
                print(f"\r    校验 {pct:5.1f}%  {mbs:7.1f} MB/s", end="", flush=True)
    print()
    return h.hexdigest()


def cmd_download(args, token: str) -> int:
    t = _table(args, token)
    name = args.download
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / name

    if dst.exists() and not args.force:
        print(f"  目标已存在：{dst}（要重下加 --force）")
    else:
        print(f"  取文件对象：{name}")
        f = t.file(name)
        size = getattr(f, "size", None)
        if size:
            print(f"  大小：{int(size) / (1 << 30):.2f} GB")
        print(f"  下载到：{dst}")
        # 客户端自带重试与分片（retryable_download），比自己写稳
        f.download(str(dst))

    expect = OFFICIAL_MD5.get(name)
    if not expect:
        print(f"  [WARN] 没有 {name} 的官方 MD5 记录，跳过校验")
        return 0
    print(f"  校验 MD5（官方 {expect}）…")
    got = md5_of(dst)
    if got == expect:
        print(f"  [OK] MD5 一致：{got}")
        return 0
    print(f"  [FAIL] MD5 不一致！\n    期望 {expect}\n    实际 {got}\n"
          f"    文件可能损坏 → 删掉重下：{dst}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=None,
                    help="Redivis API token（默认读环境变量 REDIVIS_API_TOKEN）")
    ap.add_argument("--table", default=None, help="覆盖 table 引用")
    ap.add_argument("--with-xyz", action="store_true",
                    help="用 with_xyz 版本（766 GB，本项目不需要）")
    ap.add_argument("--list", action="store_true", help="列文件与大小")
    ap.add_argument("--download", default=None,
                    help="要下载的文件名，如 area_1_no_xyz.tar")
    ap.add_argument("--out", default=r"G:\2d3ds", help="下载目录")
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    args = ap.parse_args()

    token = _require_token(args)
    if args.list:
        return cmd_list(args, token)
    if args.download:
        return cmd_download(args, token)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
