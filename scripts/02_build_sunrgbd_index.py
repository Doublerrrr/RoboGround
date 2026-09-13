#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02 · 构建 SUN RGB-D 场景索引（一次性，之后加载只要几百毫秒）。

为什么要建索引
-------------
SUN RGB-D 的官方元数据 `SUNRGBDMeta.mat` 有 13.9MB、10335 个 struct，
每次 `scipy.io.loadmat` 都要 10~30 秒。索引把这 10335 个场景需要的字段
（内参 K、重力对齐旋转 Rtilt、GT 3D 框、相对路径）压成一个紧凑 npz，
之后加载是毫秒级。

用法::

    python scripts/02_build_sunrgbd_index.py
    python scripts/02_build_sunrgbd_index.py --limit 500     # 只扫前 500 个（调试）
    python scripts/02_build_sunrgbd_index.py --root D:/sunrgbd_raw
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config                            # noqa: E402
from roboground.data.sunrgbd import build_scene_index, load_scene_index  # noqa: E402
from roboground.utils.logging import get_logger                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=r"G:\sunrgbd_raw",
                    help="SUN RGB-D 根目录（其下第一层应为 SUNRGBD/）")
    ap.add_argument("--meta", default=None,
                    help="SUNRGBDMeta.mat 路径（默认 <root>/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat）")
    ap.add_argument("--out", default=None, help="索引输出路径")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个场景")
    ap.add_argument("--no-verify-files", action="store_true",
                    help="跳过文件存在性校验（快，但可能索引到缺失的场景）")
    args = ap.parse_args()

    log = get_logger("build_index")
    cfg = load_config()
    root = Path(args.root)
    meta = Path(args.meta) if args.meta else root / "SUNRGBDtoolbox" / "Metadata" / "SUNRGBDMeta.mat"
    out = Path(args.out) if args.out else Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz"

    if not root.exists():
        log.error(f"数据根目录不存在：{root}")
        log.info("请确认 SUN RGB-D 已下载，或用 --root 指定正确路径。")
        return 2
    if not meta.exists():
        log.error(f"元数据不存在：{meta}")
        log.info("SUNRGBDtoolbox 通常与 SUNRGBD/ 同级；可用 --meta 指定。")
        return 2

    log.info(f"根目录  : {root}")
    log.info(f"元数据  : {meta}")
    log.info(f"输出    : {out}")
    if args.limit:
        log.info(f"限制    : 只处理前 {args.limit} 个场景")

    t0 = time.perf_counter()
    try:
        path = build_scene_index(
            str(root), str(meta), str(out),
            limit=args.limit, require_files=not args.no_verify_files,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        log.error(str(exc))
        return 2
    elapsed = time.perf_counter() - t0

    index = load_scene_index(str(path))
    n = len(index["sequences"])
    n_boxes = index["box_flat"].shape[0]
    log.kv("索引概况", {
        "场景数": n,
        "GT 框总数": n_boxes,
        "平均框数/场景": round(n_boxes / max(n, 1), 2),
        "构建耗时(s)": round(elapsed, 1),
        "文件大小(MB)": round(path.stat().st_size / 1e6, 2),
    })

    # 类别分布（前 15）
    from collections import Counter

    dist = Counter(str(x) for x in index["label_flat"])
    log.info("类别分布（前 15）：")
    for label, cnt in dist.most_common(15):
        log.info(f"    {label:<18} {cnt}")

    log.ok("下一步：python scripts/03_build_map.py --source sunrgbd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
