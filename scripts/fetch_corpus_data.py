#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载语料数据：HF 镜像 + 多线程分块 + 断点续传。

为什么要自己写下载器（而不是 `huggingface_hub.snapshot_download`）
=================================================================
1. **HF 主站在国内不可达**，必须走镜像（`HF_ENDPOINT=https://hf-mirror.com`）；
   而镜像站的**单连接被限速**（实测 0.76 MB/s）。多连接分块能到 ~1.3 MB/s，
   对一个 2 GB 的语料就是 40 分钟 → 20 分钟的区别，且规模越大差距越明显。
2. **必须能断点续传** —— 数据下载动辄几十分钟，中断是常态。
   本脚本按 **chunk 粒度**记录进度：重跑时已完成的分块直接跳过。
3. **必须可复现** —— 语料从哪来、多大、什么版本，都要写进 `download_manifest.json`，
   否则别人拿到这份代码根本没法复现数据。

用法::

    python scripts/fetch_corpus_data.py --dataset msrvtt
    python scripts/fetch_corpus_data.py --dataset msrvtt --chunks 12
    python scripts/fetch_corpus_data.py --dataset msrvtt --verify-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.logging import get_logger, set_verbosity  # noqa: E402

log = get_logger("fetch_corpus")

ROOT = Path(__file__).resolve().parents[1]
MIRROR = "https://hf-mirror.com"

#: 已知数据源清单。**新增数据源时在这里登记**，保证下载可复现。
CATALOG: Dict[str, Dict[str, Any]] = {
    "msrvtt": {
        "repo": "AlexZigma/msr-vtt",
        "files": [
            {"path": "data/test_videodatainfo.json.zip", "dest": "data/raw/msrvtt"},
            {"path": "data/test_videos.zip", "dest": "data/raw/msrvtt",
             "extract_to": "data/raw/msrvtt/test_videos", "chunks": 10},
            {"path": "data/train-00000-of-00001-60e50ff5fbbd1bb5.parquet",
             "dest": "data/raw/msrvtt"},
            {"path": "data/val-00000-of-00001-01bacdd7064306bc.parquet",
             "dest": "data/raw/msrvtt"},
        ],
        "note": "MSR-VTT 测试集：2990 条真实视频 + 59800 条字幕",
    },
    "activitynet": {
        "repo": "friedrichor/ActivityNet_Captions",
        "files": [
            {"path": "activitynet_captions_train.json", "dest": "data/raw/activitynet"},
            {"path": "ActivityNet_Videos.tar.part-000", "dest": "data/raw/activitynet",
             "chunks": 10, "optional": True},
        ],
        "note": "ActivityNet Captions：长视频 + 稠密时序字幕（仅登记，未默认下载）",
    },
}


# =============================================================================
def head_size(url: str) -> int:
    """取远端文件大小（支持 range 的前提）。

    ⚠️ **必须取最后一个 `Content-Length`，不能取第一个。**
    `curl -I -L` 会把重定向链上每一跳的响应头都打出来：
    镜像站先返回 `302 Found`（带一个几百字节的 body，于是有
    `Content-Length: 1068`），再跳到真正的对象（`Content-Length: 1961970391`）。
    取第一个会让脚本以为 2 GB 的文件只有 1068 字节，
    于是"分块下载"出 10 个 107 字节的碎片、合并出一个损坏的 zip ——
    **而且整个过程不报错**（HTTP 码全是 200/206）。
    镜像站另外提供 `X-Linked-Size`，用它交叉校验。
    """
    out = subprocess.run(["curl", "-sIL", "-m", "60", url],
                         capture_output=True, text=True)
    sizes: List[int] = []
    linked: Optional[int] = None
    for line in out.stdout.splitlines():
        low = line.lower()
        if low.startswith("content-length:"):
            try:
                sizes.append(int(line.split(":", 1)[1].strip()))
            except ValueError:
                continue
        elif low.startswith("x-linked-size:"):
            try:
                linked = int(line.split(":", 1)[1].strip())
            except ValueError:
                continue
    if not sizes:
        raise RuntimeError(f"拿不到 content-length：{url}")
    size = sizes[-1]
    if linked is not None and linked != size:
        log.warn(f"X-Linked-Size({linked}) 与最终 Content-Length({size}) 不一致，"
                 f"取后者")
    return size


def fetch_chunk(url: str, start: int, end: int, dest: Path,
                retries: int = 30) -> int:
    """下载 `[start, end]` 区间到 `dest`，已下完的部分自动跳过（断点续传）。

    设计要点
    --------
    1. **不要把 `curl -C -` 和 `-r` 混用** —— 两者的语义会打架：
       `-C -` 按"输出文件已有多少字节"决定从资源哪个偏移开始，
       而 `-r` 又显式指定了范围，最终请求的是哪一段取决于 curl 版本。
       这里改成**显式补缺**：算出还差哪一段，下到 `incoming` 再追加，
       语义唯一、可验证。
    2. **`dest` 自身就是累加器**（不是"临时文件拼好再搬"）。
       下载动辄几十分钟，进程被杀/断网是常态；只有把已收到的字节
       **立刻落到最终文件上**，续传粒度才能细到"最后一次 curl 调用"。
       （早期版本用 `dest.tmp` 做暂存、成功后才 append 进 `dest`，
       结果中途被杀时 tmp 里的数据全废 —— 实测重启后进度归零。）
    3. **只用截断、不用删除** —— `dest` 越界时 `open("wb")` 清零，
       `incoming` 用后同样清零。避免依赖删除操作（某些沙箱/审计环境
       会拦截批量删除并终止进程）。
    """
    want = end - start + 1
    have = dest.stat().st_size if dest.exists() else 0
    if have > want:              # 分块大小变了（换过 --chunks），整块重来
        with dest.open("wb"):
            pass
        have = 0

    incoming = dest.with_suffix(dest.suffix + ".incoming")

    # 上一轮被中断时 `incoming` 里可能留着一段**有效尾巴**（对应 [start+have, ...]）：
    # 先把它收编进 dest 再清空，这样"被杀在 curl 中途"也不会浪费已下的字节。
    if incoming.exists() and incoming.stat().st_size > 0:
        with dest.open("ab") as out, incoming.open("rb") as fh:
            shutil.copyfileobj(fh, out, length=1 << 22)
        with incoming.open("wb"):
            pass
        have = dest.stat().st_size
        if have > want:
            with dest.open("wb"):
                pass
            have = 0

    for attempt in range(3):
        if have >= want:
            return want
        cur = start + have
        cmd = ["curl", "-L", "-s", "--retry", str(retries),
               "--retry-delay", "5", "--retry-all-errors", "-m", "3600",
               "-r", f"{cur}-{end}", "-o", str(incoming), url]
        subprocess.run(cmd, check=False)
        if incoming.exists() and incoming.stat().st_size > 0:
            with dest.open("ab") as out, incoming.open("rb") as fh:
                shutil.copyfileobj(fh, out, length=1 << 22)
            with incoming.open("wb"):
                pass
            have = dest.stat().st_size
        if have >= want:
            return want
        log.warn(f"分块未完成（{dest.name}，{have}/{want} 字节，"
                 f"第 {attempt + 1} 次），重试…")
        time.sleep(3)
    return have


def download_file(url: str, dest: Path, chunks: int = 1) -> Dict[str, Any]:
    """多线程分块下载 + 合并。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = head_size(url)

    # 小文件不值得分块（每块至少 8MB），否则 1MB 的标注文件会被切成
    # 10 个 100KB 的碎片，多开 10 个连接纯属浪费且更容易被限速。
    chunks = max(1, min(chunks, total // (8 << 20) or 1))

    if chunks <= 1:
        got = fetch_chunk(url, 0, total - 1, dest)
        return {"bytes": got, "expected": total, "chunks": 1,
                "ok": got == total}

    part_dir = dest.with_suffix(dest.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    step = (total + chunks - 1) // chunks
    t0 = time.perf_counter()

    jobs = []
    with ThreadPoolExecutor(max_workers=chunks) as ex:
        for i in range(chunks):
            a = i * step
            b = min(a + step - 1, total - 1)
            if a > b:
                break
            jobs.append(ex.submit(fetch_chunk, url, a, b,
                                  part_dir / f"part-{i:03d}.bin"))
        done = 0
        for fut in as_completed(jobs):
            done += 1
            log.info(f"  分块 {done}/{len(jobs)} 完成")

    # 合并（按序号，不能按目录顺序 —— 文件系统顺序不保证）
    parts = sorted(part_dir.glob("part-*.bin"))
    with dest.open("wb") as out:
        for p in parts:
            with p.open("rb") as fh:
                shutil.copyfileobj(fh, out, length=1 << 22)
    size = dest.stat().st_size
    dt = time.perf_counter() - t0
    log.info(f"  合并完成：{size / 1e6:.1f} MB，用时 {dt / 60:.1f} 分钟，"
             f"均速 {size / dt / 1e6:.2f} MB/s")
    # ⚠️ 合并后必须**校验总大小**。分块下载最危险的失败模式不是"报错"，
    # 而是"每一块都拿到了 HTTP 206、合并出来却是个长度不对的文件"——
    # 这种损坏文件在后续 zip 解压/解码时才炸，且很难归因。
    ok = size == total
    if not ok:
        log.error(f"  大小不符：{size} != {total}（文件可能损坏）")
    # 只有校验通过才清理分块；失败时保留，方便排查与续传。
    # 清理是 **best-effort**：某些沙箱会拦截批量删除，那不该影响下载结果 ——
    # 合并好的文件已经在磁盘上，下次运行会直接命中"已存在且大小一致"。
    if ok:
        try:
            shutil.rmtree(part_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            log.warn(f"  分块目录未清理（{exc}）；不影响结果，可手动删 {part_dir}")
    return {"bytes": size, "expected": total, "chunks": chunks,
            "seconds": round(dt, 1), "mb_per_sec": round(size / dt / 1e6, 3),
            "ok": ok}


def unzip_to(src: Path, out_dir: Path) -> int:
    """解压 zip（已解压则跳过）。返回文件数。"""
    import zipfile

    marker = out_dir / ".unzipped"
    if marker.exists():
        return sum(1 for _ in out_dir.rglob("*") if _.is_file())
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(out_dir)
    n = sum(1 for p in out_dir.rglob("*") if p.is_file())
    marker.write_text(f"{n} files\n", encoding="utf-8")
    log.info(f"  解压 {src.name} → {out_dir}（{n} 个文件）")
    return n


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="下载语料数据（HF 镜像 + 分块续传）")
    ap.add_argument("--dataset", default="msrvtt", choices=sorted(CATALOG))
    ap.add_argument("--chunks", type=int, default=0,
                    help="分块数（0 = 用清单里的默认值）")
    ap.add_argument("--only", default="", help="只下载路径包含该子串的文件")
    ap.add_argument("--verify-only", action="store_true", help="只校验已有文件")
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    set_verbosity(2 if args.verbose else 1)

    spec = CATALOG[args.dataset]
    log.info(f"数据源 {args.dataset}：{spec['note']}")
    manifest: Dict[str, Any] = {"dataset": args.dataset, "repo": spec["repo"],
                                "mirror": MIRROR, "files": []}

    for f in spec["files"]:
        if f.get("optional") and not args.include_optional:
            log.info(f"跳过可选文件 {f['path']}")
            continue
        if args.only and args.only not in f["path"]:
            continue
        dest_dir = ROOT / f["dest"]
        dest = dest_dir / Path(f["path"]).name
        url = f"{MIRROR}/datasets/{spec['repo']}/resolve/main/{f['path']}"
        log.info(f"→ {f['path']}")

        expected = head_size(url)
        if dest.exists() and dest.stat().st_size == expected:
            log.info(f"  已存在且大小一致（{expected / 1e6:.1f} MB），跳过")
            info = {"bytes": expected, "expected": expected, "cached": True}
        elif args.verify_only:
            have = dest.stat().st_size if dest.exists() else 0
            log.warn(f"  校验失败：{have} / {expected} bytes")
            info = {"bytes": have, "expected": expected, "ok": False}
        else:
            info = download_file(url, dest,
                                 chunks=args.chunks or f.get("chunks", 1))
            info["ok"] = info["bytes"] == info["expected"]

        info.update({"path": f["path"], "url": url,
                     "local": str(dest.relative_to(ROOT))})

        if f.get("extract_to") and dest.exists():
            n = unzip_to(dest, ROOT / f["extract_to"])
            info["extracted_files"] = n
        manifest["files"].append(info)

    out = ROOT / "data/raw/download_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    log.info(f"下载清单 → {out}")
    bad = [f for f in manifest["files"] if not f.get("ok", True)]
    if bad:
        log.error(f"{len(bad)} 个文件未完成")
        return 1
    log.ok("全部就绪")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
