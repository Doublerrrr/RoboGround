#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05 · 基准测试：把"快不快、准不准"变成可写进简历的数字。

测什么
------
| 指标 | 含义 |
|---|---|
| 建图耗时/帧 | 感知 + 反投影 + 融合 + 关联的总开销 |
| 感知耗时拆解 | 检测 / 分割 / 编码各占多少 |
| 查询耗时 | 单次自然语言问答的响应时间 |
| **地图级定位精度** | 地图里的物体中心 vs GT 中心（**主要质量指标**）|
| 观测级定位精度 | 单帧反投影的中心误差 |
| 检测 P/R/F1/IoU | 与 GT 框对比（bbox 反投影基线，偏低属预期）|
| 查询命中率 | Top-1/3/5（给了期望答案时）|

用法::

    # 合成场景（无需数据）
    python scripts/05_benchmark.py --source synthetic --frames 5

    # 真实数据（需要先建索引）
    python scripts/05_benchmark.py --source sunrgbd --scenes 10

    # 真实数据 + 虚拟相机多视角
    python scripts/05_benchmark.py --source virtual --scenes 5 --views 6

    # 对比不同量化模式（感知编码器的 fp16/int8）
    python scripts/05_benchmark.py --source synthetic --quantization
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config                              # noqa: E402
from roboground.utils.io import save_json                              # noqa: E402
from roboground.utils.logging import get_logger                        # noqa: E402

PROMPTS = ["chair", "table", "desk", "monitor", "door", "window",
           "bookshelf", "trash can", "box", "bottle", "sofa", "bed"]


def collect_synthetic(args, cfg):
    from roboground.data.synthetic import make_synthetic_sequence

    frames = make_synthetic_sequence(
        seed=args.seed, num_frames=args.frames,
        width=args.width, height=args.height, num_objects=args.objects,
    )
    return frames, [f.meta["boxes_3d"] for f in frames]


def collect_sunrgbd(args, cfg):
    from roboground.data.sunrgbd import (
        SUNRGBDDataset,
        load_scene_index,
        load_sunrgbd_scene,
    )

    index_path = str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz")
    index = load_scene_index(index_path)

    wanted = {p.lower() for p in PROMPTS}
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))
    frames, gts, used = [], [], []

    for pos in np.argsort(-counts):
        i = int(pos)
        if len(frames) >= args.scenes:
            break
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = {str(x).lower() for x in np.asarray(index["label_flat"][off0:off1]).ravel()}
        if not (labels & wanted):
            continue
        scene = load_sunrgbd_scene(
            index, i, max_depth=8.0,
            resize=(args.width, args.height) if args.width and args.height else None,
        )
        if scene is None or scene.boxes_3d.shape[0] == 0:
            continue
        frames.append(scene.to_frame())
        gts.append(scene.boxes_3d)
        used.append(i)

    if not frames:
        raise RuntimeError("没有收集到任何含目标类别的场景；请检查索引或放宽 prompts")

    log = get_logger("bench")
    log.info(f"使用场景下标：{used}")
    return frames, gts


def collect_virtual(args, cfg):
    from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene
    from roboground.data.virtual_camera import scene_sequence_from_cloud

    index_path = str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz")
    index = load_scene_index(index_path)

    wanted = {p.lower() for p in PROMPTS}
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))
    frames, gts, used = [], [], []

    for pos in np.argsort(-counts):
        i = int(pos)
        if len(used) >= args.scenes:
            break
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = {str(x).lower() for x in np.asarray(index["label_flat"][off0:off1]).ravel()}
        if not (labels & wanted):
            continue
        scene = load_sunrgbd_scene(
            index, i, max_depth=8.0,
            resize=(args.width, args.height) if args.width and args.height else None,
        )
        if scene is None or scene.boxes_3d.shape[0] == 0:
            continue
        seq = scene_sequence_from_cloud(
            scene, num_frames=args.views, radius=args.radius,
            max_points=120_000, splat=2,
        )
        frames.extend(seq)
        gts.extend([scene.boxes_3d] * len(seq))
        used.append(i)

    if not frames:
        raise RuntimeError("虚拟相机没有生成任何帧")
    log = get_logger("bench")
    log.info(f"使用场景下标：{used}，共 {len(frames)} 帧（每场景 {args.views} 视角）")
    return frames, gts


def run_quantization_study(args, cfg):
    """对比感知编码器在不同量化模式下的体积/延迟（需要 torch）。"""
    import torch

    from roboground.deployment.quantization import compare_quantization

    log = get_logger("bench.quant")

    class ToyBackbone(torch.nn.Module):
        """一个形状接近轻量卷积骨干的小网络（用于量化对照实验）。"""

        def __init__(self, dim: int = 128) -> None:
            super().__init__()
            self.stem = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
            self.blocks = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, dim, 3, stride=2, padding=1),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(dim, 72),
            )

        def forward(self, x):
            return self.blocks(self.stem(x))

    rows = compare_quantization(
        build_fn=lambda: ToyBackbone(),
        input_factory=lambda: torch.randn(1, 3, args.height or 240, args.width or 320),
        modes=("none", "fp16", "int8"),
        warmup=2, iters=args.quant_iters,
    )
    log.info("量化对照实验（占位骨干网络，仅用于演示对比方法）：")
    header = f"  {'模式':<8}{'体积前(MB)':>12}{'体积后(MB)':>12}{'压缩比':>8}{'mean(ms)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        if not r.get("ok"):
            print(f"  {r['mode']:<8}{'失败':>12}  {r.get('error', '')[:60]}")
            continue
        print(f"  {r['mode']:<8}{r['size_mb_before']:>12.3f}{r['size_mb_after']:>12.3f}"
              f"{r['compression']:>8.2f}{r['mean_ms']:>10.3f}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "sunrgbd", "virtual"], default="synthetic")
    ap.add_argument("--frames", type=int, default=5, help="合成模式下的帧数")
    ap.add_argument("--scenes", type=int, default=5, help="真实模式下的场景数")
    ap.add_argument("--views", type=int, default=6, help="虚拟模式下的视角数")
    ap.add_argument("--radius", type=float, default=1.8)
    ap.add_argument("--objects", type=int, default=6, help="合成模式下的物体数")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--quantization", action="store_true", help="额外跑量化对照实验")
    ap.add_argument("--quant-iters", type=int, default=10)
    ap.add_argument("--out", default="runs/benchmark.json")
    args = ap.parse_args()

    log = get_logger("bench")
    cfg = load_config()
    prompts = list(args.prompts) if args.prompts else PROMPTS
    cfg.set("perception.prompts", prompts)

    # ---- 收集数据 ----
    log.info(f"数据来源：{args.source}")
    if args.source == "synthetic":
        frames, gts = collect_synthetic(args, cfg)
    elif args.source == "sunrgbd":
        frames, gts = collect_sunrgbd(args, cfg)
    else:
        frames, gts = collect_virtual(args, cfg)
    log.info(f"共 {len(frames)} 帧，GT 框 {sum(len(g) for g in gts)} 个")

    # ---- 查询集（用地图里真实出现的类别构造，保证可评测）----
    from roboground.eval.benchmark import benchmark_pipeline, format_report

    labels = sorted({str(l) for f in frames
                     for l in (f.meta.get("labels") or [])
                     if str(l) in set(prompts)})
    queries = [f"{lab} 在哪" for lab in labels[:5]] or ["描述一下场景"]
    expectations = labels[:5] if labels else None
    log.info(f"查询集：{queries}")

    # ---- 主基准 ----
    result = benchmark_pipeline(
        cfg, frames, prompts=prompts, queries=queries,
        gt_boxes=gts, query_expectations=expectations,
    )
    result["source"] = args.source
    result["prompts"] = prompts

    title = f"RoboGround 基准报告（{args.source}，{len(frames)} 帧）"
    report = format_report(result, title=title)
    print()
    print(report)

    # ---- 量化对照 ----
    if args.quantization:
        result["quantization"] = run_quantization_study(args, cfg)

    # ---- 落盘 ----
    if args.out:
        out = Path(args.out)
        save_json(result, out)
        out.with_suffix(".md").write_text(report, encoding="utf-8")
        log.ok(f"报告已保存：{out} 与 {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
