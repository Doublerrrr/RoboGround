"""命令行入口：`roboground <subcommand>` 或 `python -m roboground.cli`。

子命令
------
```
roboground check                环境自检（依赖、GPU、数据、后端可用性）
roboground backends             列出所有可用的感知后端
roboground demo                 跑一个最小端到端 demo（合成场景 → 建图 → 查询）
roboground index                构建 SUN RGB-D 场景索引
roboground map                 用真实/合成数据建图并存盘
roboground query               对已存盘的地图做语言查询
roboground bench               跑基准并输出报告
roboground label               跑自动化标注
```

设计原则：**每个子命令都可独立运行、失败信息可操作**，
不依赖任何外部脚本或环境变量。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

VERSION = "0.1.0"


# ==========================================================================
# check
# ==========================================================================
def cmd_check(args: argparse.Namespace) -> int:
    from roboground.config import load_config, resolve_device
    from roboground.utils.logging import get_logger

    log = get_logger("cli.check")
    cfg = load_config(args.config)
    ok = True

    log.info("=" * 62)
    log.info(f"RoboGround 环境自检 v{VERSION}")
    log.info("=" * 62)

    # Python
    log.info(f"Python      : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 9):
        log.error("需要 Python >= 3.9")
        ok = False

    # 核心依赖
    for mod in ("numpy", "scipy", "yaml", "PIL", "matplotlib"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            log.ok(f"{mod:<12}: {ver}")
        except ImportError:
            log.warn(f"{mod:<12}: 缺失（pip install -e .）")
            ok = False

    # torch / GPU
    try:
        import torch

        cuda = torch.cuda.is_available()
        log.ok(f"{'torch':<12}: {torch.__version__}  cuda={cuda}")
        if cuda:
            log.ok(f"{'GPU':<12}: {torch.cuda.get_device_name(0)}  "
                   f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        log.info(f"{'device':<12}: {resolve_device(cfg.project.device)}")
    except ImportError:
        log.warn(f"{'torch':<12}: 未安装（几何/建图/推理仍可用，模型后端不可用）")

    # 可选依赖
    for mod, extra in (("transformers", "perception/vlm"),
                       ("sklearn", "聚类（有内置回退）"),
                       ("onnx", "export"),
                       ("onnxruntime", "export"),
                       ("pytest", "dev")):
        try:
            m = __import__(mod)
            log.ok(f"{mod:<12}: {getattr(m, '__version__', '?')}  [{extra}]")
        except ImportError:
            log.info(f"{mod:<12}: 未安装（可选，用于 {extra}）")

    # ROS2
    from roboground.deployment.ros2.bridge import ROS2_AVAILABLE

    log.info(f"{'rclpy':<12}: {'可用' if ROS2_AVAILABLE else '不可用（ROS 节点将无法启动，其余功能不受影响）'}")

    # 感知后端
    from roboground.perception import (
        available_detectors,
        available_encoders,
        available_segmenters,
    )

    log.info("-" * 62)
    log.info(f"检测器后端 : {sorted(set(available_detectors()))}")
    log.info(f"分割器后端 : {sorted(set(available_segmenters()))}")
    log.info(f"编码器后端 : {sorted(set(available_encoders()))}")
    log.info(f"当前配置   : detector={cfg.perception.detector} "
             f"segmenter={cfg.perception.segmenter} encoder={cfg.perception.encoder}")

    # 数据
    log.info("-" * 62)
    processed = Path(str(cfg.get("data.sunrgbd_processed", "")))
    log.info(f"SUN RGB-D 预处理目录 : {processed}  "
             f"{'存在' if processed.exists() else '不存在（可选）'}")

    index_path = Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz"
    log.info(f"场景索引            : {index_path}  "
             f"{'已建' if index_path.exists() else '未建（运行 roboground index）'}")

    raw_root = Path(r"G:\sunrgbd_raw")
    log.info(f"SUN RGB-D 原始数据   : {raw_root}  "
             f"{'存在' if raw_root.exists() else '不存在（可选）'}")

    log.info("=" * 62)
    if ok:
        log.ok("核心环境检查通过。")
        log.info("下一步：python scripts/00_quickstart.py  (最小端到端 demo)")
    else:
        log.warn("有核心依赖缺失，请先 pip install -e . （见 setup_env.md）")
    return 0 if ok else 1


# ==========================================================================
# backends
# ==========================================================================
def cmd_backends(args: argparse.Namespace) -> int:
    from roboground.perception import (
        available_detectors,
        available_encoders,
        available_segmenters,
        build_encoder,
    )
    from roboground.config import load_config
    from roboground.utils.logging import get_logger

    log = get_logger("cli.backends")
    cfg = load_config(args.config)

    log.info("检测器（Detector）")
    for name, cls in available_detectors().items():
        log.info(f"  {name:<16} -> {cls}")
    log.info("分割器（Segmenter）")
    for name, cls in available_segmenters().items():
        log.info(f"  {name:<16} -> {cls}")
    log.info("编码器（Encoder）")
    for name, cls in available_encoders().items():
        log.info(f"  {name:<16} -> {cls}")

    log.info("-" * 50)
    log.info("编码器能力对比（决定能否做「真·开放词汇」文本查询）")
    for name in ("color_hist", "dinov2", "clip"):
        try:
            enc = build_encoder(cfg, name=name)
            log.info(f"  {name:<12} dim={enc.feature_dim:<5} 支持文本={enc.supports_text}")
        except Exception as exc:
            log.info(f"  {name:<12} 不可用（{type(exc).__name__}）")
    return 0


# ==========================================================================
# demo
# ==========================================================================
def cmd_demo(args: argparse.Namespace) -> int:
    from roboground.config import load_config
    from roboground.data.synthetic import make_synthetic_sequence
    from roboground.mapping import MapBuilder
    from roboground.reasoning import RuleEngine
    from roboground.utils.logging import get_logger

    log = get_logger("cli.demo")
    cfg = load_config(args.config)
    prompts = ["table", "chair", "cup", "box", "bottle", "sofa",
               "shelf", "monitor", "trash can", "lamp"]
    cfg.set("perception.prompts", prompts)

    log.info("1) 生成合成房间（3 视角）")
    frames = make_synthetic_sequence(seed=args.seed, num_frames=3,
                                     width=160, height=120, num_objects=5)
    log.info(f"   GT 类别：{frames[0].meta['labels']}")

    log.info("2) 建图（感知 → 反投影 → 实例关联 → 体素融合）")
    smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
    log.kv("地图概况", {
        "物体数": smap.num_objects, "体素数": smap.num_voxels,
        "特征维度": smap.feature_dim, "类别": smap.labels,
    })

    log.info("3) 语言查询")
    engine = RuleEngine(smap, cfg=cfg)
    for q in ["描述一下场景"] + [f"{lab} 在哪" for lab in smap.labels[:2]]:
        res = engine.answer(q)
        log.info(f"   Q: {q}")
        log.info(f"   A: {res.answer}")

    log.ok("demo 完成。用 --json 可输出结构化结果。")
    if args.json:
        print(json.dumps({
            "objects": [o.to_dict() for o in smap.objects],
            "labels": smap.labels,
            "map_meta": smap.meta,
        }, ensure_ascii=False, indent=2, default=str))
    return 0


# ==========================================================================
# index
# ==========================================================================
def cmd_index(args: argparse.Namespace) -> int:
    from roboground.config import load_config
    from roboground.data.sunrgbd import build_scene_index
    from roboground.utils.logging import get_logger

    log = get_logger("cli.index")
    cfg = load_config(args.config)
    out = args.out or str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz")
    try:
        path = build_scene_index(args.root, args.meta, out, limit=args.limit)
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 2
    log.ok(f"索引已生成：{path}")
    return 0


# ==========================================================================
# map
# ==========================================================================
def cmd_map(args: argparse.Namespace) -> int:
    import numpy as np

    from roboground.config import load_config
    from roboground.data.synthetic import make_synthetic_sequence
    from roboground.mapping import MapBuilder
    from roboground.utils.logging import get_logger

    log = get_logger("cli.map")
    cfg = load_config(args.config)
    if args.prompts:
        cfg.set("perception.prompts", args.prompts)
    prompts = list(cfg.get("perception.prompts", []))

    if args.source == "synthetic":
        frames = make_synthetic_sequence(seed=args.seed, num_frames=args.frames,
                                         width=320, height=240, num_objects=6)
    else:
        from roboground.data.sunrgbd import SUNRGBDDataset, load_scene_index

        index_path = str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz")
        index = load_scene_index(index_path)
        ds = SUNRGBDDataset(index, require_gt=True, resize=(320, 240))
        frames = [ds.sample(1, seed=args.seed + i)[0].to_frame() for i in range(args.frames)]

    smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
    out = args.out or "runs/map.npz"
    smap.save(out)
    log.ok(f"地图已保存：{out}")
    log.kv("概况", {"物体": smap.num_objects, "体素": smap.num_voxels, "类别": smap.labels})
    return 0


# ==========================================================================
# query
# ==========================================================================
def cmd_query(args: argparse.Namespace) -> int:
    from roboground.config import load_config
    from roboground.mapping.semantic_map import SemanticMap
    from roboground.reasoning import RuleEngine
    from roboground.utils.logging import get_logger

    log = get_logger("cli.query")
    cfg = load_config(args.config)
    smap = SemanticMap.load(args.map)
    engine = RuleEngine(smap, cfg=cfg)

    queries: List[str] = list(args.questions or [])
    if not queries:
        queries = ["描述一下场景"] + [f"{lab} 在哪" for lab in smap.labels[:5]]

    results = []
    for q in queries:
        res = engine.answer(q)
        log.info(f"Q: {q}")
        log.info(f"A: {res.answer}")
        results.append(res.to_dict())

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0


# ==========================================================================
# bench
# ==========================================================================
def cmd_bench(args: argparse.Namespace) -> int:
    from roboground.config import load_config
    from roboground.data.synthetic import make_synthetic_sequence
    from roboground.eval import benchmark_pipeline, format_report
    from roboground.utils.io import save_json
    from roboground.utils.logging import get_logger

    log = get_logger("cli.bench")
    cfg = load_config(args.config)
    prompts = list(cfg.get("perception.prompts", []))
    frames = make_synthetic_sequence(seed=args.seed, num_frames=args.frames,
                                     width=320, height=240, num_objects=6)
    gt = [f.meta["boxes_3d"] for f in frames]
    labels = sorted({str(l) for f in frames for l in f.meta["labels"]})
    queries = [f"{lab} 在哪" for lab in labels[:5]] or ["描述一下场景"]

    result = benchmark_pipeline(cfg, frames, prompts=prompts, queries=queries, gt_boxes=gt)
    report = format_report(result, title="RoboGround 基准报告（合成场景）")
    print(report)

    if args.out:
        save_json(result, args.out)
        Path(args.out).with_suffix(".md").write_text(report, encoding="utf-8")
        log.ok(f"报告已保存：{args.out} / {Path(args.out).with_suffix('.md')}")
    return 0


# ==========================================================================
# label
# ==========================================================================
def cmd_label(args: argparse.Namespace) -> int:
    from roboground.config import load_config
    from roboground.data.auto_label import AutoLabeler
    from roboground.data.synthetic import make_synthetic_sequence
    from roboground.utils.logging import get_logger

    log = get_logger("cli.label")
    cfg = load_config(args.config)
    prompts = ["table", "chair", "cup", "box", "bottle", "sofa",
               "shelf", "monitor", "trash can", "lamp"]
    cfg.set("perception.prompts", prompts)

    frames = make_synthetic_sequence(seed=args.seed, num_frames=args.frames,
                                     width=320, height=240, num_objects=6)
    report = AutoLabeler(cfg, prompts=prompts).label_frames(frames, out_dir=args.out)
    log.kv("标注报告", {k: v for k, v in report.items() if k != "class_distribution"})
    log.info(f"类别分布：{report['class_distribution']}")
    return 0


# ==========================================================================
# 解析
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roboground",
        description="RoboGround —— 面向服务机器人的开放词汇 3D 场景理解与语言接地系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version=f"roboground {VERSION}")
    sub = p.add_subparsers(dest="command")

    def _add_config(sp):
        sp.add_argument("--config", default=None, help="配置文件路径（默认用内置默认配置）")

    sp = sub.add_parser("check", help="环境自检")
    _add_config(sp)
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("backends", help="列出可用感知后端")
    _add_config(sp)
    sp.set_defaults(func=cmd_backends)

    sp = sub.add_parser("demo", help="最小端到端 demo（合成场景）")
    _add_config(sp)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--json", action="store_true", help="额外输出 JSON")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("index", help="构建 SUN RGB-D 场景索引")
    _add_config(sp)
    sp.add_argument("--root", default=r"G:\sunrgbd_raw")
    sp.add_argument("--meta", default=r"G:\sunrgbd_raw\SUNRGBDtoolbox\Metadata\SUNRGBDMeta.mat")
    sp.add_argument("--out", default=None)
    sp.add_argument("--limit", type=int, default=None, help="只处理前 N 个场景（调试）")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("map", help="建图并存盘")
    _add_config(sp)
    sp.add_argument("--source", choices=["synthetic", "sunrgbd"], default="synthetic")
    sp.add_argument("--frames", type=int, default=3)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--prompts", nargs="*", default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_map)

    sp = sub.add_parser("query", help="对已存盘地图做语言查询")
    _add_config(sp)
    sp.add_argument("map", help="地图 npz 路径")
    sp.add_argument("questions", nargs="*", default=None)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("bench", help="跑基准并输出报告")
    _add_config(sp)
    sp.add_argument("--frames", type=int, default=3)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("label", help="跑自动化标注")
    _add_config(sp)
    sp.add_argument("--frames", type=int, default=4)
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--out", default="runs/labels")
    sp.set_defaults(func=cmd_label)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
