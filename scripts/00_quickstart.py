#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""00 · 最小端到端 demo（不需要任何数据、模型、网络、GPU）。

跑通这条链路：
    合成 RGB-D 序列 → 开放词汇感知 → 2D 反投影 → 3D 体素特征场
    → 物体关联 → 语义地图 → 自然语言查询 → 空间关系推理

用法::

    python scripts/00_quickstart.py
    python scripts/00_quickstart.py --frames 5 --objects 8 --save runs/demo_map.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground import load_config                                    # noqa: E402
from roboground.data.synthetic import make_synthetic_sequence         # noqa: E402
from roboground.mapping import MapBuilder                             # noqa: E402
from roboground.reasoning import RuleEngine                           # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity        # noqa: E402

PROMPTS = ["table", "chair", "cup", "box", "bottle", "sofa",
           "shelf", "monitor", "trash can", "lamp"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=3, help="视角数量")
    ap.add_argument("--objects", type=int, default=5, help="场景中物体数量")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--save", default=None, help="把地图存到该路径")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)
    log = get_logger("quickstart")

    log.info("=" * 66)
    log.info("RoboGround 快速上手（全离线：无数据 / 无模型 / 无网络 / 无 GPU）")
    log.info("=" * 66)

    cfg = load_config()
    cfg.set("perception.prompts", PROMPTS)

    # ---- 1) 合成一段"机器人在房间里移动"的 RGB-D 序列 ----
    log.info(f"[1/5] 生成合成场景：{args.frames} 个视角、{args.objects} 个物体")
    frames = make_synthetic_sequence(
        seed=args.seed, num_frames=args.frames,
        width=args.width, height=args.height, num_objects=args.objects,
    )
    log.info(f"      GT 类别：{frames[0].meta['labels']}")
    log.info(f"      图像尺寸：{frames[0].width}×{frames[0].height}，"
             f"有效深度像素：{int((frames[0].depth_m > 0).sum())}")

    # ---- 2) 建图 ----
    log.info("[2/5] 建图（感知 → 反投影 → 实例关联 → 体素融合 → 物体聚类）")
    builder = MapBuilder(cfg, prompts=PROMPTS)
    smap = builder.build_from_frames(frames)
    log.kv("地图概况", {
        "观测数": smap.meta["num_observations"],
        "体素数": smap.num_voxels,
        "物体数": smap.num_objects,
        "特征维度": smap.feature_dim,
        "类别": smap.labels,
    })

    # ---- 3) 打印物体清单 ----
    log.info("[3/5] 地图中的物体")
    for obj in sorted(smap.objects, key=lambda o: -o.confidence)[:10]:
        log.info("      " + obj.describe())

    # ---- 4) 语言查询 ----
    log.info("[4/5] 自然语言查询（词法匹配，无需文本编码器）")
    for obj in smap.objects[:2]:
        for query in (obj.label, _to_chinese(obj.label) or obj.label):
            hits = smap.query_text(query, top_k=1)
            if hits:
                log.info(f"      {query!r:14} → {hits[0].label:<12} "
                         f"@{_fmt(hits[0].position)}  分数={hits[0].score:.3f}")

    # ---- 5) 空间推理 ----
    log.info("[5/5] 规则引擎问答（几何 + 米制距离）")
    engine = RuleEngine(smap, cfg=cfg)
    questions = ["描述一下场景"]
    if smap.labels:
        questions.append(f"{smap.labels[0]} 在哪")
    if len(smap.labels) >= 2:
        questions.append(f"{smap.labels[0]} 离 {smap.labels[1]} 多远")
    for q in questions:
        res = engine.answer(q)
        log.info(f"      Q: {q}")
        log.info(f"      A: {res.answer}")

    # ---- 可选落盘 ----
    if args.save:
        path = smap.save(args.save)
        log.ok(f"地图已保存：{path}")

    log.info("=" * 66)
    log.ok("全链路跑通。下一步：")
    log.info("  · 用真实数据：python scripts/02_build_sunrgbd_index.py && "
             "python scripts/03_build_map.py --source sunrgbd")
    log.info("  · 换成真实开放词汇模型：pip install -e \".[perception]\"，"
             "然后把 configs/*.yaml 里的 detector 改成 grounding_dino")
    log.info("  · 跑基准出数字：python scripts/05_benchmark.py")
    return 0


def _fmt(v) -> str:
    return "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"


def _to_chinese(label: str):
    try:
        from roboground.mapping.query import ALIAS_LEXICON
        for concept, aliases in ALIAS_LEXICON.items():
            if concept.lower() == str(label).lower():
                for a in aliases:
                    if any("\u4e00" <= ch <= "\u9fff" for ch in a):
                        return a
    except Exception:
        pass
    return None


if __name__ == "__main__":
    raise SystemExit(main())
