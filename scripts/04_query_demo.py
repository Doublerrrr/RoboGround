#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""04 · 语言查询与空间推理 demo（对已存盘的地图提问）。

用法::

    python scripts/04_query_demo.py runs/map.npz
    python scripts/04_query_demo.py runs/map.npz "杯子在哪" "桌子上面有什么"
    python scripts/04_query_demo.py runs/map.npz --auto          # 自动生成一批问题
    python scripts/04_query_demo.py runs/map.npz --relations      # 打印全部空间关系
    python scripts/04_query_demo.py runs/map.npz --vlm            # 尝试用 VLM 增强回答
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config                              # noqa: E402
from roboground.mapping.semantic_map import SemanticMap                # noqa: E402
from roboground.reasoning import HybridReasoner, RuleEngine            # noqa: E402
from roboground.reasoning.spatial_relations import describe_relations  # noqa: E402
from roboground.utils.logging import get_logger                        # noqa: E402


def auto_questions(smap) -> list:
    """根据地图内容自动生成一批有代表性的问题。"""
    labels = smap.labels
    qs = ["描述一下场景"]
    for lab in labels[:3]:
        qs.append(f"{lab} 在哪")
    if len(labels) >= 2:
        qs.append(f"{labels[0]} 离 {labels[1]} 多远")
        qs.append(f"{labels[1]} 上有什么")
    if labels:
        qs.append(f"离我最近的 {labels[0]}")
        qs.append(f"有几个 {labels[0]}")
    qs.append("冰箱在哪")        # 故意问一个地图里没有的，验证"没找到"路径
    return qs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="地图 npz 路径")
    ap.add_argument("questions", nargs="*", default=None)
    ap.add_argument("--auto", action="store_true", help="自动生成问题")
    ap.add_argument("--relations", action="store_true", help="打印所有物体两两关系")
    ap.add_argument("--relations-radius", type=float, default=3.0)
    ap.add_argument("--vlm", action="store_true", help="尝试用 VLM 增强（需装依赖与权重）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    log = get_logger("query")
    cfg = load_config()

    try:
        smap = SemanticMap.load(args.map)
    except FileNotFoundError:
        log.error(f"地图不存在：{args.map}")
        log.info("请先建图：python scripts/03_build_map.py --save runs/map.npz")
        return 2

    log.info(f"地图：{smap}")
    log.info(f"类别：{smap.labels}")
    print()

    # ---------------- 原始地图查询（不走规则引擎）----------------
    questions = list(args.questions or [])
    if args.auto or not questions:
        questions = auto_questions(smap)

    log.info("— 地图级文本查询（top-1 命中）—")
    for q in questions[:8]:
        hits = smap.query_text(q, top_k=args.top_k)
        if hits:
            top = hits[0]
            log.info(f"  {q!r:22} → {str(top.label):<14} "
                     f"@{_fmt(top.position)}  score={top.score:.3f} ({top.matched_by})")
        else:
            log.info(f"  {q!r:22} → （无命中）")
    print()

    # ---------------- 规则引擎问答 ----------------
    log.info("— 规则引擎问答（几何 + 米制距离，可解释）—")
    engine = RuleEngine(smap, cfg=cfg)
    outputs = []
    for q in questions:
        res = engine.answer(q)
        print(f"  Q: {q}")
        print(f"  A: {res.answer}")
        if res.relations:
            for r in res.relations[:3]:
                print(f"     · 关系：{r.to_text()}")
        if res.distances:
            dist_txt = ", ".join(f"{k}={v:.3f}m" for k, v in list(res.distances.items())[:4])
            print(f"     · 距离：{dist_txt}")
        print()
        outputs.append(res.to_dict())

    # ---------------- 全部空间关系 ----------------
    if args.relations:
        rels = engine.all_relations(max_distance=args.relations_radius)
        log.info(f"— 空间关系（距离 ≤ {args.relations_radius}m，共 {len(rels)} 条）—")
        print(describe_relations(rels, top=30, use_chinese=True))
        print()

    # ---------------- 可选 VLM ----------------
    if args.vlm:
        log.info("— VLM 增强（若不可用会自动退回规则引擎）—")
        reasoner = HybridReasoner(smap, cfg=cfg)
        if reasoner.use_vlm and reasoner.vlm.is_available():
            for q in questions[:3]:
                res = reasoner.answer(q)
                print(f"  Q: {q}\n  A: {res.answer}  (backend={res.backend})\n")
        else:
            err = getattr(reasoner.vlm, "load_error", None) if reasoner.vlm else "未配置"
            log.warn(f"VLM 不可用：{err}")
            log.info('安装方式：pip install -e ".[vlm]"；'
                     "并在 config 里把 reasoning.backend 设为 qwen2vl")

    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2, default=str))
    return 0


def _fmt(v) -> str:
    return "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"


if __name__ == "__main__":
    raise SystemExit(main())
