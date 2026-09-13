#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""10 · 嵌入查询的分数标定（选温度与接受阈值）。

为什么需要标定
-------------
嵌入匹配的分数是"query 相似度 vs 空文本相似度"的 softmax 概率。
这个分数的**分布形态完全由温度 τ 决定**：
- τ 太小（如 0.01）→ 任何正 margin 都趋近 1.0，分数退化成阶跃函数，
  阈值失去意义（实测：查"桌子"误命中 bookshelf 得 0.971）；
- τ 太大（如 0.5）→ 所有候选都挤在 0.5 附近，同样无法区分。

所以 τ 和接受阈值必须**用数据选**，不能拍脑袋。

标定方法
--------
在真实场景上构造两组查询：
- **正样本**：地图里确实存在的类别（应当命中）；
- **负样本**：地图里确定没有的类别（应当返回空）。

然后扫描 (τ, threshold) 网格，选出让两组**分离度最大**的组合。
分离度用 "正样本最小分 - 负样本最大分"（越大越好），
若重叠则记为负值。

用法::

    python scripts/10_calibrate_embedding.py
    python scripts/10_calibrate_embedding.py --scenes 3 --output runs/calib.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.io import ensure_dir                       # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity   # noqa: E402

log = get_logger("calibrate_embedding")

#: 与已知类别无关的负样本查询（用于检验拒识能力）
NEGATIVE_QUERIES = [
    "冰箱", "微波炉", "马桶", "浴缸", "飞机", "汽车",
    "refrigerator", "airplane", "bicycle", "helicopter",
]

#: 中文→英文的类别桥（用于生成正样本查询）
ZH = {"cup": "杯子", "table": "桌子", "chair": "椅子", "monitor": "显示器",
      "door": "门", "window": "窗户", "trash can": "垃圾桶", "bottle": "瓶子",
      "laptop": "笔记本电脑", "bookshelf": "书架", "box": "箱子", "bed": "床",
      "sofa": "沙发", "desk": "书桌", "person": "人"}


def build_map(cfg, args):
    """用真实检测结果建一张地图（一次，供所有查询复用）。"""
    from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene
    from roboground.mapping import MapBuilder
    from roboground.perception import build_detector, build_encoder, build_segmenter

    detector = build_detector(cfg, name="grounding_dino")
    detector.warmup()
    segmenter = build_segmenter(cfg, name="sam")
    segmenter.warmup()
    encoder = build_encoder(cfg, name="clip")
    encoder.warmup()

    index = load_scene_index(str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz"))
    prompts = {str(p).lower() for p in cfg.get("perception.prompts", [])}
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))

    frames = []
    used = []
    for pos in np.argsort(-counts)[:4000]:
        i = int(pos)
        if len(frames) >= args.scenes:
            break
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = {str(x).lower() for x in np.asarray(index["label_flat"][off0:off1]).ravel()}
        if len(labels & prompts) < 2:
            continue
        sc = load_sunrgbd_scene(index, i, max_depth=8.0, resize=(args.width, args.height))
        if sc is not None and sc.boxes_3d.shape[0] > 0:
            frames.append(sc.to_frame())
            used.append(i)
    if not frames:
        raise RuntimeError("没有收集到场景")

    log.info(f"使用场景：{used}（{len(frames)} 帧）")

    # 真实检测 → 标签规范化 → 塞进 builder
    class _Precomputed:
        def __init__(self, per_frame, encoder, detector, segmenter):
            self._per_frame = per_frame
            self.encoder = encoder
            self.detector = detector
            self.segmenter = segmenter
            self.stats = {}

        def run(self, frame, prompts=None):
            dets = list(self._per_frame.get(frame.frame_id, []))
            if self.encoder is not None and dets:
                feats = self.encoder.encode_regions(frame, dets)
                for d, f in zip(dets, feats):
                    d.feature = f
            return dets

        def profile(self):
            return dict(self.stats)

    per_frame = {}
    for frame in frames:
        dets = detector.detect(frame, list(cfg.get("perception.prompts", [])))
        dets = segmenter.segment(frame, dets) if dets else []
        per_frame[frame.frame_id] = dets

    builder = MapBuilder(cfg, pipeline=_Precomputed(per_frame, encoder, detector, segmenter),
                         prompts=list(cfg.get("perception.prompts", [])))
    smap = builder.build_from_frames(frames)
    smap.text_encoder = encoder
    return smap, encoder


def collect_scores(smap, encoder, temperature: float):
    """在给定温度下，收集正/负样本查询的分数分布。"""
    from roboground.mapping.query import QueryEngine

    # 直接改 EmbeddingMatcher 的温度（标定用）
    engine = QueryEngine(smap, text_encoder=encoder)
    engine.embedding_matcher.temperature = float(temperature)
    engine.embedding_matcher._null_feats = None       # 温度变了不影响空文本特征，但保险起见

    map_labels = set(smap.labels)
    pos_queries = []
    for lab in map_labels:
        pos_queries.append(lab)
        if lab in ZH:
            pos_queries.append(ZH[lab])

    # 正样本：直接取 embedding 分数（绕过双阈值，看原始分布）
    def raw_scores(queries):
        out = []
        labels = [o.label for o in smap.objects]
        feats = np.stack([o.feature for o in smap.objects], axis=0) if smap.objects else None
        for q in queries:
            if feats is None:
                out.append((q, 0.0, None))
                continue
            s = engine.embedding_matcher.score(q, labels=labels, features=feats)
            best = int(np.argmax(s))
            out.append((q, float(s[best]), labels[best]))
        return out

    pos = raw_scores(pos_queries)
    neg = raw_scores(NEGATIVE_QUERIES)

    # 正样本只保留"确实命中同类"的（避免用错标签的查询污染标定）
    pos_good = []
    for q, s, lab in pos:
        if lab is None:
            continue
        # 中文查询要求命中的标签与其概念一致
        want = None
        for concept, zh in ZH.items():
            if q == zh:
                want = concept
        if want is None:
            want = q
        from roboground.mapping.query import canonical_concepts
        if canonical_concepts(want) & canonical_concepts(lab) or want == lab:
            pos_good.append((q, s, lab))

    return pos, pos_good, neg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--output", default="runs/calibration.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    from roboground import load_config

    cfg = load_config("configs/perception_openvocab.yaml")
    cfg.set("perception.encoder_kwargs.model_id", r"G:\RoboGround\weights\clip-vit-base-patch32")

    smap, encoder = build_map(cfg, args)
    log.info(f"地图：{smap.num_objects} 个物体，类别 = {smap.labels}")

    grid = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    results = []
    print()
    print(f"{'τ':>6} | {'正样本 min':>10} {'正样本 中位':>11} | "
          f"{'负样本 max':>10} {'负样本 中位':>11} | {'分离度':>8}")
    print("-" * 78)

    for tau in grid:
        pos, pos_good, neg = collect_scores(smap, encoder, tau)
        p_scores = np.array([s for _, s, _ in pos_good]) if pos_good else np.array([0.0])
        n_scores = np.array([s for _, s, _ in neg]) if neg else np.array([0.0])

        p_min = float(p_scores.min())
        n_max = float(n_scores.max())
        separation = p_min - n_max
        results.append({
            "temperature": tau,
            "pos_min": p_min,
            "pos_median": float(np.median(p_scores)),
            "neg_max": n_max,
            "neg_median": float(np.median(n_scores)),
            "separation": separation,
            "pos_samples": [{"query": q, "score": round(s, 4), "matched": lab} for q, s, lab in pos_good],
            "neg_samples": [{"query": q, "score": round(s, 4), "matched": lab} for q, s, lab in neg],
        })
        flag = "  ← 可完全分离" if separation > 0 else ""
        print(f"{tau:>6.2f} | {p_min:>10.4f} {np.median(p_scores):>11.4f} | "
              f"{n_max:>10.4f} {np.median(n_scores):>11.4f} | {separation:>+8.4f}{flag}")

    # ---- 择优：分离度最大；若都不可分，选分离度最大者（最不坏）----
    best = max(results, key=lambda r: r["separation"])
    print()
    if best["separation"] > 0:
        threshold = (best["pos_min"] + best["neg_max"]) / 2.0
        log.ok(f"最佳温度 τ={best['temperature']}，"
               f"建议接受阈值 = {threshold:.4f}（取正样本最小值与负样本最大值的中间）")
    else:
        # 无法完全分离 → 用"最大化正确率"的方式选阈值
        threshold = best["pos_median"]
        log.warn(f"没有任何温度能完全分离正负样本（最大分离度 {best['separation']:+.4f}）。")
        log.warn("这说明当前场景里 CLIP 的语义区分度有限 —— 需要改用更强的模型")
        log.warn("（如 SigLIP / CLIP-L/14）或引入更多空文本模板。")
        log.info(f"作为折中：τ={best['temperature']}，阈值={threshold:.4f}"
                 f"（该阈值下正样本约一半通过）")

    log.info(f"当前地图的实际类别：{smap.labels}")
    log.info("正/负样本明细（τ=%s）：" % best["temperature"])
    for s in best["pos_samples"]:
        log.info(f"  [正] {s['query']!r:<16} score={s['score']:.4f} → {s['matched']}")
    for s in best["neg_samples"][:6]:
        log.info(f"  [负] {s['query']!r:<16} score={s['score']:.4f} → {s['matched']}")

    out = {"best": best, "grid": results, "map_labels": smap.labels,
           "suggested_threshold": threshold}
    if args.output:
        ensure_dir(Path(args.output).parent)
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        log.ok(f"标定结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
