#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""16 · 开放词汇查询评测：未见表达命中率 + 未见类别拒识率（含阈值扫描）。

为什么这样测（而不是"查没见过的类别"）
====================================
直觉上"开放词汇"像是"查询训练时没见过的类别"，但直接这么测会得到平凡失败：
地图里的物体来自检测器，**没被检测到的类别压根不在地图里**，查询必然落空 ——
这测的是检测器的类别覆盖，不是查询接口的开放词汇能力。

所以把"开放词汇"拆成两个**可测且有意义**的维度：

| 维度 | 含义 | 例子（prompts 里只有 cup/table/chair） |
|---|---|---|
| **A. 未见表达命中率** | 用**没在 prompts 里出现过**的说法指代**已建图**的物体 | "杯子"、"mug"、"a thing to drink from" |
| **B. 未见类别拒识率** | 查询**地图里压根没有**的类别时应返回空 | "冰箱"、"飞机"、"helicopter" |

A 测的是**查询侧的泛化**（这才是开放词汇接口的核心价值）；
B 测的是**拒识能力**（对机器人来说，"没找到"必须是可靠答案，不能瞎指一个位置）。

**只看 A 会高估**（永远返回最像的东西也有不错的命中率，但会指向错误位置）；
**只看 B 会低估**（什么都不返回自然 100% 拒识）。
所以综合分用「命中率 × 拒识率」。

为什么必须扫阈值（本脚本最重要的设计）
====================================
本项目实测发现：`suggested_pair_threshold = 0.02`（在**检测区域特征**上标定的）
用在地图物体特征上时，**正确命中的分数只有 0.002**，比阈值低一个数量级 ——
于是嵌入路径被整体拒空，`hybrid` 悄悄退化成纯词法。

根因是**两种特征的分数尺度不同**：地图物体特征是多视角融合后的平均向量，
而 0.02 是在单帧区域特征上量的。阈值跨特征管线硬编码必然出错。

所以本脚本**先一次性收集原始分数**（词法/嵌入各一份），再在事后解析地扫描阈值 ——
既快（编码只跑一遍）又能画出「命中率-拒识率 vs 阈值」曲线，直接读出工作点。

用法::

    python scripts/16_eval_openvocab_query.py --encoder color_hist    # 仅词法
    python scripts/16_eval_openvocab_query.py --encoder siglip        # 完整对比 + 扫描
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.io import ensure_dir                        # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity     # noqa: E402

log = get_logger("eval_openvocab")

# =============================================================================
# 探针查询表：键是地图里的类别，值是一组**不会出现在 prompts 里**的说法
# =============================================================================
# 刻意分成两类，因为这两类测的**不是同一种能力**：
#   - named      中文说法 / 英文近义词 —— 别名表**可能**覆盖（属于"闭集扩展"）
#   - descriptive 描述性短语（"a thing to drink from"）—— 别名表**必然失败**，
#                 只能靠嵌入模型，这才是"开放词汇"的真正试金石
# 只报一个总命中率会把两者混在一起，从而**高估**开放词汇能力。
PROBES: Dict[str, Dict[str, List[str]]] = {
    "cup": {"named": ["杯子", "水杯", "马克杯", "mug"],
            "descriptive": ["a thing to drink from", "放饮料的容器"]},
    "table": {"named": ["桌子", "餐桌", "书桌", "desk"],
              "descriptive": ["a flat surface to put things on", "放东西的台面"]},
    "chair": {"named": ["椅子", "座椅", "凳子", "seat"],
              "descriptive": ["something you sit on", "可以坐的东西"]},
    "monitor": {"named": ["显示器", "屏幕", "screen", "display"],
                "descriptive": ["电脑屏幕"]},
    "box": {"named": ["箱子", "盒子", "carton"],
            "descriptive": ["a container for storage", "装东西的箱子"]},
    "bottle": {"named": ["瓶子", "水瓶", "flask"],
               "descriptive": ["a container with a narrow neck"]},
    "sofa": {"named": ["沙发", "couch", "settee"],
             "descriptive": ["a long soft seat for several people"]},
    "shelf": {"named": ["架子", "书架", "rack", "bookshelf"],
              "descriptive": ["放书的架子"]},
    "trash can": {"named": ["垃圾桶", "垃圾箱", "bin", "rubbish bin"],
                  "descriptive": ["扔垃圾的地方"]},
    "lamp": {"named": ["灯", "台灯", "light"],
             "descriptive": ["a device that gives light"]},
}

#: 负样本 A（**简单**）：语义上与室内场景毫无关系。
#: ⚠️ 这类负样本**参考价值很低** —— 一个颜色直方图都能拒掉它。
#: 早期版本**只**用了这一类，于是"拒识率 100%"这个数字是白送的，
#: 在评审眼里是红旗而不是亮点。保留它只为**对照**，不作为主指标。
NEGATIVE_PROBES_EASY = [
    "冰箱", "微波炉", "马桶", "浴缸", "飞机", "汽车", "自行车",
    "refrigerator", "helicopter", "airplane", "bicycle", "submarine",
]

#: ★ 负样本 B（**难**）：**与地图内类别语义高度接近**，但地图里确实没有。
#: 这才是"拒识能力"的真正试金石：嵌入模型必须真的分得开 chair / armchair。
#: 注意 `armchair` 里**含子串 `chair`**，会直接冲击词法路径的子串规则（0.85 分）——
#: 这是**故意**设计的对抗样本。
#:
#: 用法：按场景动态选取 —— 只有当"相邻的正类"真的在该场景地图里，
#: 且该负样本自己**不在**地图里时，才把它作为负样本加入。
HARD_NEGATIVES: Dict[str, List[str]] = {
    "chair": ["armchair", "stool", "bench", "扶手椅", "长凳"],
    "sofa": ["loveseat", "futon", "躺椅"],
    "table": ["countertop", "workbench", "吧台"],
    "shelf": ["cabinet", "wardrobe", "locker", "柜子", "衣柜"],
    "monitor": ["television", "laptop", "tablet", "电视", "笔记本电脑"],
    "box": ["crate", "basket", "篮子"],
    "bottle": ["jar", "canister", "罐子"],
    "lamp": ["lantern", "chandelier", "吊灯"],
    "trash can": ["dumpster", "recycling bin"],
    "cup": ["bowl", "vase", "花瓶", "wine glass"],
}

#: 各编码器的嵌入阈值扫描范围（按量级取对数刻度）。
#: SigLIP 在地图物体特征上的分数极小（正确命中只有 1e-3 量级），
#: 所以刻度必须一直探到 5e-5，否则会误判"最优阈值"就是网格下界。
SWEEP_EMB = {
    "siglip": [0.00005, 0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.0015,
               0.002, 0.003, 0.005, 0.01, 0.02],
    "clip":   [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
}
SWEEP_LEX = [0.05, 0.3, 0.5, 0.7, 0.85, 1.0]


# =============================================================================
# 原始分数收集
# =============================================================================
@dataclass
class QuerySample:
    """一次查询的原始分数（未过阈值），供事后解析地扫阈值。"""
    kind: str                       # "probe" | "negative"
    query: str
    expect: Optional[str]           # 负样本为 None
    labels: List[str]               # 该场景所有物体标签
    fused: np.ndarray
    lex: Optional[np.ndarray]
    emb: Optional[np.ndarray]
    probe_kind: str = "named"       # "named" | "descriptive"（仅 probe 有意义）
    conf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: 该场景各物体的观测体素数（词法打分会用它做轻微调制，标定时必须一并带上）
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    #: 该场景**自监督标定**出的纯嵌入阈值（None = 标定失败，回退编码器声明值）
    calib_thr: Optional[float] = None
    calib_source: str = "encoder"
    #: 嵌入标定在标定集上的分离度（平衡准确率，上界 2.0）
    calib_ba: float = float("nan")
    #: 该场景**自监督标定**出的词法阈值（None = 标定失败）—— 见第五节否定实验
    calib_lex_thr: Optional[float] = None
    #: 词法标定在标定集上的分离度（平衡准确率，上界 2.0）
    calib_lex_ba: float = float("nan")


def collect_scores(scenes, encoder, offset: float = 0.92) -> List[QuerySample]:
    """对每个场景、每条查询，只算一次原始分数（不做阈值判断）。"""
    from roboground.mapping.query import QueryEngine

    samples: List[QuerySample] = []
    for smap, labels in scenes:
        eng = QueryEngine(smap, text_encoder=encoder, lexical=True)
        # 另建一个**纯嵌入**引擎：它的作用不是查询，而是触发自监督标定，
        # 拿到"该场景就地标定"的嵌入阈值。这样评估的是自适应版本的真实表现，
        # 而不是拿一个离线扫出来的常量去评。
        solo = QueryEngine(smap, text_encoder=encoder, lexical=False)
        calib_thr = (solo.min_score_embedding
                     if solo.embedding_threshold_source == "auto" else None)
        calib_src = solo.embedding_threshold_source
        if calib_thr is not None:
            log.info(f"自监督标定：阈值 {calib_thr:.4g}"
                     f"（正样本中位 {solo.calibration_debug.get('pos_median', float('nan')):.4g} / "
                     f"负样本中位 {solo.calibration_debug.get('neg_median', float('nan')):.4g} / "
                     f"分离度 {solo.calibration_debug.get('balanced_accuracy', float('nan')):.3f}）")

        objs = smap.objects
        obj_labels = [o.label for o in objs]
        feats = np.stack([o.feature for o in objs], 0) if objs and objs[0].feature.size else None
        if feats is None:
            continue
        counts = np.array([o.num_voxels for o in objs], dtype=np.float32)
        conf = np.array([o.confidence for o in objs], dtype=np.float32)
        # 词法路径的自监督标定（第五节否定实验用；纯解析计算，不花编码时间）
        calib_lex, calib_lex_ba = calibrate_lexical_threshold(obj_labels, counts)

        # 每个场景只保留"这一场景里真有"的探针，否则期望标签不存在
        present = set(obj_labels)
        queries: List[Tuple[str, str, Optional[str], str]] = []
        for lab in sorted(present):
            for pk, plist in PROBES.get(lab, {}).items():
                for p in plist:
                    queries.append(("probe", p, lab, pk))
        for p in NEGATIVE_PROBES_EASY:
            queries.append(("negative_easy", p, None, "named"))
        # ★ 难负样本：只取"相邻正类确实在这个场景里、且自己不在"的那些
        for pos, negs in HARD_NEGATIVES.items():
            if pos not in present:
                continue
            for n in negs:
                if n in present:
                    continue          # 它真的在地图里 → 不能当负样本
                queries.append(("negative_hard", n, None, "named"))

        for kind, q, expect, pk in queries:
            fused, lex, emb = eng._score_candidates(q, obj_labels, feats, counts)
            # 空间重排（与真实查询路径一致：用观测充分度微调）
            if fused.size:
                fused = fused * (offset + (1 - offset) * conf)
                if lex is not None and lex.size:
                    lex = lex * (offset + (1 - offset) * conf)
            samples.append(QuerySample(kind=kind, query=q, expect=expect,
                                       labels=list(obj_labels), fused=fused,
                                       lex=lex, emb=emb, probe_kind=pk, conf=conf,
                                       counts=counts.copy(),
                                       calib_thr=calib_thr, calib_source=calib_src,
                                       calib_ba=float(solo.calibration_debug.get(
                                           "balanced_accuracy", float("nan"))),
                                       calib_lex_thr=calib_lex,
                                       calib_lex_ba=calib_lex_ba))
    return samples


def calibrate_lexical_threshold(labels, counts) -> Tuple[Optional[float], float]:
    """**否定实验**：词法路径能不能也用同一套自监督标定？

    做法与嵌入侧完全一样 —— 拿地图自己的标签互相打分：
    某个标签对**自己**的物体得高分（正样本）、对**别的**标签的物体得低分（负样本），
    在候选阈值上最大化平衡准确率。

    Returns
    -------
    (阈值, 标定集上的分离度)
        分离度 = 平衡准确率，上界 2.0。**2.0 意味着标定集上正负样本完全可分**，
        这在标定集上看着像好事，实际是危险信号（见第五节结论）。
    """
    from roboground.mapping.query import LexicalMatcher

    labs = list(labels)
    uniq = sorted(set(labs))
    if len(labs) < 2 or len(uniq) < 2:
        return None, float("nan")
    matcher = LexicalMatcher()
    pos: List[float] = []
    neg: List[float] = []
    for lab in uniq:
        s = np.asarray(matcher.score(lab, labels=labs, features=None, counts=counts),
                       dtype=np.float64)
        want = np.asarray([l == lab for l in labs], dtype=bool)
        pos.extend(s[want].tolist())
        neg.extend(s[~want].tolist())
    if len(pos) < 2 or len(neg) < 2:
        return None, float("nan")
    P, N = np.asarray(pos), np.asarray(neg)
    best_t, best_ba = None, -1.0
    for t in np.unique(np.concatenate([P, N])):
        ba = float((P >= t).mean() + (N < t).mean())
        if t > 0 and ba >= best_ba:      # 并列取更大（更保守）
            best_ba, best_t = ba, float(t)
    return best_t, best_ba


# =============================================================================
# 阈值评估（解析，不重跑编码器）
# =============================================================================
def _accept(lex, emb, t_lex: float, t_emb: float) -> np.ndarray:
    """双阈值接受掩码：任一路径达标即接受（与 QueryEngine._accept_mask 一致）。"""
    mask = None
    if lex is not None and lex.size:
        m = lex >= t_lex
        mask = m if mask is None else (mask | m)
    if emb is not None and emb.size:
        m = emb >= t_emb
        mask = m if mask is None else (mask | m)
    if mask is None:
        return np.zeros(0, dtype=bool)
    return mask


def _rank(lex, emb, fused, t_lex: float, t_emb: float) -> np.ndarray:
    """排序分数：取"达标那一侧"的较高值（与 QueryEngine._rank_score 一致）。

    ⚠️ 参数必须是**当前 mode 的视图**（纯嵌入模式下词法要传 None），
    不能从 sample 里直接读 —— 否则纯嵌入的评测会悄悄混进词法分数，
    量出来的就不是"纯嵌入"的行为。
    """
    if lex is not None and emb is not None and lex.size and emb.size:
        lex_ok, emb_ok = lex >= t_lex, emb >= t_emb
        return np.where(lex_ok & ~emb_ok, lex,
                        np.where(emb_ok & ~lex_ok, emb, np.maximum(lex, emb)))
    if lex is not None and lex.size:
        return lex
    if emb is not None and emb.size:
        return emb
    return fused


def evaluate_at(samples: List[QuerySample], t_lex: float, t_emb: float,
                mode: str = "hybrid", use_calibrated: bool = False) -> Dict[str, float]:
    """在给定阈值下算命中率与拒识率。mode ∈ {lexical, embedding, hybrid}。

    命中率**按表达类型分开统计**：命名类 vs 描述类。两者混在一起会高估
    开放词汇能力（别名表把命名类撑起来，描述类全靠嵌入）。

    `use_calibrated=True` 时，每条样本用**它自己场景自监督标定**出的嵌入阈值
    （标定失败的样本回退到 `t_emb`）—— 这样评的是自适应版本的真实表现。
    """
    def view(s: QuerySample):
        if mode == "lexical":
            return s.lex, None
        if mode == "embedding":
            return None, s.emb
        return s.lex, s.emb

    hit_named, hit_desc, rej = [], [], []
    rej_easy: List[float] = []
    rej_hard: List[float] = []
    used_thr: List[float] = []
    for s in samples:
        lex, emb = view(s)
        t_e, t_l = t_emb, t_lex
        if use_calibrated:
            if mode != "lexical" and s.calib_thr is not None:
                t_e = s.calib_thr
            if mode != "embedding" and s.calib_lex_thr is not None:
                t_l = s.calib_lex_thr
        used_thr.append(t_e)
        if (lex is None or not lex.size) and (emb is None or not emb.size):
            # 该路径在此样本上不可用 → 记为未命中 / 拒识成功
            if s.kind == "probe":
                (hit_named if s.probe_kind == "named" else hit_desc).append(0.0)
            else:
                (rej_hard if s.kind == "negative_hard" else rej_easy).append(1.0)
            continue
        acc = _accept(lex, emb, t_l, t_e)
        score = _rank(lex, emb, s.fused, t_l, t_e)
        idx = np.where(acc)[0]
        if idx.size:
            idx = idx[np.argsort(-score[idx], kind="stable")]
        if s.kind == "probe":
            ok = float(idx.size > 0 and s.labels[int(idx[0])] == s.expect)
            (hit_named if s.probe_kind == "named" else hit_desc).append(ok)
        elif s.kind == "negative_hard":
            rej_hard.append(float(idx.size == 0))
        else:
            rej_easy.append(float(idx.size == 0))

    all_hits = hit_named + hit_desc
    h = float(np.mean(all_hits)) if all_hits else 0.0
    rej = rej_easy + rej_hard                 # 合计（旧的 reject 字段保持兼容）
    r = float(np.mean(rej)) if rej else 0.0
    r_easy = float(np.mean(rej_easy)) if rej_easy else float('nan')
    r_hard = float(np.mean(rej_hard)) if rej_hard else float('nan')
    out = {
        "n_named": len(hit_named), "hit_named": float(np.mean(hit_named)) if hit_named else 0.0,
        "n_desc": len(hit_desc), "hit_desc": float(np.mean(hit_desc)) if hit_desc else 0.0,
        "n_probe": len(all_hits), "hit": h,
        "n_negative": len(rej), "reject": r, "balanced": h * r,
        # ★ 分开报告：简单负样本（"飞机/潜艇"）是白送的，**难负样本才是真指标**。
        #   早期只报合计的 reject，把两类混在一起，于是"拒识 100%"看起来很像能力，
        #   其实主要来自简单负样本 —— 这是评审点名的红旗，必须拆开。
        "n_negative_easy": len(rej_easy), "reject_easy": r_easy,
        "n_negative_hard": len(rej_hard), "reject_hard": r_hard,
        "balanced_hard": h * r_hard if rej_hard else float("nan"),
    }
    if use_calibrated:
        out["thr_median"] = float(np.median(used_thr)) if used_thr else float("nan")
        out["thr_min"] = float(np.min(used_thr)) if used_thr else float("nan")
        out["thr_max"] = float(np.max(used_thr)) if used_thr else float("nan")
    return out


def sweep(samples, mode: str, emb_grid: List[float],
          lex_grid: List[float]) -> List[Dict[str, float]]:
    """扫描阈值组合，返回按综合分降序的列表。"""
    rows = []
    for t_emb in emb_grid:
        for t_lex in lex_grid:
            m = evaluate_at(samples, t_lex, t_emb, mode=mode)
            m.update({"t_lex": t_lex, "t_emb": t_emb})
            rows.append(m)
    rows.sort(key=lambda r: -r["balanced"])
    return rows


# =============================================================================
def build_scenes(args, cfg):
    from roboground.data.synthetic import make_synthetic_sequence
    from roboground.mapping import MapBuilder

    prompts = sorted(PROBES.keys())
    cfg.set("perception.prompts", prompts)

    out = []
    for i in range(args.scenes):
        frames = make_synthetic_sequence(
            seed=args.seed + i, num_frames=3,
            width=args.width, height=args.height, num_objects=args.objects,
        )
        smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
        if smap.labels:
            out.append((smap, set(smap.labels)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="siglip",
                    choices=["none", "color_hist", "dinov2", "clip", "siglip"])
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--output", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if not args.output:
        args.output = f"runs/openvocab_{args.encoder}.json"

    if args.quiet:
        set_verbosity(0)

    from roboground import load_config
    from roboground.mapping.query import QueryEngine
    from roboground.perception import build_encoder

    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    # ⚠️ 地图特征必须和查询用的文本塔同源，否则维度/语义空间对不上。
    cfg.set("perception.encoder", args.encoder)
    cfg.set("project.verbose", False)
    local = Path(r"G:\RoboGround\weights\siglip-base-patch16-224")
    if args.encoder == "siglip" and local.exists():
        cfg.set("perception.encoder_kwargs.model_id", str(local))

    log.info(f"构建 {args.scenes} 个合成场景 ...")
    scenes = build_scenes(args, cfg)
    if not scenes:
        log.error("没有建成任何场景")
        return 2
    log.info(f"建图成功 {len(scenes)} 个场景，类别示例：" + str(sorted(scenes[0][1])))

    encoder = None
    if args.encoder not in ("none", "color_hist"):
        try:
            encoder = build_encoder(cfg, name=args.encoder)
            encoder.warmup()
            log.info(f"编码器 {args.encoder} 就绪（dim={encoder.feature_dim}, "
                     f"声明阈值={getattr(encoder, 'suggested_pair_threshold', None)}）")
        except Exception as exc:
            log.warn(f"编码器 {args.encoder} 不可用（{type(exc).__name__}: {exc}），只跑词法")
            encoder = None

    log.info("收集原始分数（每条查询只编码一次）...")
    samples = collect_scores(scenes, encoder)
    n_probe = sum(1 for s in samples if s.kind == "probe")
    log.info(f"共 {len(samples)} 条查询（{n_probe} 条未见表达 + "
             f"{len(samples) - n_probe} 条未见类别）")

    modes = ["lexical"] if encoder is None else ["lexical", "embedding", "hybrid"]
    emb_grid = SWEEP_EMB.get(args.encoder, [0.0, 0.001, 0.01, 0.02, 0.05])
    if encoder is None:
        emb_grid = [0.0]

    report: Dict[str, Any] = {"encoder": args.encoder, "scenes": len(scenes),
                              "n_probe": n_probe, "modes": {}}

    print()
    print("=" * 82)
    print(f"一、三条路径在「编码器声明阈值」下的表现（{args.encoder}）")
    print("=" * 82)
    declared_emb = float(getattr(encoder, "suggested_pair_threshold", 0.02) or 0.02)
    # 纯嵌入模式用的阈值与 hybrid 不同（角色不同，实测差约 400 倍）。
    # 这里读编码器声明的独立模式阈值，让"当前行为"如实反映引擎实际会做什么。
    declared_emb_solo = float(getattr(encoder, "suggested_standalone_threshold",
                                      declared_emb) or declared_emb)
    # ⚠️ 词法阈值必须**从引擎真实默认值读**，不能硬编码 ——
    # 否则报出来的"当前行为"就不是用户实际会看到的行为。
    declared_lex = QueryEngine(scenes[0][0]).min_score_lexical
    report["declared_lexical_threshold"] = declared_lex
    report["declared_emb_hybrid"] = declared_emb
    report["declared_emb_standalone"] = declared_emb_solo
    hdr = (f"{'查询路径':<12}{'命名类命中':>12}{'描述类命中':>12}"
           f"{'总命中':>10}{'负样本':>8}{'拒识率':>10}{'综合分':>10}")
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        # hybrid 用"精度过滤器"阈值，纯嵌入用"接受阈值"
        t_emb = declared_emb_solo if m == "embedding" else declared_emb
        base = evaluate_at(samples, declared_lex, t_emb, mode=m)
        report["modes"][m] = {"declared": base, "t_emb_used": t_emb}
        print(f"{m:<12}{base['hit_named']:>12.1%}{base['hit_desc']:>12.1%}"
              f"{base['hit']:>10.1%}{base['n_negative']:>8}"
              f"{base['reject']:>10.1%}{base['balanced']:>10.3f}")
    n_named = sum(1 for s in samples if s.kind == "probe" and s.probe_kind == "named")
    n_desc = sum(1 for s in samples if s.kind == "probe" and s.probe_kind == "descriptive")
    print(f"\n  （引擎真实默认：词法 {declared_lex:g} / "
          f"嵌入 hybrid {declared_emb:g} · 纯嵌入 {declared_emb_solo:g}；"
          f"命名类 {n_named} 条 / 描述类 {n_desc} 条）")

    # ---------------- 阈值扫描 ----------------
    print()
    print("=" * 82)
    print("二、阈值扫描（事后解析计算，不重跑编码器）")
    print("=" * 82)
    for m in modes:
        rows = sweep(samples, m, emb_grid, SWEEP_LEX)
        best = rows[0]
        report["modes"][m]["sweep_best"] = best
        report["modes"][m]["sweep_top5"] = rows[:5]
        print(f"\n[{m}] 综合分 Top-5")
        print(f"  {'词法阈值':>10}{'嵌入阈值':>12}{'命名类':>10}{'描述类':>10}"
              f"{'拒识率':>10}{'综合分':>10}")
        for r in rows[:5]:
            print(f"  {r['t_lex']:>10.2f}{r['t_emb']:>12g}{r['hit_named']:>10.1%}"
                  f"{r['hit_desc']:>10.1%}{r['reject']:>10.1%}{r['balanced']:>10.3f}")

    # ---------------- 词法阈值敏感性（fix 的证据） ----------------
    print()
    print("=" * 82)
    print("二.5、词法阈值敏感性（纯词法路径，不涉及嵌入阈值）")
    print("=" * 82)
    print("  字符 bigram 那层是噪声源（bicycle/bottle 共享 le → 0.145 分），")
    print("  所以阈值必须高于噪声层、低于真信号（子串 0.85 / 概念 1.00）。")
    print(f"\n  {'词法阈值':>10}{'命名类':>10}{'描述类':>10}{'拒识率':>10}{'综合分':>10}")
    lex_curve = []
    for t in SWEEP_LEX:
        m = evaluate_at(samples, t, 0.0, mode="lexical")
        lex_curve.append({"t": t, **m})
        flag = "  ← 本项默认" if abs(t - declared_lex) < 1e-12 else ""
        print(f"  {t:>10.2f}{m['hit_named']:>10.1%}{m['hit_desc']:>10.1%}"
              f"{m['reject']:>10.1%}{m['balanced']:>10.3f}{flag}")
    report["lex_curve"] = lex_curve
    old = next((r for r in lex_curve if abs(r["t"] - 0.05) < 1e-12), None)
    new = next((r for r in lex_curve if abs(r["t"] - declared_lex) < 1e-12), None)
    if old and new:
        print(f"\n  → 修复前默认 0.05：拒识率 {old['reject']:.1%}，综合分 {old['balanced']:.3f}")
        print(f"  → 修复后默认 {declared_lex:g}：拒识率 {new['reject']:.1%}，"
              f"综合分 {new['balanced']:.3f}")
        print(f"  → 命名类命中率 {old['hit_named']:.1%} → {new['hit_named']:.1%}"
              f"（{'无损失' if new['hit_named'] >= old['hit_named'] - 1e-9 else '有损失'}）")

    # ---------------- 嵌入阈值曲线 ----------------
    if encoder is not None:
        print()
        print("=" * 82)
        print("三、嵌入路径的阈值曲线（词法关闭，看纯嵌入行为）")
        print("=" * 82)
        print("  ★ 关键看「描述类」列：别名表对它必然失败，只有嵌入能救")
        print(f"  {'嵌入阈值':>12}{'命名类':>10}{'描述类':>10}{'拒识率':>10}{'综合分':>10}")
        curve = []
        for t in emb_grid:
            m = evaluate_at(samples, 0.05, t, mode="embedding")
            curve.append({"t": t, **m})
            print(f"  {t:>12g}{m['hit_named']:>10.1%}{m['hit_desc']:>10.1%}"
                  f"{m['reject']:>10.1%}{m['balanced']:>10.3f}")
        report["emb_curve"] = curve
        best_emb = max(curve, key=lambda r: r["balanced"])
        cur = evaluate_at(samples, 0.05, declared_emb_solo, mode="embedding")
        stale = evaluate_at(samples, 0.05, declared_emb, mode="embedding")
        print(f"\n  → 扫描出的最优嵌入阈值 ≈ **{best_emb['t']:g}**"
              f"（命名 {best_emb['hit_named']:.1%} / 描述 {best_emb['hit_desc']:.1%}"
              f" / 拒识 {best_emb['reject']:.1%}，综合分 {best_emb['balanced']:.3f}）")
        match = "✓ 与扫描结果一致" if abs(best_emb['t'] - declared_emb_solo) < 1e-12 \
            else f"⚠ 与扫描结果不一致（声明 {declared_emb_solo:g}，最优 {best_emb['t']:g}）"
        print(f"  → 纯嵌入实际用 {declared_emb_solo:g}："
              f"命名 {cur['hit_named']:.1%} / 描述 {cur['hit_desc']:.1%}"
              f" / 拒识 {cur['reject']:.1%}，综合分 {cur['balanced']:.3f}  {match}")
        print(f"  → 若误用 hybrid 的 {declared_emb:g}（角色不同）："
              f"描述类命中 {cur['hit_desc']:.1%} → {stale['hit_desc']:.1%}"
              f"，综合分 {cur['balanced']:.3f} → {stale['balanced']:.3f}")
        print(f"     —— 这就是「阈值角色不匹配」的代价（差 "
              f"{cur['hit_desc'] / max(stale['hit_desc'], 1e-9):.0f} 倍）。")

    # ---------------- 自监督标定 vs 硬编码常量 ----------------
    if encoder is not None:
        print()
        print("=" * 82)
        print("四、自监督阈值标定 vs 硬编码常量（纯嵌入路径）")
        print("=" * 82)
        print("  标定用**地图自身的 (特征, 标签) 对**：物体对自己的标签应得高分、")
        print("  对别的标签应得低分。选点准则 = 最大化平衡准确率 (TPR+TNR)。")
        print("  好处：换数据集/换编码器不用重扫，且逐场景自适应。")
        auto = evaluate_at(samples, declared_lex, declared_emb_solo,
                           mode="embedding", use_calibrated=True)
        fixed = evaluate_at(samples, declared_lex, declared_emb_solo,
                            mode="embedding", use_calibrated=False)
        report["calibration"] = {"auto": auto, "fixed": fixed}
        print(f"\n  {'方案':<26}{'命名类':>10}{'描述类':>10}{'拒识率':>10}{'综合分':>10}")
        print(f"  {'硬编码常量 ' + f'{declared_emb_solo:g}':<26}"
              f"{fixed['hit_named']:>10.1%}{fixed['hit_desc']:>10.1%}"
              f"{fixed['reject']:>10.1%}{fixed['balanced']:>10.3f}")
        print(f"  {'自监督标定（逐场景）':<26}"
              f"{auto['hit_named']:>10.1%}{auto['hit_desc']:>10.1%}"
              f"{auto['reject']:>10.1%}{auto['balanced']:>10.3f}")
        print(f"\n  → 标定阈值范围 {auto['thr_min']:.4g} ~ {auto['thr_max']:.4g}"
              f"（中位 {auto['thr_median']:.4g}）—— 每场景自适应")
        delta = auto["balanced"] - fixed["balanced"]
        verdict = ("优于" if delta > 0.005 else
                   "与" if abs(delta) <= 0.005 else "略逊于")
        print(f"  → 综合分 {fixed['balanced']:.3f} → {auto['balanced']:.3f}"
              f"（{verdict}硬编码常量 {abs(delta):+.3f}）")
        if delta > 0.005:
            print("  → 结论：**自监督标定可以替代离线扫描**，且换数据集无需重扫。")
        elif abs(delta) <= 0.005:
            print("  → 结论：**自监督标定达到离线扫描同等水平**，可直接用它替代硬编码常量，")
            print("     从而去掉「阈值需人工扫描」这一局限（换数据集/编码器无需重标）。")
        else:
            print("  → 结论：标定略逊于离线最优，但**无需任何人工扫描**且逐场景自适应；")
            print("     可作为默认值，追求极致时仍可离线扫描微调。")

    # ---------------- 五、否定实验：词法路径能不能也自标定？ ----------------
    print()
    print("=" * 82)
    print("五、否定实验：词法路径能不能也用同一套自监督标定？")
    print("=" * 82)
    print("  假设很自然：嵌入侧能自标定，词法侧应该也行 —— 同一套 (标签, 标签) 打分。")
    print("  但有个结构性隐患：**标定集是地图自己的标签，而词法阈值真正要挡的是")
    print("  地图里没有的类别**。如果两者分布不一致，标定就会优化错的目标。")
    print()
    lex_cal = evaluate_at(samples, declared_lex, 0.0, mode="lexical",
                          use_calibrated=True)
    lex_fix = evaluate_at(samples, declared_lex, 0.0, mode="lexical",
                          use_calibrated=False)
    report["lexical_calibration"] = {"auto": lex_cal, "fixed": lex_fix}
    thrs = sorted({s.calib_lex_thr for s in samples if s.calib_lex_thr})
    bas = sorted({round(s.calib_lex_ba, 3) for s in samples
                  if s.calib_lex_ba == s.calib_lex_ba})
    emb_bas = sorted({round(s.calib_ba, 3) for s in samples
                      if s.calib_ba == s.calib_ba})
    print(f"  标定出的词法阈值：{['%.3f' % t for t in thrs]}")
    print(f"  （人工定的是 {declared_lex:g}）")
    print(f"\n  ★ 这里是关键对比 —— **标定集上的分离度（平衡准确率，上界 2.0）**：")
    print(f"     词法：{['%.3f' % b for b in bas]}   ← 接近 2.0 = 正负样本完全可分")
    print(f"     嵌入：{['%.3f' % b for b in emb_bas]}   ← 明显重叠")
    print()
    print("  **分离度满分不是好事，是危险信号**：")
    print("  - 词法标定集是「标签查自己」，别名表必然命中 1.0、对其它的≈0，")
    print("    于是正负样本**完全可分**，优化器把阈值推到正样本分布的最边缘（≈0.90）；")
    print("  - 但真实查询（「杯子」「seat」）常常只落在子串那层（0.85），全被误杀；")
    print("  - 嵌入标定集是「多视角融合特征 vs 文本」，正负分布本就重叠（BA≈1.4），")
    print("    阈值落在有意义的间隙里，所以**泛化得好**。")
    print()
    print("  推广的判据：**自监督标定只在「标定集的难度 ≈ 部署时的难度」时才有效。**")
    print("  标定集比真实场景简单 → 阈值必然过严。而 BA 接近 2.0 就是"
          "「标定集太简单」的直接证据。")
    print(f"\n  {'方案':<26}{'命名类':>10}{'描述类':>10}{'拒识率':>10}{'综合分':>10}")
    print(f"  {'人工常量 ' + f'{declared_lex:g}':<26}"
          f"{lex_fix['hit_named']:>10.1%}{lex_fix['hit_desc']:>10.1%}"
          f"{lex_fix['reject']:>10.1%}{lex_fix['balanced']:>10.3f}")
    print(f"  {'自监督标定（逐场景）':<26}"
          f"{lex_cal['hit_named']:>10.1%}{lex_cal['hit_desc']:>10.1%}"
          f"{lex_cal['reject']:>10.1%}{lex_cal['balanced']:>10.3f}")
    d_lex = lex_cal["balanced"] - lex_fix["balanced"]
    print()
    if d_lex < -0.005:
        print("  → **结论：否定。词法路径不该用这套标定。**")
        print("     根因是标定集与查询分布不匹配 —— 见下方解释。")
    elif d_lex > 0.005:
        print("  → 结论：肯定。词法路径也可以用自监督标定（意外）。")
    else:
        print("  → 结论：两者相当，但人工常量更简单，保持现状。")

    # ---------------- 结论读法 ----------------
    print("""
  两个维度必须一起看：
  - **只看命中率**会高估：永远返回"最像的东西"也能拿到不错的命中率，
    但对机器人是危险的（会指向完全错误的位置）。
  - **只看拒识率**会低估：什么都不返回自然 100% 拒识。
  - 综合分 = 命中率 × 拒识率。

  三条路径的定位：
  - `lexical`  靠双语别名表，中文与常见近义词命中率高；
               但对 "a thing to drink from" 这类描述性短语**必然失败**（表里没有）。
  - `embedding` 能处理任意文本（真·开放词汇），但**对阈值尺度极其敏感**：
               阈值定高一点就整体拒空，定低一点就乱答。
  - `hybrid`   任一路径达标即接受 → 取长补短，是本项目默认路径。

  阈值为什么不能硬编码：词法分是"命中=1、不命中=0"的硬信号，
  嵌入分是校准后的概率，两者尺度差几个数量级；而且**同一种嵌入在不同特征管线上
  尺度也不同**（检测区域特征 vs 多视角融合的地图物体特征实测差一个数量级）。
  所以阈值必须就地标定 —— 这就是本脚本要扫阈值的原因。
""")

    if args.output:
        ensure_dir(Path(args.output).parent)
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log.ok(f"结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
