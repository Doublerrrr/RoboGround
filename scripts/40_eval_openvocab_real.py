# -*- coding: utf-8 -*-
"""40 · 在**真实数据**上做开放词汇检索评测（难负样本 + 真实候选池）。

为什么必须新写一个，而不是继续用 `scripts/16`
==========================================
`scripts/16` 用的是**合成场景**，而合成场景的类别池是**写死的 10 类**
（`SyntheticRoom.random` 的 `catalog`），默认只用 6 个物体。这带来两个致命问题：

1. **候选池太小**：在 6 个候选里做 top-1，"命中率 100%"几乎不构成证据；
2. **负样本太简单**：原来的负样本是「冰箱/飞机/潜艇」这类与室内毫无关系的词，
   一个颜色直方图都能拒掉 —— 所以"拒识 100%"是白送的。

评审的原话是「满分指标在资深面试官眼里是红旗」，指的就是这两点。

本脚本的做法
==========
把评测搬到 **Stanford 2D-3D-S 真实数据**上：

| 维度 | scripts/16（合成） | 本脚本（真实） |
|---|---|---|
| 图像 | 合成渲染 | 真实采集（官方重建渲染） |
| 候选池 | **固定 6~10 个** | **该房间真实物体数（实测 11~152，中位 35）** |
| 负样本 | 飞机/潜艇 | **难例**：armchair ↔ chair、cabinet ↔ shelf 等语义近邻 |
| 特征 | 颜色直方图或 SigLIP | **SigLIP 真特征**（物体图像区域编码） |

评测协议（写清楚，便于别人复现或反驳）
==================================
· **候选池**：一个采集点里**可见的** GT 物体（每个物体一个候选），池大小逐点报告；
· **正样本**：池内每个物体的**类别名**（英文）作为查询 → 期望 top-1 命中它自己；
· **描述型查询**：类别名之外的描述短语（如 "something you sit on"）→ 同样期望命中；
· **难负样本**：该 **不在** 池内、但与池内某类语义相邻的词（armchair/cabinet/…）→
  期望**没有任何候选被接受**；
· **简单负样本**（飞机/潜艇）**单独报告**，只作对照，不与难负样本混在一起；
· **指标**：top-1 命中率、top-5 命中率、候选池大小、以及**分难度的拒识率**。

用法::

    python scripts/40_eval_openvocab_real.py --points 8 --max-views 24
    python scripts/40_eval_openvocab_real.py --points 20 --encoder siglip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module  # noqa: E402

_v16 = import_module("16_eval_openvocab_query")
HARD_NEGATIVES = _v16.HARD_NEGATIVES
NEGATIVE_PROBES_EASY = _v16.NEGATIVE_PROBES_EASY
PROBES = _v16.PROBES

from roboground.data.pano_scene import load_scene, select_location, visible_gt  # noqa: E402
from roboground.data.stanford2d3d import list_locations  # noqa: E402

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")

#: 描述型查询（与类别名不同、别名表覆盖不到）—— 这才是开放词汇的试金石
DESCRIPTIVE: Dict[str, List[str]] = {
    "chair": ["something you sit on"],
    "table": ["a flat surface to put things on"],
    "door": ["a thing you walk through to enter a room"],
    "bookcase": ["a piece of furniture holding many books"],
    "window": ["an opening in the wall letting in daylight"],
    "sofa": ["a long soft seat for several people"],
    "board": ["a flat panel on the wall for writing"],
    "column": ["a tall vertical support pillar"],
    "beam": ["a horizontal structure across the ceiling"],
}


def crop_patch(img: np.ndarray, uv: Tuple[int, int, int, int],
               pad: int = 4) -> Optional[np.ndarray]:
    """按投影框裁一块图（带边界裁剪）。框太小就返回 None。"""
    H, W = img.shape[:2]
    u0, v0, u1, v1 = uv
    u0 = max(0, min(int(u0) - pad, W - 1)); u1 = max(0, min(int(u1) + pad, W - 1))
    v0 = max(0, min(int(v0) - pad, H - 1)); v1 = max(0, min(int(v1) + pad, H - 1))
    if u1 <= u0 or v1 <= v0:
        return None
    if (u1 - u0) < 8 or (v1 - v0) < 8:
        return None
    return img[v0:v1, u0:u1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--max-views", type=int, default=24)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--min-pool", type=int, default=5, help="候选池小于此值就跳过该采集点")
    ap.add_argument("--out", default="runs/40_eval_openvocab_real.json")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2

    # ---- 加载 SigLIP（真视觉语言特征）----
    # ⚠️ 必须显式指向**本地权重目录**：默认 model_id 是 `google/siglip-base-patch16-224`，
    #    会去 HF 缓存找，而本机缓存不完整（缺 spiece.model）→ 直接加载失败。
    #    这里跟随 `scripts/16` 的做法用 `build_encoder` + 本地路径。
    from roboground.config import load_config
    from roboground.perception import build_encoder

    cfg = load_config()
    local = Path(r"G:\RoboGround\weights\siglip-base-patch16-224")
    if local.exists():
        cfg.set("perception.encoder_kwargs.model_id", str(local))
    enc = build_encoder(cfg, name="siglip")
    try:
        # 该编码器是**懒加载**的（没有 available() 方法）；用它跑一次文本编码
        # 来触发加载并校验权重真的可用。失败就明确退出 —— 本评测**不做降级**，
        # 因为要测的就是"真视觉语言特征"的能力，降级成颜色直方图就没意义了。
        _ = enc.encode_text(["warmup"])
    except Exception as exc:                              # noqa: BLE001
        print(f"[FAIL] SigLIP 不可用（{type(exc).__name__}: {exc}）")
        return 2
    print(f"编码器：{type(enc).__name__}，特征维度 {enc.feature_dim}")

    locs = sorted(list_locations(args.root), key=lambda l: -len(l.frame_ids))
    locs = [l for l in locs if len(l.frame_ids) >= 10]
    idx = np.linspace(0, len(locs) - 1, num=min(args.points, len(locs))).round().astype(int)
    locs = [locs[int(i)] for i in np.unique(idx)]

    all_rows: List[Dict[str, Any]] = []
    for loc in locs:
        sc = load_scene(args.root, loc, width=args.width, height=args.height,
                        max_frames=args.max_views, max_depth=args.max_range)
        vis = visible_gt(sc, max_range_m=args.max_range)
        if len(vis) < args.min_pool:
            print(f"  [skip] {loc.room}: 可见物体只有 {len(vis)} 个")
            continue

        img = sc.panorama.rgb
        labels: List[str] = []
        feats: List[np.ndarray] = []
        for o in vis:
            patch = crop_patch(img, o["uv"])
            if patch is None:
                continue
            # ⚠️ `encode_image` 收**单张**图（不是列表）—— 与本项目其它编码器的
            # 约定不同，第一版按列表传会直接抛错。
            f = np.asarray(enc.encode_image(patch), dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(f))
            if n < 1e-8:
                continue
            labels.append(str(o["label"]))
            feats.append(f / n)
        if len(labels) < args.min_pool:
            print(f"  [skip] {loc.room}: 可编码物体只有 {len(labels)} 个")
            continue
        F = np.stack(feats)                       # (N, D) 归一化后的候选特征
        pool = len(labels)
        present = set(labels)

        def rank_of(text: str) -> np.ndarray:
            t = np.asarray(enc.encode_text([text]), dtype=np.float32).reshape(-1)
            t = t / max(float(np.linalg.norm(t)), 1e-8)
            return F @ t                          # 余弦相似度

        # ---- 正样本：类别名（池内每个类别各测一次）----
        hit1 = hit5 = 0
        n_pos = 0
        for i, lab in enumerate(labels):
            sc_ = rank_of(lab)
            order = np.argsort(-sc_, kind="stable")
            n_pos += 1
            hit1 += int(labels[int(order[0])] == lab)
            hit5 += int(lab in [labels[int(j)] for j in order[:5]])

        # ---- 描述型查询（只测池内有的类）----
        d_hit1 = d_n = 0
        for lab, phrases in DESCRIPTIVE.items():
            if lab not in present:
                continue
            for ph in phrases:
                sc_ = rank_of(ph)
                order = np.argsort(-sc_, kind="stable")
                d_n += 1
                d_hit1 += int(labels[int(order[0])] == lab)

        # ---- 难 / 易负样本：用同一套"接受阈值"判是否误接受 ----
        # 阈值取"正样本余弦分布的下分位数"是自监督的；这里为了**纯粹比较**，
        # 直接报告**最大相似度**与"是否超过池内正样本的最小相似度"两种判据。
        pos_sims = []
        for i, lab in enumerate(labels):
            pos_sims.append(float(rank_of(lab)[i]))
        pos_min = float(np.min(pos_sims))

        def neg_accept(text: str) -> float:
            """返回该负样本在池上的**最大余弦相似度**（越高越容易被误接受）。"""
            return float(np.max(rank_of(text)))

        easy_sims = [neg_accept(t) for t in NEGATIVE_PROBES_EASY]
        hard_terms: List[str] = []
        for pos, negs in HARD_NEGATIVES.items():
            if pos not in present:
                continue
            hard_terms += [n for n in negs if n not in present]
        hard_sims = [neg_accept(t) for t in hard_terms]

        # ---- 判别力指标：AUC + 最佳阈值的平衡准确率 ----
        # ⚠️ 第一版用的是"所有负样本都低于所有正样本"这种**完全可分**判据，
        #    那太严了（真实数据上几乎必然为 False，因为正样本里也有难例）。
        #    标准做法是报 **AUC** 与**最佳阈值下的平衡准确率**。
        def auc_ba(pos: List[float], neg: List[float]) -> Tuple[Optional[float], Optional[float]]:
            if not pos or not neg:
                return None, None
            a = np.asarray(pos, dtype=np.float64)[:, None]
            b = np.asarray(neg, dtype=np.float64)[None, :]
            auc = float((a > b).mean() + 0.5 * (a == b).mean())
            cands = np.unique(np.concatenate([np.asarray(pos), np.asarray(neg)]))
            acc_pos = np.array([(np.asarray(pos) > t).mean() for t in cands])
            acc_neg = np.array([(np.asarray(neg) <= t).mean() for t in cands])
            return auc, float(np.max((acc_pos + acc_neg) / 2.0))

        auc_easy, ba_easy = auc_ba(pos_sims, easy_sims)
        auc_hard, ba_hard = auc_ba(pos_sims, hard_sims)

        row = {
            "room": sc.room, "uuid": sc.uuid[:12], "pool_size": pool,
            "auc_easy": auc_easy, "ba_easy": ba_easy,
            "auc_hard": auc_hard, "ba_hard": ba_hard,
            "n_pos_probe": n_pos,
            "top1": hit1 / max(n_pos, 1), "top5": hit5 / max(n_pos, 1),
            "desc_n": d_n, "desc_top1": (d_hit1 / d_n) if d_n else None,
            "pos_sim_min": pos_min, "pos_sim_median": float(np.median(pos_sims)),
            "n_easy_neg": len(easy_sims),
            "easy_max_sim": max(easy_sims) if easy_sims else None,
            "n_hard_neg": len(hard_sims), "hard_terms": hard_terms[:12],
            "hard_max_sim": max(hard_sims) if hard_sims else None,
            # 判据：负样本最高分 < 正样本最低分 ⇒ 存在一个阈值能把它们分开
            "easy_separable": (max(easy_sims) < pos_min) if easy_sims else None,
            "hard_separable": (max(hard_sims) < pos_min) if hard_sims else None,
        }
        all_rows.append(row)
        d_txt = ("n/a" if row["desc_top1"] is None
                 else f"{row['desc_top1']*100:4.1f}%")
        ah = "n/a" if row["auc_hard"] is None else f"{row['auc_hard']:.3f}"
        print(f"  {sc.room:<18} 池 {pool:>3}  top1 {row['top1']*100:5.1f}%  "
              f"描述 {d_txt:>5}  AUC(易) {row['auc_easy']:.3f}  AUC(难) {ah:>5}")

    if not all_rows:
        print("[FAIL] 没有可用采集点")
        return 2

    # ---- 汇总 ----
    pools = [r["pool_size"] for r in all_rows]
    print("\n" + "=" * 78)
    print("汇总（真实数据 + 真 SigLIP 特征）")
    print("=" * 78)
    top1 = float(np.mean([r["top1"] for r in all_rows]))
    top5 = float(np.mean([r["top5"] for r in all_rows]))
    d_rows = [r for r in all_rows if r["desc_top1"] is not None]
    d1 = float(np.mean([r["desc_top1"] for r in d_rows])) if d_rows else float("nan")
    hard_sep = [r["hard_separable"] for r in all_rows if r["hard_separable"] is not None]
    easy_sep = [r["easy_separable"] for r in all_rows if r["easy_separable"] is not None]
    n_hard = sum(r["n_hard_neg"] for r in all_rows)
    print(f"  采集点 {len(all_rows)} 个；候选池大小 min {min(pools)} / 中位 "
          f"{int(np.median(pools))} / max {max(pools)}")
    print(f"  类别名 top-1 命中率：{top1*100:.1f}%      top-5：{top5*100:.1f}%")
    print(f"  描述型 top-1 命中率：{d1*100:.1f}%（{len(d_rows)} 个采集点有描述探针）")
    def _mean(key):
        v = [r[key] for r in all_rows if r.get(key) is not None]
        return float(np.mean(v)) if v else float("nan")
    print(f"  ★ 判别力 AUC（正样本 vs 负样本）：简单负样本 {_mean('auc_easy'):.3f}"
          f" / 难负样本 {_mean('auc_hard'):.3f}   （0.5 = 纯随机）")
    print(f"  ★ 最佳阈值平衡准确率：简单负样本 {_mean('ba_easy'):.3f}"
          f" / 难负样本 {_mean('ba_hard'):.3f}   （0.5 = 无判别力）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"points": len(all_rows), "max_views": args.max_views,
                   "width": args.width, "height": args.height,
                   "max_range": args.max_range, "root": str(args.root),
                   "encoder": "siglip"},
        "summary": {"pool_min": min(pools), "pool_median": int(np.median(pools)),
                    "pool_max": max(pools),
                    "top1": top1, "top5": top5, "desc_top1": d1,
                    "auc_easy": _mean("auc_easy"), "auc_hard": _mean("auc_hard"),
                    "ba_easy": _mean("ba_easy"), "ba_hard": _mean("ba_hard"),
                    "n_hard_neg_terms": n_hard},
        "per_point": all_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物：{out}")
    print("\n判据（诚实版）")
    print("  · 候选池必须明显大于合成评测的 6~10，否则指标仍然不可信")
    print("  · ★ 难负样本可分离率才是真正的『拒识能力』；简单负样本只作对照")
    print("  · 描述型命中率才反映嵌入模型的开放词汇泛化（类别名可能被词法覆盖）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
