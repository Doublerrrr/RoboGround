# -*- coding: utf-8 -*-
"""3D 特征场到底有没有"开放词汇"能力？—— `color_hist` vs 真视觉语言特征。

为什么必须单独做这个实验（批评 #2）
=================================
项目的叙事里写着"体素级 3D 特征场**存特征而非类别 id**，所以支持开放词汇"。
但**离线默认的编码器是 `color_hist`**（72 维 HSV 直方图），
它根本没有文本路径 —— 也就是说：**默认链路里不存在开放词汇能力**，
查询能工作只是因为物体自带标签、走的是词法匹配。

而"装了 SigLIP 就开放词汇了"同样不能靠嘴说。`scripts/40` 已经在**编码器侧**
量过判别力（真实数据上难负样本 AUC 0.352），但那是"拿全景图上裁的 patch 比"，
不是**特征场**。这个脚本量的是端到端的那条路：

    GT 注入检测 → 用 SigLIP 编码每个物体的 patch → 融进体素场 →
    物体级特征 → `SemanticMap.query_text()` → 排序结果

两条臂，**同一批检测、同一张全景、同一个下游**，只换编码器：

| 臂 | 特征 | 文本侧 | 说明 |
|---|---|---|---|
| `color_hist` | 72 维 HSV 直方图 | 无（`supports_text=False`） | **当前默认**；只能走词法匹配 |
| `siglip` | 768 维 SigLIP 图像嵌入 | SigLIP 文本塔 | 真正的开放词汇路径 |

指标（写清楚口径，避免被误读）
============================
· **候选池** = 该采集点地图里的全部物体（池大小逐点报告）；
· **类别名查询**：对每个物体，用它的类别名去查 → 看 top-1 / top-5 里的物体
  **标签是否与查询同类**（多实例时不该只认某一个实例，所以按"同类"判）；
· **描述型查询**：别名表覆盖不到的短语（"something you sit on"）→ 这才是试金石；
· **难负样本**：地图里**没有**、但与池内某类语义相邻的词 → 期望**零接受**；
· **AUC**：正样本分数 vs 难负样本分数（0.5 = 纯随机）。

跑法：
    python scripts/44_eval_feature_field.py --points 3
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
_v40 = import_module("40_eval_openvocab_real")

HARD_NEGATIVES: Dict[str, List[str]] = _v16.HARD_NEGATIVES
NEGATIVE_PROBES_EASY: List[str] = _v16.NEGATIVE_PROBES_EASY
DESCRIPTIVE: Dict[str, List[str]] = _v40.DESCRIPTIVE
crop_patch = _v40.crop_patch

from roboground.config import load_config  # noqa: E402
from roboground.data.pano_scene import load_scene, visible_gt  # noqa: E402
from roboground.data.stanford2d3d import list_locations  # noqa: E402
from roboground.types import Detection2D  # noqa: E402

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")
LOCAL_SIGLIP = Path(r"G:\RoboGround\weights\siglip-base-patch16-224")
#: 命中判定：返回物体与查询类别的任一实例中心距离 ≤ 该值
HIT_DIST_M = 2.0


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def auc_ba(pos: List[float], neg: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """AUC 与最佳阈值下的平衡准确率（0.5 = 无判别力）。"""
    if not pos or not neg:
        return None, None
    a = np.asarray(pos, dtype=np.float64)[:, None]
    b = np.asarray(neg, dtype=np.float64)[None, :]
    auc = float((a > b).mean() + 0.5 * (a == b).mean())
    cands = np.unique(np.concatenate([np.asarray(pos), np.asarray(neg)]))
    ap = np.array([(np.asarray(pos) > t).mean() for t in cands])
    an = np.array([(np.asarray(neg) <= t).mean() for t in cands])
    return auc, float(np.max((ap + an) / 2.0))


def build_map(cfg, root: Path, loc, args, *, encoder_name: str):
    """建一张地图，并把每个 GT 物体的 **patch 特征**注入检测里。

    ⚠️ 必须自己注入特征：`detector=stub` 时流水线不会真的跑编码器，
    而我们注入检测正是为了绕过检测器。特征这一环不能一起绕过，
    否则测的就成了"没有特征的体素场"。
    """
    from roboground.mapping import MapBuilder
    from roboground.perception import build_encoder

    scene = load_scene(root, loc, width=args.width, height=args.height,
                       max_frames=args.max_views, max_depth=args.max_range,
                       resize=(args.resize, args.resize))
    vis = visible_gt(scene, max_range_m=args.max_range)
    if not vis:
        return None, None, None

    enc = None
    c = load_config()
    if LOCAL_SIGLIP.exists():
        c.set("perception.encoder_kwargs.model_id", str(LOCAL_SIGLIP))
    enc = build_encoder(c, name=encoder_name)

    from roboground.perception import PerceptionPipeline, build_segmenter

    # 待注入的检测（只有标签与投影框；**特征由流水线自己算**）
    dets: List[Detection2D] = []
    for it in vis:
        parts = it.get("uv_parts") or [it["uv"]]
        # 跨接缝的物体会被拆成两段：取面积大的那一段（覆盖更完整）
        part = max(parts, key=lambda p: max(0, p[2] - p[0]) * max(0, p[3] - p[1]))
        dets.append(Detection2D(label=str(it["label"]), score=1.0,
                                bbox=np.array(part, dtype=np.float64),
                                prompt=str(it["label"])))

    # ★ 设计方案：**用桩检测器注入 GT 框，但让真正的流水线去做特征编码**。
    #
    #   踩过两次的坑（都写下来）：
    #   ① 自己 `encode_image(patch)` 不行 —— `ColorHistogramEncoder` 只实现了
    #      **区域编码** `encode_regions`，`encode_image` 直接 `NotImplementedError`；
    #   ② 把 `pipeline.run` 整个替换成 lambda 也不行 —— 那样编码器根本不会跑，
    #      地图里的特征全是 0，指标却照常打出来（"看起来像模型不行"）。
    #
    #   所以只替换**检测**这一环，后面的分割 + 编码仍走项目自己的实现，
    #   两条臂的差别就严格只有"编码器是谁"。
    class _InjectedDetector:
        name = "injected"
        supports_text = False

        def detect(self, frame, prompts=None):               # noqa: D102
            return [Detection2D(label=d.label, score=d.score,
                                bbox=np.array(d.bbox, dtype=np.float64),
                                prompt=d.prompt) for d in dets]

    seg = build_segmenter(cfg)
    pipeline = PerceptionPipeline(_InjectedDetector(), seg, enc, prompts=[])

    b = MapBuilder(cfg, pipeline=pipeline,
                   prompts=sorted({str(v["label"]) for v in vis}))
    smap = b.build_from_frames([scene.frame()])
    # ★ 最后一道自检：**特征场不能是空的**。
    #   这个实验第一次跑出来"两条臂指标一模一样"，就是因为 768 维特征
    #   没能进网格、地图里全是 0 向量 —— 而指标看上去还挺像回事
    #   （类别名 100%、描述 25%），很容易被当成"结论"。宁可当场炸。
    feats = np.stack([np.asarray(o.feature, dtype=np.float64).reshape(-1)
                      for o in smap.objects]) if smap.objects else np.zeros((0, 1))
    if feats.size and float(np.abs(feats).mean()) <= 0.0:
        raise RuntimeError(
            f"{encoder_name} 臂建出来的地图**特征全为 0**（{len(smap.objects)} 个物体，"
            f"声明维度 {smap.feature_dim}）—— 特征没进体素场，实验无效")
    return smap, vis, enc


def eval_map(smap, enc, vis, *, text_encoder) -> Dict[str, Any]:
    """在**地图物体**上跑类别名 / 描述型 / 难负样本三类查询。"""
    objs = list(smap.objects)
    if not objs:
        return {"error": "地图里没有物体"}
    labels = [str(o.label).lower() for o in objs]
    centers = np.stack([np.asarray(o.center, dtype=np.float64).reshape(3)
                        for o in objs], axis=0)
    by_label: Dict[str, List[int]] = {}
    for i, lb in enumerate(labels):
        by_label.setdefault(lb, []).append(i)

    def query(q: str, k: int = 5):
        try:
            return list(smap.query_text(q, top_k=k, text_encoder=text_encoder))
        except Exception:                                    # noqa: BLE001
            return []

    def same_label_hit(res, lb: str) -> bool:
        for r in res:
            try:
                p = np.asarray(r.position, dtype=np.float64).reshape(3)
            except Exception:                                # noqa: BLE001
                continue
            for i in by_label[lb]:
                if float(np.linalg.norm(p - centers[i])) <= HIT_DIST_M:
                    return True
        return False

    out: Dict[str, Any] = {"pool_size": len(objs), "n_labels": len(by_label),
                           "feature_dim": int(smap.feature_dim)}

    # ---- 1) 类别名查询 ----
    top1 = top5 = n_named = 0
    for lb in by_label:
        res = query(lb, 5)
        n_named += 1
        if res and same_label_hit(res[:1], lb):
            top1 += 1
        if same_label_hit(res, lb):
            top5 += 1
    out["named_top1"] = top1 / max(n_named, 1)
    out["named_top5"] = top5 / max(n_named, 1)
    out["n_named_queries"] = n_named

    # ---- 2) 描述型查询（别名表覆盖不到）----
    d1 = d5 = n_desc = 0
    for lb, phrases in DESCRIPTIVE.items():
        if lb not in by_label:
            continue
        for ph in phrases:
            res = query(ph, 5)
            n_desc += 1
            if res and same_label_hit(res[:1], lb):
                d1 += 1
            if same_label_hit(res, lb):
                d5 += 1
    out["desc_top1"] = (d1 / n_desc) if n_desc else None
    out["desc_top5"] = (d5 / n_desc) if n_desc else None
    out["n_desc_queries"] = n_desc

    # ---- 3) 难负样本：地图里没有的类别必须**零接受** ----
    #
    # 两类负样本，缺一不可：
    # · 策划好的**语义近邻**（armchair ↔ chair、cabinet ↔ shelf …）；
    # · 该数据集**官方类别里本图没有的**（2D-3D-S 共 13 类，很多房间只有 6~8 类）。
    #   第一版只用了前者，结果在一个全是 clutter/wall/ceiling 的房间里
    #   一条负样本都生成不出来（`n_hard_neg=0`），指标直接 `n/a` ——
    #   静默变成"没测"，很容易被误读成"通过了"。
    negs: List[str] = []
    for lb in by_label:
        for neg in HARD_NEGATIVES.get(lb, []):
            n = neg.lower()
            if n not in by_label and neg not in negs:
                negs.append(neg)
    for c in ("chair", "table", "door", "bookcase", "window", "board",
              "column", "beam", "sofa", "stairs", "ceiling", "floor", "wall"):
        if c not in by_label and c not in negs:
            negs.append(c)
    negs = negs[:12]

    accepted = 0
    for neg in negs:
        if query(neg, 1):
            accepted += 1
    out["n_hard_neg"] = len(negs)
    out["hard_neg_probes"] = negs
    out["hard_neg_accept"] = (accepted / len(negs)) if negs else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--points", type=int, default=3,
                    help="在全部采集点里等间隔取几个")
    ap.add_argument("--max-views", type=int, default=24)
    ap.add_argument("--resize", type=int, default=540)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--out", default="runs/44_eval_feature_field.json")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2

    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("project.verbose", False)

    locs = sorted(list_locations(args.root), key=lambda l: -len(l.frame_ids))
    locs = [l for l in locs if len(l.frame_ids) >= 24]
    idx = np.linspace(0, len(locs) - 1,
                      num=min(int(args.points), len(locs))).round().astype(int)
    targets = [locs[int(i)] for i in np.unique(idx)]

    arms: Dict[str, List[Dict[str, Any]]] = {"color_hist": [], "siglip": []}
    for loc in targets:
        hr(f"{loc.room}（{loc.uuid[:12]}）")
        for arm in ("color_hist", "siglip"):
            try:
                smap, vis, enc = build_map(cfg, args.root, loc, args,
                                           encoder_name=arm)
            except Exception as e:                           # noqa: BLE001
                print(f"  [{arm}] 建图失败：{type(e).__name__}: {e}")
                continue
            if smap is None:
                print(f"  [{arm}] 无可见 GT，跳过")
                continue
            r = eval_map(smap, enc, vis,
                         text_encoder=(enc if arm != "color_hist" else None))
            r.update({"room": loc.room, "uuid": loc.uuid[:12], "arm": arm,
                      "n_visible_gt": len(vis)})
            arms[arm].append(r)
            f = lambda x: "n/a" if x is None else f"{x*100:5.1f}%"   # noqa: E731
            print(f"  [{arm:<11}] 池 {r['pool_size']:>3}  特征 "
                  f"{r['feature_dim']:>4} 维  类别名 top1 {f(r['named_top1'])}  "
                  f"top5 {f(r['named_top5'])}  描述 top1 {f(r['desc_top1'])}  "
                  f"难负接受 {f(r['hard_neg_accept'])}")

    hr("汇总")
    print(f"{'臂':<13}{'池均值':>7}{'特征维':>7}{'类别名top1':>11}"
          f"{'类别名top5':>11}{'描述top1':>10}{'难负接受':>10}")
    keep: Dict[str, Dict[str, Any]] = {}
    for arm, rs in arms.items():
        if not rs:
            print(f"{arm:<13}  n/a")
            continue
        def m(key: str) -> Optional[float]:
            v = [r[key] for r in rs if r.get(key) is not None]
            return float(np.mean(v)) if v else None
        keep[arm] = {k: m(k) for k in ("pool_size", "feature_dim", "named_top1",
                                       "named_top5", "desc_top1", "hard_neg_accept")}
        fm = lambda x: "n/a" if x is None else f"{x*100:5.1f}%"   # noqa: E731
        print(f"{arm:<13}{keep[arm]['pool_size']:>7.1f}"
              f"{keep[arm]['feature_dim']:>7.0f}"
              f"{fm(keep[arm]['named_top1']):>11}{fm(keep[arm]['named_top5']):>11}"
              f"{fm(keep[arm]['desc_top1']):>10}"
              f"{fm(keep[arm]['hard_neg_accept']):>10}")

    ok = True
    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    hr("判据")
    ch, sg = keep.get("color_hist"), keep.get("siglip")
    if ch and sg:
        check("真视觉语言特征在类别名检索上不劣于 HSV 直方图",
              (sg["named_top1"] or 0) >= (ch["named_top1"] or 0),
              f"siglip {sg['named_top1'] or 0:.3f} vs color_hist {ch['named_top1'] or 0:.3f}")
        check("描述型查询：真特征可用（>30%），默认直方图不可用",
              (sg["desc_top1"] or 0) > 0.30,
              f"siglip {(sg['desc_top1'] or 0)*100:.1f}% vs "
              f"color_hist {((ch['desc_top1'] or 0))*100:.1f}%")
    else:
        check("两条臂都跑出来了", False, "缺少 color_hist 或 siglip 的结果")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": keep, "points": arms},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物已写入 {out}")
    print("\n" + "=" * 78)
    print("结论: " + ("判据通过 ✓" if ok else "存在不通过项 ✗（如实记录）"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
