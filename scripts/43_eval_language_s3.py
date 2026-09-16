# -*- coding: utf-8 -*-
"""S3 语言侧的定量评测：7 类意图在**真实地图**上答对了多少？（批评 #1）

批评 #1 说的是实话：项目一直讲 S1/S2 的数字，S3（"自然语言 → 3D 位置"）
只有定性描述（"有规则引擎、有 VLM 后端"），**没有一个可引用的准确率**。
这个脚本就是补这一块。

怎么做到"答案可判对错"
====================
关键在于**答案可以从真值算出来**。做法：

1. 在真实 2D-3D-S 采集点上建一张全景地图，检测用 **GT 框注入**
   （和 `scripts/37/41/42` 同一手法）——
   于是"地图里有什么、在哪"是**已知的**，不是猜的；
2. 题面**由真值程序化生成**，每一类的期望答案都能从 GT 框算出来；
3. 逐题比对。

七类意图与判对标准
=================
| 意图 | 问法 | 判对标准 |
|---|---|---|
| `locate` | 「<类>在哪？」 | top-1 命中必须落在某个同类 GT 的 2 m 内；另外记 `recall@k`（该类有几个 GT 被返回集覆盖） |
| `count` | 「有几个<类>？」 | 回答里的数字 == **可见 GT** 的该类个数 |
| `distance` | 「<A>到<B>多远？」 | `distances[A<->B_center]` 与 GT 中心距之差 ≤ 1.0 m |
| `nearest` | 「离我最近的<类>？」 | 返回的就是 GT 里**真的最近**那一个 |
| `relation` | 「<A>在<B>上面吗？」 | 用 GT 框算出的上下关系与引擎的判定一致 |
| `list_on` | 「<锚点>上有什么？」 | 返回集合 == GT 里中心落在锚点框内的物体集合 |
| `describe` | 「场景里有什么？」 | 回答文本至少提到 1 个地图里真实存在的标签 |
| 拒识 | 「冰箱在哪？」（地图里没有） | 必须**明确说没找到**，不能返回别的物体 |

`relation` 只测**上下**（`above`/`below`）：世界系是 z 轴朝上，GT 里
"谁在上面"没有歧义；左右前后要依赖朝向约定，容易把"约定不同"误判成"答错"。

跑法：
    python scripts/43_eval_language_s3.py --points 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config  # noqa: E402
from roboground.data.pano_scene import PanoScene, load_scene, visible_gt  # noqa: E402
from roboground.data.stanford2d3d import list_locations  # noqa: E402

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")
HIT_DIST_M = 2.0          # locate/nearest 的"命中"距离
DIST_TOL_M = 1.0          # distance 意图的容差
#: 用于"拒识"测试的类别：2D-3D-S 的 13 类里**没有**这些
ABSENT_LABELS = ("refrigerator", "airplane", "submarine", "microwave")


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def _hit(targets: Sequence[Any], box: np.ndarray, *,
         max_dist: float = HIT_DIST_M) -> bool:
    c = np.asarray(box[:3], dtype=np.float64)
    for t in targets or []:
        try:
            p = np.asarray(t.center, dtype=np.float64).reshape(3)
        except Exception:                                       # noqa: BLE001
            continue
        if float(np.linalg.norm(p - c)) <= max_dist:
            return True
    return False


def _parse_count(text: str) -> Optional[int]:
    """从回答里抠出数字（中文/阿拉伯数字都认）。"""
    digits = re.findall(r"\d+", str(text))
    if digits:
        return int(digits[0])
    cn = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
          "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    for ch in str(text):
        if ch in cn:
            return cn[ch]
    return None


# ==========================================================================
# 一个采集点：建图 + 出题 + 判分
# ==========================================================================
def eval_point(cfg, root: Path, loc, args) -> Dict[str, Any]:
    from roboground.reasoning import RuleEngine
    from roboground.types import Detection2D
    from roboground.mapping import MapBuilder

    scene: PanoScene = load_scene(root, loc, width=args.width, height=args.height,
                                  max_frames=args.max_views,
                                  max_depth=args.max_range,
                                  resize=(args.resize, args.resize))
    vis = visible_gt(scene, max_range_m=args.max_range)
    if not vis:
        return {"room": loc.room, "uuid": loc.uuid[:12], "skipped": "无可见 GT"}

    dets = []
    for it in vis:
        for (u0, v0, u1, v1) in it.get("uv_parts") or [it["uv"]]:
            dets.append(Detection2D(label=str(it["label"]), score=1.0,
                                    bbox=np.array([u0, v0, u1, v1], dtype=np.float64),
                                    prompt=str(it["label"])))
    prompts = sorted({str(v["label"]) for v in vis})
    b = MapBuilder(cfg, prompts=prompts)
    b.pipeline.run = lambda f, prompts=None: dets            # type: ignore[assignment]
    smap = b.build_from_frames([scene.frame()])
    engine = RuleEngine(smap, cfg=cfg)

    labels = [str(v["label"]) for v in vis]
    boxes = np.stack([np.asarray(v["box"], dtype=np.float64) for v in vis], axis=0)
    by_label: Dict[str, List[int]] = {}
    for i, lb in enumerate(labels):
        by_label.setdefault(lb, []).append(i)
    robot = np.asarray(smap.meta.get("robot_position", [0, 0, 0]), dtype=np.float64)

    rows: List[Dict[str, Any]] = []

    def add(kind: str, query: str, ok: Optional[bool], **extra) -> None:
        rows.append({"intent": kind, "query": query, "ok": ok, **extra})

    # ⚠️ 多实例的歧义问题：`distance`/`relation`/`list_on` 这三类问句，
    #    如果某个类在地图里有**多个**实例，"A 到 B 多远"就没有唯一真值 ——
    #    引擎挑了哪一对、我按哪一对算 GT，都会让结果看起来"答错了"，
    #    实际只是问句本身有歧义。所以这三类**只用单实例类别出题**。
    #    （第一版没做这个限制，relation 因此只有 57% —— 复查后发现
    #     失败样例全是"beams/书柜在桌子上面吗"这种多实例配对问题，
    #     属于判据问题，不是引擎问题。）
    uniq = {lb: idxs[0] for lb, idxs in by_label.items() if len(idxs) == 1}

    # ---------------- locate ----------------
    for lb, idxs in sorted(by_label.items()):
        res = engine.answer(f"{lb} 在哪")
        tgt = list(res.targets or [])
        top_ok = bool(tgt) and _hit(tgt[:1], boxes[idxs[0]])
        covered = sum(1 for i in idxs if _hit(tgt, boxes[i]))
        add("locate", f"{lb} 在哪", top_ok,
            recall_at_k=covered / max(len(idxs), 1), n_gt=len(idxs),
            top1_any_gt=bool(tgt) and any(_hit(tgt[:1], boxes[j]) for j in idxs))

    # ---------------- count ----------------
    # 同时记录**地图里的实例数**，用来把"答错"归因到映射层还是语言层：
    #   地图数 ≠ 可见 GT 数  → 是关联/建图的问题
    #   地图数 = 可见 GT 数，但回答的数字不对 → 才是语言层的问题
    for lb, idxs in sorted(by_label.items()):
        res = engine.answer(f"有几个{lb}")
        got = _parse_count(res.answer)
        n_map = sum(1 for o in smap.objects if str(o.label).lower() == lb.lower())
        add("count", f"有几个{lb}", got == len(idxs),
            got=got, expect=len(idxs), map_count=n_map)

    # ---------------- distance（只用单实例类别）----------------
    uniq_labels = sorted(uniq)
    for a, b_ in ((a, c) for a in uniq_labels for c in uniq_labels if a < c):
        ia, ib = uniq[a], uniq[b_]
        d_gt = float(np.linalg.norm(boxes[ia, :3] - boxes[ib, :3]))
        res = engine.answer(f"{a} 到 {b_} 有多远")
        key = f"{a}<->{b_}_center"
        got = (res.distances or {}).get(key)
        if got is None:                       # 键名可能因标签解析顺序而反
            got = (res.distances or {}).get(f"{b_}<->{a}_center")
        ok = got is not None and abs(float(got) - d_gt) <= DIST_TOL_M
        add("distance", f"{a} 到 {b_} 有多远", ok,
            gt_m=d_gt, got_m=(None if got is None else float(got)))

    # ---------------- nearest ----------------
    for lb, idxs in sorted(by_label.items()):
        dc = [float(np.linalg.norm(boxes[i, :3] - robot)) for i in idxs]
        best = idxs[int(np.argmin(dc))]
        res = engine.answer(f"离我最近的{lb}在哪")
        ok = _hit(list(res.targets or [])[:1], boxes[best])
        add("nearest", f"离我最近的{lb}在哪", ok, n_gt=len(idxs))

    # ---------------- relation（只用单实例 + GT 本身无歧义的配对）----------------
    #
    # 引擎的定义（`dominant_relation`）：先看**归一化分离量最大的那个轴**，
    # 该轴是 z 且 |dz| ≥ 0.10 才叫 above/below。所以"在不在上面"这种问法
    # 在"水平方向分离更大"的配对上**本来就没有确定答案**。
    # 出题时只保留 GT 也满足同一条件的配对：z 轴占主导且 |dz| ≥ 0.10。
    # 这不是在放水 —— 而是在问一个**有确定答案**的问题。
    for a, b_ in ((a, c) for a in uniq_labels for c in uniq_labels if a != c):
        ia, ib = uniq[a], uniq[b_]
        dz = float(boxes[ia, :3][2] - boxes[ib, :3][2])
        dxy = float(np.linalg.norm(boxes[ia, :3][:2] - boxes[ib, :3][:2]))
        if abs(dz) < 0.10 or abs(dz) <= dxy:
            continue                                   # GT 本身就没有确定答案
        gt_above = dz > 0
        ask = "上面" if gt_above else "下面"
        res = engine.answer(f"{a} 在 {b_} {ask}吗")
        flag = (res.debug or {}).get("relation_matches_question")
        # ★ 问句的方向**总是按 GT 来写**（"在上面吗" 对应 gt_above=True，
        #   "在下面吗" 对应 gt_above=False），所以正确答案永远是"是"。
        #   `relation_matches_question` 的含义是"引擎的判定与所问的关系一致"，
        #   因此判对的标准就是它 == True。
        #   ⚠️ 第一版写成 `flag == gt_above`，把"问了『下面吗』、引擎正确答是"
        #   判成了错 —— 一个纯粹的符号错误，却让 relation 准确率凭空掉到 50%。
        ok = (flag is True)
        add("relation", f"{a} 在 {b_} {ask}吗", ok, gt_above=gt_above,
            engine=flag, answer=str(res.answer)[:60])

    # ---------------- list_on（**只报信息，不参与判据**）----------------
    #
    # ⚠️ 为什么不作为判据：2D-3D-S 的 GT 是**轴对齐包围盒**，
    #    "桌子底下的椅子"其中心天然落在桌子的 AABB 里 ——
    #    于是"桌子上面有什么"的 GT 答案是"椅子"，而引擎返回别的，
    #    两边都不算错，只是"AABB 包含"不等于口语里的"放在上面"。
    #    要把它做成可靠判据，需要支撑面/接触关系标注，本项目没有。
    for lb, anchor in sorted(uniq.items()):
        lo = boxes[anchor, :3] - boxes[anchor, 3:6] / 2.0 - 0.05
        hi = boxes[anchor, :3] + boxes[anchor, 3:6] / 2.0 + 0.05
        inside = sorted({labels[j] for j in range(len(labels))
                         if j != anchor and bool(np.all(boxes[j, :3] >= lo)
                                                 and np.all(boxes[j, :3] <= hi))})
        if not inside:
            continue
        res = engine.answer(f"{lb} 上有什么")
        got = sorted({str(o.label) for o in (res.targets or [])})
        add("list_on", f"{lb} 上有什么", None,
            expect=inside, got=got, covered=set(inside).issubset(got))

    # ---------------- describe ----------------
    res = engine.answer("场景里有什么")
    txt = str(res.answer)
    mentioned = [lb for lb in by_label if lb in txt]
    add("describe", "场景里有什么", len(mentioned) > 0,
        n_labels=len(by_label), mentioned=len(mentioned))

    # ---------------- 拒识 ----------------
    for lb in ABSENT_LABELS:
        if lb in by_label:
            continue
        res = engine.answer(f"{lb} 在哪")
        tgt = list(res.targets or [])
        said_no = ("没找到" in str(res.answer)) or ("没有" in str(res.answer))
        add("reject", f"{lb} 在哪", (not tgt) and said_no,
            n_targets=len(tgt), answer=str(res.answer)[:40])

    return {
        "room": loc.room, "uuid": loc.uuid[:12],
        "n_gt_visible": len(vis), "n_map_objects": int(smap.num_objects),
        "n_labels": len(by_label), "questions": rows,
    }


def summarize(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    kinds = sorted({r["intent"] for r in all_rows})
    for k in kinds:
        rs = [r for r in all_rows if r["intent"] == k]
        ok = [r for r in rs if r.get("ok") is not None]
        out[k] = {
            "n": len(ok),
            "acc": (sum(1 for r in ok if r["ok"]) / len(ok)) if ok else None,
        }
        if k == "locate":
            rk = [r["recall_at_k"] for r in rs if r.get("recall_at_k") is not None]
            tl = [r["top1_any_gt"] for r in rs if r.get("top1_any_gt") is not None]
            out[k]["recall_at_k"] = float(np.mean(rk)) if rk else None
            out[k]["top1_any_gt"] = float(np.mean(tl)) if tl else None
        if k == "distance":
            e = [abs(r["got_m"] - r["gt_m"]) for r in rs
                 if r.get("got_m") is not None and r.get("gt_m") is not None]
            out[k]["median_abs_err_m"] = float(np.median(e)) if e else None
        if k == "count":
            # 归因：地图实例数对不对（映射层）vs 数字报得对不对（语言层）
            mc = [r for r in rs if r.get("map_count") is not None
                  and r.get("expect") is not None]
            out[k]["map_count_ok"] = (float(np.mean([1.0 if r["map_count"] == r["expect"]
                                                     else 0.0 for r in mc]))
                                      if mc else None)
        if k == "list_on":
            cv = [r["covered"] for r in rs if r.get("covered") is not None]
            out[k]["covered_rate"] = float(np.mean(cv)) if cv else None
    n_ok = sum(1 for r in all_rows if r.get("ok") is not None)
    out["_overall"] = {
        "n": n_ok,
        "acc": (sum(1 for r in all_rows if r.get("ok")) / n_ok) if n_ok else None,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--points", type=int, default=4,
                    help="在全部采集点里等间隔取几个（--rooms 给了就忽略）")
    ap.add_argument("--max-views", type=int, default=24)
    ap.add_argument("--resize", type=int, default=540)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--out", default="runs/43_eval_language_s3.json")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2

    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("project.verbose", False)

    if args.rooms:
        from roboground.data.pano_scene import select_location

        targets = [select_location(args.root, room=r, min_frames=8)
                   for r in args.rooms]
    else:
        locs = sorted(list_locations(args.root), key=lambda l: -len(l.frame_ids))
        locs = [l for l in locs if len(l.frame_ids) >= 24]
        idx = np.linspace(0, len(locs) - 1,
                          num=min(int(args.points), len(locs))).round().astype(int)
        targets = [locs[int(i)] for i in np.unique(idx)]

    results: List[Dict[str, Any]] = []
    for loc in targets:
        hr(f"{loc.room}（{loc.uuid[:12]}）")
        try:
            r = eval_point(cfg, args.root, loc, args)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [skip] {type(e).__name__}: {e}")
            continue
        results.append(r)
        if "skipped" in r:
            print(f"  跳过：{r['skipped']}")
            continue
        s = summarize(r["questions"])
        print(f"  可见 GT {r['n_gt_visible']}（{r['n_labels']} 类）→ 地图 "
              f"{r['n_map_objects']} 个物体；出题 {len(r['questions'])} 道")
        for k in sorted(s):
            if k == "_overall":
                continue
            v = s[k]
            extra = ""
            if k == "locate" and v.get("recall_at_k") is not None:
                extra = (f"  top1命中 {v['top1_any_gt']*100:.0f}%  "
                         f"recall@k {v['recall_at_k']*100:.0f}%")
            if k == "distance" and v.get("median_abs_err_m") is not None:
                extra = f"  误差中位 {v['median_abs_err_m']:.3f} m"
            print(f"    {k:<9} n={v['n']:>3}  准确率 "
                  f"{('n/a' if v['acc'] is None else format(v['acc']*100, '.1f') + '%')}"
                  + extra)
        print(f"    {'总体':<9} n={s['_overall']['n']:>3}  准确率 "
              f"{s['_overall']['acc']*100:.1f}%")

    all_rows = [q for r in results for q in r.get("questions", [])]
    s_all = summarize(all_rows)

    hr("汇总（所有采集点合并）")
    print(f"{'意图':<10}{'题数':>6}{'准确率':>9}   备注")
    notes = {
        "locate": "top1 命中同类 GT 的 2 m 内",
        "count": "数字 == 可见 GT 个数",
        "distance": "误差 ≤ 1.0 m（只用单实例类别）",
        "nearest": "返回的确实是最近的那个",
        "relation": "上下关系与 GT 一致（只用 GT 无歧义的配对）",
        "list_on": "**仅参考**：AABB 包含 ≠ 口语的『在上面』",
        "describe": "至少提到 1 个真实标签",
        "reject": "地图里没有的类别必须明确说没找到",
    }
    for k in sorted(s_all):
        if k == "_overall":
            continue
        v = s_all[k]
        extra = ""
        if k == "count" and v.get("map_count_ok") is not None:
            extra = f"（其中『地图实例数就对』占 {v['map_count_ok']*100:.0f}%）"
        if k == "list_on" and v.get("covered_rate") is not None:
            extra = f"（返回集覆盖 GT 的比例 {v['covered_rate']*100:.0f}%）"
        print(f"{k:<10}{v['n']:>6}"
              f"{('n/a' if v['acc'] is None else format(v['acc']*100, '.1f') + '%'):>9}"
              f"   {notes.get(k, '')}{extra}")
    print(f"{'总体':<10}{s_all['_overall']['n']:>6}"
          f"{s_all['_overall']['acc']*100:>8.1f}%")
    lo = s_all.get("locate", {})
    if lo.get("recall_at_k") is not None:
        print(f"\n  locate 细分：top-1 落在同类 GT 2 m 内 "
              f"{(lo.get('top1_any_gt') or 0)*100:.1f}%  "
              f"（但要求正好是**第 idxs[0] 个**同类 GT 时 "
              f"{(lo['acc'] or 0)*100:.1f}%）")
        print(f"               recall@k（该类有几个 GT 被返回集覆盖）"
              f"{lo['recall_at_k']*100:.1f}%")
    ds = s_all.get("distance", {})
    if ds.get("median_abs_err_m") is not None:
        print(f"  distance 细分：中心距误差中位 {ds['median_abs_err_m']:.3f} m")

    ok = True
    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    hr("判据")
    tot = s_all["_overall"]
    check("总题量足够（≥100 道）", tot["n"] >= 100, f"{tot['n']} 道")
    check("拒识率 = 100%（地图里没有的类别绝不能返回别的物体）",
          (s_all.get("reject", {}).get("acc") or 0) >= 0.999,
          f"{(s_all.get('reject', {}).get('acc') or 0)*100:.1f}%")
    check("定位问答 top-1 命中 ≥ 90%",
          (s_all.get("locate", {}).get("top1_any_gt") or 0) >= 0.90,
          f"{(s_all.get('locate', {}).get('top1_any_gt') or 0)*100:.1f}%")
    check("计数问答准确率 ≥ 90%",
          (s_all.get("count", {}).get("acc") or 0) >= 0.90,
          f"{(s_all.get('count', {}).get('acc') or 0)*100:.1f}%")
    check("最近物体问答准确率 ≥ 90%",
          (s_all.get("nearest", {}).get("acc") or 0) >= 0.90,
          f"{(s_all.get('nearest', {}).get('acc') or 0)*100:.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": s_all, "points": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物已写入 {out}")
    print("\n" + "=" * 78)
    print("结论: " + ("判据通过 ✓" if ok else "存在不通过项 ✗（如实记录）"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
