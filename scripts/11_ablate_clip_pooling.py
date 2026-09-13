#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""11 · CLIP 区域特征池化策略消融。

背景
----
`CLIPEncoder` 有三种把"检测区域"变成"一个特征向量"的方式：

| 策略 | 做法 |
|---|---|
| `crop` | 按 bbox 裁剪（+margin）后整块编码 |
| `mask` | 裁剪后把**掩码外像素涂成区域均值色**，再编码 |
| `full` | 直接编码整图（无区域信息，作为下界基线） |

这三者对应三种工程假设，**必须用数据选**，不能想当然：
- `crop` 假设"框内主要是目标" —— 但实测 bbox 里 35~78% 是背景像素；
- `mask` 假设"掩码外的像素是噪声" —— 但涂成均值色会引入一块非自然区域；
- CLIP 是在**整图**上训练的，裁剪会改变目标的尺度与上下文。

评估指标
--------
在真实场景上，对每个有有效深度的检测：
- `self_sim`：与**自身标签**文本的余弦；
- `other_sim`：与**其他类别**文本的最大余弦；
- `null_sim`：与空文本的最大余弦；
- `margin_pos = self_sim - null_sim`（越大越好，代表"像自己、不像空"）；
- `margin_neg = other_sim - null_sim`（越小越好，代表"不像别的"）；
- `separation = self_sim - other_sim`（**越大越好**，代表类间区分度）；
- `top1_acc`：在全部类别文本里，自身标签是否排第一。

用法::

    python scripts/11_ablate_clip_pooling.py
    python scripts/11_ablate_clip_pooling.py --scenes 3 --output runs/pooling_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.io import ensure_dir                     # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity  # noqa: E402

log = get_logger("ablate_pooling")

NULL_TEXTS = [
    "a photo of nothing.",
    "an empty background.",
    "a photo of a blank wall.",
    "an out of focus image.",
    "a photo of the floor.",
    "a random texture.",
]


def collect_detections(cfg, args):
    """在真实场景上跑检测 + 分割，只保留"掩码内有足够有效深度"的检测。"""
    from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene
    from roboground.perception import build_detector, build_segmenter

    det = build_detector(cfg, name="grounding_dino")
    det.warmup()
    seg = build_segmenter(cfg, name="sam")
    seg.warmup()

    index = load_scene_index(str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz"))
    prompts = list(cfg.get("perception.prompts", []))
    prompt_set = {str(p).lower() for p in prompts}
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))

    samples = []      # (frame, [detections])
    used = []
    for pos in np.argsort(-counts)[:4000]:
        i = int(pos)
        if len(used) >= args.scenes:
            break
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = {str(x).lower() for x in np.asarray(index["label_flat"][off0:off1]).ravel()}
        if len(labels & prompt_set) < 2:
            continue
        sc = load_sunrgbd_scene(index, i, max_depth=8.0, resize=(args.width, args.height))
        if sc is None or sc.boxes_3d.shape[0] == 0:
            continue

        frame = sc.to_frame()
        dets = det.detect(frame, prompts)
        dets = seg.segment(frame, dets) if dets else []

        valid_d = (frame.depth_m > 0.1) & (frame.depth_m < 8.0)
        kept = []
        for d in dets:
            m = np.asarray(d.mask, bool) if d.mask is not None else None
            if m is not None and int((m & valid_d).sum()) >= args.min_valid_px:
                kept.append(d)
        if len(kept) >= 2:
            samples.append((frame, kept))
            used.append({"scene": i, "num_dets": len(kept),
                         "labels": sorted({d.label for d in kept})})

    if not samples:
        raise RuntimeError("没有收集到足够的检测样本")
    log.info(f"使用场景：{[u['scene'] for u in used]}")
    for u in used:
        log.info(f"  场景 {u['scene']}: {u['num_dets']} 个检测，类别 {u['labels']}")
    return samples, used


def evaluate_pooling(samples, encoder_spec: str, pool: str, crop_margin: float = 0.15,
                     square: bool = True) -> dict:
    """在给定 (编码器, 池化策略) 下评估语义区分度。

    `encoder_spec` 形如 `"clip"` 或 `"siglip"`，也可以是具体模型 id。
    """
    from roboground.perception import build_encoder
    from roboground import load_config

    cfg = load_config("configs/perception_openvocab.yaml")
    name = encoder_spec
    model_id = None
    if "/" in encoder_spec or encoder_spec.count("-") > 2:
        # 看起来像 HuggingFace 模型 id → 按前缀判断是哪种编码器
        model_id = encoder_spec
        name = "siglip" if "siglip" in encoder_spec.lower() else "clip"

    kwargs = {"pool": pool, "crop_margin": crop_margin, "square": square}
    if model_id:
        kwargs["model_id"] = model_id
        cfg.set(f"perception.encoder_kwargs.model_id", model_id)
    enc = build_encoder(cfg, name=name)
    # 覆盖构造参数（build_encoder 只传了 config 里的 kwargs）
    for k, v in kwargs.items():
        setattr(enc, k, v)

    self_sims, other_sims, null_sims, top1, rank_hits = [], [], [], [], []
    all_labels: set = set()
    for _, dets in samples:
        all_labels.update(d.label for d in dets)
    labels_sorted = sorted(all_labels)
    texts = [f"a photo of a {l}." for l in labels_sorted]

    # 是否使用模型原生校准（SigLIP）
    native = bool(getattr(enc, "is_calibrated", False))

    for frame, dets in samples:
        feats = enc.encode_regions(frame, dets)
        txt = enc.encode_text(texts)
        nul = enc.encode_text(NULL_TEXTS)

        if native:
            # 原生校准：分数已是概率，直接用于比较
            sims_all = enc.pair_scores(feats, txt)          # (M, N) 概率
            sims_null = enc.pair_scores(feats, nul)         # (M, K) 概率
        else:
            sims_all = feats @ txt.T                        # (M, N) 余弦
            sims_null = feats @ nul.T

        for i, d in enumerate(dets):
            if i >= feats.shape[0]:
                continue
            j_self = labels_sorted.index(d.label)
            s_self = float(sims_all[i, j_self])
            s_other = float(max(sims_all[i, j] for j in range(len(texts)) if j != j_self))
            s_null = float(sims_null[i].max())

            self_sims.append(s_self)
            other_sims.append(s_other)
            null_sims.append(s_null)
            top1.append(int(np.argmax(sims_all[i]) == j_self))
            order = np.argsort(-sims_all[i]).tolist()
            rank_hits.append(order.index(j_self) + 1)

    ss, os_, ns = np.array(self_sims), np.array(other_sims), np.array(null_sims)
    return {
        "encoder": name,
        "pool": pool,
        "native_calibration": native,
        "n": int(ss.size),
        "self_sim": float(ss.mean()),
        "other_sim": float(os_.mean()),
        "null_sim": float(ns.mean()),
        "margin_pos": float((ss - ns).mean()),
        "margin_neg": float((os_ - ns).mean()),
        "separation": float((ss - os_).mean()),
        "top1_acc": float(np.mean(top1)),
        "median_rank": float(np.median(rank_hits)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--min-valid-px", type=int, default=100,
                    help="掩码内至少多少有效深度像素才纳入评估")
    ap.add_argument("--clip-path", default=r"G:\RoboGround\weights\clip-vit-base-patch32")
    ap.add_argument("--siglip-path", default=r"G:\RoboGround\weights\siglip-base-patch16-224")
    ap.add_argument("--pools", nargs="*", default=["crop", "mask"])
    ap.add_argument("--encoders", nargs="*", default=["clip", "siglip"],
                    help="要对比的编码器：clip / siglip（也可传 HF 模型 id）")
    ap.add_argument("--crop-margin", type=float, default=0.15)
    ap.add_argument("--no-square", action="store_true", help="关闭方形填充（对照用）")
    ap.add_argument("--output", default="runs/pooling_ablation.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    from roboground import load_config

    cfg = load_config("configs/perception_openvocab.yaml")
    cfg.set("perception.encoder_kwargs.model_id", args.clip_path)

    samples, meta = collect_detections(cfg, args)
    total = sum(len(d) for _, d in samples)
    log.info(f"共 {total} 个检测样本参与评估")

    path_map = {"clip": args.clip_path, "siglip": args.siglip_path}

    print()
    header = (f"{'编码器':<8}{'池化':<7}{'校准':<10}{'n':>5}{'self':>8}{'other':>8}{'null':>8}"
              f"{'margin+':>10}{'分离度':>10}{'Top1':>7}{'中位排名':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for enc_name in args.encoders:
        spec = path_map.get(enc_name, enc_name)
        for pool in args.pools:
            try:
                r = evaluate_pooling(samples, spec, pool, args.crop_margin,
                                     square=not args.no_square)
                r["encoder"] = enc_name
                rows.append(r)
                calib = "原生sigmooid" if r["native_calibration"] else "余弦(需校准)"
                print(f"{enc_name:<8}{pool:<7}{calib:<10}{r['n']:>5}{r['self_sim']:>8.3f}"
                      f"{r['other_sim']:>8.3f}{r['null_sim']:>8.3f}"
                      f"{r['margin_pos']:>+10.4f}{r['separation']:>+10.4f}"
                      f"{r['top1_acc']:>7.2f}{r['median_rank']:>9.1f}")
            except Exception as exc:
                log.warn(f"{enc_name}/{pool} 失败：{type(exc).__name__}: {str(exc)[:160]}")

    print()
    if not rows:
        log.error("所有组合都失败")
        return 2

    best = max(rows, key=lambda r: (r["top1_acc"], r["separation"]))
    log.ok(f"最佳组合：{best['encoder']} + pool='{best['pool']}'  "
           f"（Top1={best['top1_acc']:.2f}, 分离度={best['separation']:+.4f}, "
           f"中位排名={best['median_rank']:.1f}）")

    clip_best = max([r for r in rows if r["encoder"] == "clip"],
                    key=lambda r: r["top1_acc"], default=None)
    siglip_best = max([r for r in rows if r["encoder"] == "siglip"],
                      key=lambda r: r["top1_acc"], default=None)
    if clip_best and siglip_best:
        delta = siglip_best["top1_acc"] - clip_best["top1_acc"]
        log.info(f"对比：CLIP Top1={clip_best['top1_acc']:.2f} → "
                 f"SigLIP Top1={siglip_best['top1_acc']:.2f}  "
                 f"（{'提升' if delta >= 0 else '下降'} {abs(delta):.2f} 绝对百分点）")

    if best["top1_acc"] < 0.5:
        log.warn("最佳 Top-1 仍低于 0.5 —— 区域级开放词汇的区分度有限。")
        log.info("可行的改进方向：① 多尺度裁剪 + 上下文扩展；"
                 "② 融合 DINOv2 的密集特征做区域池化；"
                 "③ 用检测器的文本对齐头（Grounding DINO 本身就是开放词汇）"
                 "作为标签来源，把嵌入匹配降级为辅助。")

    if args.output:
        ensure_dir(Path(args.output).parent)
        Path(args.output).write_text(
            json.dumps({"rows": rows, "scenes": meta}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.ok(f"消融结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
