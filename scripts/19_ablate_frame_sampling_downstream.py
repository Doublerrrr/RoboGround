#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""19 · 下游消融：抽帧策略 → VLM 微调 → QA 准确率。

为什么需要这个脚本（而不是只看数据侧指标）
========================================
`scripts/17` 已经证明"内容自适应抽帧"在**数据侧**明显更好：
同样 24 帧预算，变化捕获 34.3% → 65.9%（1.92×）。

但**"变化捕获高"未必等于"训练效果好"** —— 因为训练真正需要的是
**概念/场景分布的覆盖**，而不是运动量的覆盖。
如果 adaptive 把配额都给了高运动镜头，导致某个静止但含独特物体的长镜头
几乎没进训练集，它的下游效果**可能反而不如 uniform**。

也就是说：**变化捕获这个数据侧指标，可能不是训练价值的正确代理。**
这个结论只能靠真训一遍得到，不能靠推测。本脚本就是干这个的。

实验设计（关键在于"只变一个量"）
=============================
1. **每个镜头 = 一个合成场景**（相机在镜头内平移），用整段帧建一张 3D 地图，
   从中生成 QA —— **地图与 QA 和抽帧策略无关**；
2. 把所有镜头的帧拼成一段视频，用**两种策略、同一个帧预算**抽帧；
3. 把抽中的帧作为**训练图像**，配上它所属镜头的 QA
   （按轮转取，保证**两个训练集样本数完全一致**）；
4. 各微调一次，在**覆盖全部镜头**的验证集上对照。

于是唯一的变量是"**抽中的帧落在哪些镜头 → 哪些场景进了训练集**"。

用法::

    python scripts/19_ablate_frame_sampling_downstream.py --budget 60
    # 然后跑两次微调 + 两次评测（脚本会把命令打出来）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.data.video import (                                    # noqa: E402
    PipelineConfig, QualityConfig, SamplingConfig, ShotDetectionConfig,
    process_video,
)
from roboground.utils.io import ensure_dir                             # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity         # noqa: E402

log = get_logger("ablate_sampling")

PROMPTS = ["table", "chair", "cup", "box", "bottle", "sofa",
           "shelf", "monitor", "trash can", "lamp"]

#: 与 `make_synthetic_shot_video(heterogeneous=True)` 用同一组结构参数：
#: 长短镜头混合 + 运动量差异大。**这是"抽帧策略能拉开差距"的前提条件** ——
#: 如果所有镜头长度与运动都差不多，按内容分配和按时间分配恰好等价。
HETERO_LENGTHS = [34, 5, 18, 4, 26, 7]
HETERO_STRIDES = [0.010, 0.10, 0.03, 0.12, 0.02, 0.08]


def _load_generator():
    """加载 `12_gen_spatial_qa.py`（文件名以数字开头，不能直接 import）。"""
    path = Path(__file__).resolve().parent / "12_gen_spatial_qa.py"
    spec = importlib.util.spec_from_file_location("_gen_spatial_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
def build_shots(args, out_dir: Path) -> Tuple[List[Dict[str, Any]], List[int], List[Any]]:
    """每个镜头建一张 3D 地图并生成 QA；把所有帧拼成一段视频。

    Returns
    -------
    (shots, labels, frames)
        `shots[i]` 含该镜头的 QA 记录池与帧区间；
        `labels[k]` 是第 k 帧属于哪个镜头（**真值标签**，不是检测出来的）。
    """
    from roboground import load_config
    from roboground.data.synthetic import make_synthetic_sequence
    from roboground.mapping import MapBuilder

    gen = _load_generator()
    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("perception.prompts", PROMPTS)
    cfg.set("project.verbose", False)

    shots: List[Dict[str, Any]] = []
    frames: List[Any] = []
    labels: List[int] = []
    cursor = 0

    for i, (length, stride) in enumerate(zip(HETERO_LENGTHS, HETERO_STRIDES)):
        seq = make_synthetic_sequence(
            seed=args.seed + i, num_frames=length, width=args.width,
            height=args.height, num_objects=args.objects, stride=stride)
        smap = MapBuilder(cfg, prompts=PROMPTS).build_from_frames(seq)
        if not smap.objects:
            log.warn(f"镜头 {i} 没建出物体，跳过")
            continue
        # 占位图路径，稍后按"抽中的帧"逐条替换
        recs = gen.build_records(smap, ["images/PLACEHOLDER.png"],
                                 seed=args.seed + i)
        if len(recs) < args.val_per_shot + 2:
            log.warn(f"镜头 {i} 的 QA 太少（{len(recs)}），跳过")
            continue

        shot_frames = list(seq)
        shots.append({
            "index": len(shots), "length": length, "stride": stride,
            "recs": recs, "n_frames": len(shot_frames),
            "start": cursor, "end": cursor + len(shot_frames),
        })
        frames.extend(shot_frames)
        labels.extend([len(shots) - 1] * len(shot_frames))
        cursor += len(shot_frames)
        log.info(f"镜头 {len(shots)-1}: {len(shot_frames)} 帧 / {len(recs)} 条 QA "
                 f"（物体 {len(smap.objects)} 个）")

    return shots, labels, frames


def save_frame_image(frame, out_dir: Path, name: str) -> str:
    from PIL import Image

    ensure_dir(out_dir)
    Image.fromarray(np.asarray(frame.color, dtype=np.uint8)).save(out_dir / name)
    return f"images/{name}"


def _reattach_image(rec: Dict[str, Any], image_rel: str) -> Dict[str, Any]:
    """把记录里的 `<image>...</image>` 换成指定图像（QA 内容不变）。"""
    import re

    out = json.loads(json.dumps(rec))          # 深拷贝
    for m in out["messages"]:
        if m["role"] == "user":
            m["content"] = re.sub(r"<image>.*?</image>",
                                  f"<image>{image_rel}</image>",
                                  m["content"], flags=re.DOTALL)
    return out


def build_strategy_dataset(strategy: str, args, shots, labels, frames,
                           out_dir: Path) -> Dict[str, Any]:
    """用指定抽帧策略产出一个训练集。"""
    from roboground.data.video import DedupConfig

    # ---- 1) 跑管线拿到"被抽中的帧" ----
    # ★ `--disable-dedup`：**这个开关不是图省事，是让实验成立的前提。**
    #
    # 实测发现（见 `docs/视频数据管线报告.md` 第七节）：
    # 去重会把抽帧策略的差异**吃掉**。原始抽帧时
    #   uniform  [22, 3, 11, 3, 16, 5]      ← 按镜头长度成比例
    #   adaptive [ 7, 5, 11, 4, 26, 7]      ← 偏向长且高运动的镜头
    # 镜头 0 差了 3 倍；但**去重之后变成**
    #   uniform  [4, 3, 3, 2, 4, 4]
    #   adaptive [3, 3, 3, 2, 4, 4]
    # 几乎完全一样 —— 因为静止长镜头里抽到的 22 帧本就互为重复，
    # 去重后只剩 4 帧。**变量在上游就被抹平了，下游自然测不出差异。**
    #
    # 所以做消融时必须关掉去重，否则"无差异"这个结果毫无信息量
    # （它只说明两个训练集本来就一样）。
    dedup_cfg = DedupConfig(hash_threshold=-1, use_embedding=False) \
        if args.disable_dedup else None
    cfg = PipelineConfig(
        sampling=SamplingConfig(strategy=strategy, target_frames=args.budget),
        # 放宽 min_side：合成图较小，否则会被分辨率闸门全拒
        quality=QualityConfig(min_side=args.width // 3),
        **({"dedup": dedup_cfg} if dedup_cfg is not None else {}),
    )
    res = process_video([np.asarray(f.color, dtype=np.uint8) for f in frames],
                        cfg=cfg)
    kept = res.kept_indices

    # ---- 2) 每个抽中的帧 → 它所属镜头的下一条 QA（轮转，跳过 val 那几条）----
    per_shot_pool: Dict[int, List[Dict[str, Any]]] = {}
    for s in shots:
        per_shot_pool[s["index"]] = s["recs"][args.val_per_shot:]
    cursor: Dict[int, int] = {}

    records: List[Dict[str, Any]] = []
    shot_hits: Dict[int, int] = {}
    for j, gi in enumerate(kept):
        sid = int(labels[gi])
        pool = per_shot_pool.get(sid, [])
        if not pool:
            continue
        k = cursor.get(sid, 0) % len(pool)
        cursor[sid] = k + 1
        img_rel = save_frame_image(frames[gi], out_dir / "images",
                                   f"{strategy}_shot{sid}_f{gi:04d}.png")
        records.append(_reattach_image(pool[k], img_rel))
        shot_hits[sid] = shot_hits.get(sid, 0) + 1

    return {"strategy": strategy, "records": records,
            "n_frames_picked": len(kept), "shot_hits": shot_hits}


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=60,
                    help="抽帧预算（两种策略必须相同，否则不可比）")
    ap.add_argument("--val-per-shot", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3301)
    ap.add_argument("--objects", type=int, default=5)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--strategies", nargs="*", default=["uniform", "adaptive"])
    ap.add_argument("--out-dir", default="runs/sampling_ablation")
    ap.add_argument("--disable-dedup", action="store_true",
                    help="关掉去重，让抽帧策略的差异真的传到下游（见函数内注释）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    root = Path(args.out_dir)
    ensure_dir(root)

    log.info("构建镜头（每个镜头一张 3D 地图 + QA 池）...")
    t0 = time.perf_counter()
    shots, labels, frames = build_shots(args, root)
    if len(shots) < 2:
        log.error("有效镜头太少，无法做消融")
        return 2
    log.info(f"共 {len(shots)} 个镜头 / {len(frames)} 帧 / "
             f"{sum(len(s['recs']) for s in shots)} 条 QA "
             f"（构建耗时 {time.perf_counter() - t0:.1f}s）")

    # ---- 验证集：覆盖**全部**镜头 ----
    val_recs: List[Dict[str, Any]] = []
    for s in shots:
        for j in range(min(args.val_per_shot, len(s["recs"]))):
            # 用该镜头的中段帧作为验证图像
            gi = (s["start"] + s["end"]) // 2
            img_rel = save_frame_image(frames[gi], root / "images",
                                       f"val_shot{s['index']}_f{gi:04d}.png")
            val_recs.append(_reattach_image(s["recs"][j], img_rel))
    (root / "val.json").write_text(
        json.dumps(val_recs, ensure_ascii=False, indent=2), encoding="utf-8")
    log.ok(f"验证集：{len(val_recs)} 条，覆盖全部 {len(shots)} 个镜头")

    # ---- 各策略产训练集 ----
    summary: Dict[str, Any] = {"n_shots": len(shots), "n_frames": len(frames),
                               "budget": args.budget,
                               "shot_lengths": [s["length"] for s in shots],
                               "val_size": len(val_recs), "runs": {}}
    print()
    print("=" * 88)
    print("抽帧策略 → 训练集（**样本数相同，唯一变量是覆盖了哪些镜头**）")
    print("=" * 88)
    print(f"  镜头长度: {summary['shot_lengths']}   总帧数 {len(frames)}   "
          f"预算 {args.budget}")
    print()
    print(f"  {'策略':<12}{'抽中帧':>8}{'训练样本':>10}{'覆盖镜头':>10}"
          f"{'各镜头样本数':>34}")
    print("  " + "-" * 84)
    for strat in args.strategies:
        ds = build_strategy_dataset(strat, args, shots, labels, frames, root)
        hits = ds["shot_hits"]
        recs = ds["records"]
        (root / f"train_{strat}.json").write_text(
            json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
        cover = len(hits) / len(shots)
        dist = [hits.get(s["index"], 0) for s in shots]
        summary["runs"][strat] = {
            "n_train": len(recs), "n_frames_picked": ds["n_frames_picked"],
            "shot_coverage": cover, "per_shot_counts": dist,
        }
        print(f"  {strat:<12}{ds['n_frames_picked']:>8}{len(recs):>10}"
              f"{cover:>10.0%}   {dist}")

    # ---- 结论预览 ----
    print()
    print("  ★ 读法：两个训练集**样本数相同**，差别只在「覆盖了哪些镜头」。")
    print("     验证集**每个镜头都出题** —— 所以哪个训练集漏掉了镜头，")
    print("     它就会在验证集对应部分丢分。这是本消融能测出差异的前提。")
    unif = summary["runs"].get("uniform", {})
    adap = summary["runs"].get("adaptive", {})
    if unif and adap:
        du, da = unif["per_shot_counts"], adap["per_shot_counts"]
        print()
        print(f"     uniform  各镜头样本数 {du}   ← 分布更均匀")
        print(f"     adaptive 各镜头样本数 {da}   ← 偏向长/高运动镜头")
        print(f"     镜头覆盖：uniform {unif['shot_coverage']:.0%} "
              f"vs adaptive {adap['shot_coverage']:.0%}")
        print()
        print("  ⚠️ 数据侧 `变化捕获` 是 adaptive 更高（见 scripts/17）；")
        print("     但**训练需要的是场景/概念覆盖**，不是运动量覆盖。")
        print("     这两者谁更重要，只能靠下面的微调 + 评测来回答。")

    print()
    print("=" * 88)
    print("下一步（微调 + 评测）")
    print("=" * 88)
    print(f"""
  $ python scripts/13_train_vlm_spatial.py --data {root}/train_uniform.json \\
        --val {root}/val.json --out {root}/lora_uniform --epochs 3
  $ python scripts/13_train_vlm_spatial.py --data {root}/train_adaptive.json \\
        --val {root}/val.json --out {root}/lora_adaptive --epochs 3
  $ python scripts/14_eval_vlm_spatial.py --data {root}/val.json \\
        --adapter {root}/lora_uniform/epoch3
  $ python scripts/14_eval_vlm_spatial.py --data {root}/val.json \\
        --adapter {root}/lora_adaptive/epoch3
""")

    ensure_dir(root)
    (root / "ablation_setup.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.ok(f"配置已保存：{root}/ablation_setup.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
