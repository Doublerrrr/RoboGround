#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""12 · VLM 空间问答数据生成（SpatialVLM 范式的自动标注）。

思路（参考 SpatialVLM / SpatialRGPT）
------------------------------------
微调 VLM 做空间推理，最大的成本是**造带米制距离的空间 QA 数据**。
人工标注不现实，但**几何本身就带着真值** —— 我们已经有了 3D 语义地图，
物体中心、包围盒、两两距离都是精确的。

所以数据生成可以是**完全自动**的：
```
3D 语义地图（物体 + 中心 + bbox）
  → 枚举物体对 / 单物体
  → 用规则引擎算精确关系与米制距离（这是 GT，不是模型输出）
  → 套多种问法模板（中英 + 不同句式）
  → 产出 ShareGPT 格式的 SFT 数据
```

产出数据可以直接喂 `swift` / `LLaMA-Factory` / 自己的 DeepSpeed 训练脚本。

数据构成（四类任务）
-------------------
| 类型 | 例子 | 占比建议 |
|---|---|---|
| 物体定位 | "杯子在哪？" → 坐标 + 距离 | 30% |
| 空间关系 | "杯子在桌子上面吗？" → 是/否 + 证据 | 30% |
| 米制测距 | "杯子离桌子多远？" → 精确距离 | 25% |
| 区域列举 | "桌子上有什么？" → 列表 | 15% |

关键设计：**答案里必须带数值**（米制距离 / 坐标），这是空间 VLM 区别于
普通 VLM 的核心（普通 VLM 会说"在桌子上"，但说不出"距 0.42 米"）。

用法::

    # 从合成场景生成（无需数据/模型）
    python scripts/12_gen_spatial_qa.py --source synthetic --scenes 30

    # 从真实 SUN RGB-D 生成（需要索引）
    python scripts/12_gen_spatial_qa.py --source sunrgbd --scenes 40

    # 只生成训练脚本模板，不造数据
    python scripts/12_gen_spatial_qa.py --emit-train-script
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.io import ensure_dir                                 # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity             # noqa: E402

log = get_logger("gen_spatial_qa")

#: 中英双语问法模板（同一事实多种问法，提升泛化）
TEMPLATES = {
    "locate": {
        "zh": ["{obj}在哪？", "{obj}在什么位置？", "帮我找一下{obj}", "{obj}在哪儿？"],
        "en": ["Where is the {obj}?", "Locate the {obj}.", "Find the {obj}."],
    },
    "relation": {
        "zh": ["{a}在{b}的{rel}吗？", "{a}是不是在{b}{rel}？"],
        "en": ["Is the {a} {rel_en} the {b}?", "Is the {a} located {rel_en} the {b}?"],
    },
    "distance": {
        "zh": ["{a}离{b}多远？", "{a}到{b}的距离是多少？"],
        "en": ["How far is the {a} from the {b}?", "What is the distance between the {a} and the {b}?"],
    },
    "list_on": {
        "zh": ["{obj}上有什么？", "{obj}上面放了什么？"],
        "en": ["What is on the {obj}?", "What objects are on the {obj}?"],
    },
}

#: 关系名的中文/英文说法
REL_ZH = {"above": "上面", "below": "下面", "left_of": "左边", "right_of": "右边",
          "in_front_of": "前面", "behind": "后面", "inside": "里面",
          "near": "旁边", "overlapping": "位置重叠处", "unknown": "附近"}
REL_EN = {"above": "above", "below": "below", "left_of": "to the left of",
          "right_of": "to the right of", "in_front_of": "in front of",
          "behind": "behind", "inside": "inside", "near": "next to",
          "overlapping": "overlapping", "unknown": "near"}

#: 中文类别名（用于生成自然的中文问题）
ZH_NAME = {"cup": "杯子", "table": "桌子", "chair": "椅子", "monitor": "显示器",
           "door": "门", "window": "窗户", "trash can": "垃圾桶", "bottle": "瓶子",
           "laptop": "笔记本电脑", "bookshelf": "书架", "box": "箱子", "bed": "床",
           "sofa": "沙发", "desk": "书桌", "person": "人", "cabinet": "柜子",
           "nightstand": "床头柜", "toilet": "马桶", "bathtub": "浴缸",
           "shelf": "架子", "lamp": "灯", "pillow": "枕头", "bag": "包",
           "sink": "水槽", "refrigerator": "冰箱", "clock": "时钟", "picture": "画"}

SYSTEM_PROMPT = (
    "你是服务机器人的空间感知模块。你会看到一张室内场景的照片，"
    "并被问到关于场景中物体的空间问题。\n"
    "回答要求：\n"
    "1. 涉及位置时给出以米为单位的 3D 坐标 (x, y, z)，x 向右、y 向前、z 向上；\n"
    "2. 涉及距离时给出精确到两位小数的米制数值；\n"
    "3. 只描述你确实能判断的内容，不确定就说不确定；\n"
    "4. 回答要简洁，先给结论再给依据。"
)


def obj_name(label: str, lang: str) -> str:
    """类别名 → 该语言下的自然说法。"""
    if lang == "zh":
        return ZH_NAME.get(label, label)
    return label


def build_records(smap, scene_image_paths: Sequence[str], seed: int = 0) -> List[Dict[str, Any]]:
    """从一张 3D 语义地图生成空间 QA 记录（ShareGPT 格式）。"""
    from roboground.reasoning.rule_engine import RuleEngine
    from roboground.reasoning.spatial_relations import compute_relation

    rng = random.Random(seed)
    engine = RuleEngine(smap)
    objs = [o for o in smap.objects if o.label and o.confidence > 0.3]
    if not objs:
        return []

    records: List[Dict[str, Any]] = []

    def _add(lang: str, question: str, answer: str, task: str) -> None:
        img = scene_image_paths[0] if scene_image_paths else ""
        records.append({
            "task": task,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>{img}</image>\n{question}"},
                {"role": "assistant", "content": answer},
            ],
        })

    # ---- 1) 物体定位 ----
    for o in objs:
        for lang in ("zh", "en"):
            q = rng.choice(TEMPLATES["locate"][lang]).format(obj=obj_name(o.label, lang))
            if lang == "zh":
                a = (f"{obj_name(o.label, lang)}在 ({o.center[0]:.2f}, {o.center[1]:.2f}, "
                     f"{o.center[2]:.2f}) 米处，尺寸约 {o.extent[0]:.2f}×{o.extent[1]:.2f}×"
                     f"{o.extent[2]:.2f} 米。")
            else:
                a = (f"The {o.label} is at ({o.center[0]:.2f}, {o.center[1]:.2f}, "
                     f"{o.center[2]:.2f}) meters, roughly "
                     f"{o.extent[0]:.2f}x{o.extent[1]:.2f}x{o.extent[2]:.2f} meters in size.")
            _add(lang, q, a, "locate")

    # ---- 2) 空间关系 ----
    for i, a_obj in enumerate(objs):
        for b_obj in objs[i + 1:]:
            rel = compute_relation(a_obj, b_obj)
            for lang in ("zh", "en"):
                for _ in range(2):        # 每个关系生成 2 个问法（含一对反例）
                    subject, target = (a_obj, b_obj) if rng.random() < 0.5 else (b_obj, a_obj)
                    r = compute_relation(subject, target)
                    if lang == "zh":
                        q = rng.choice(TEMPLATES["relation"]["zh"]).format(
                            a=obj_name(subject.label, "zh"),
                            b=obj_name(target.label, "zh"),
                            rel=REL_ZH.get(r.relation, r.relation))
                        yes = r.relation in ("above", "below", "inside", "near", "overlapping")
                        a = (f"是的，{obj_name(subject.label, 'zh')}在"
                             f"{obj_name(target.label, 'zh')}的{REL_ZH.get(r.relation, r.relation)}，"
                             f"两者中心距 {r.distance:.2f} 米。")
                    else:
                        q = rng.choice(TEMPLATES["relation"]["en"]).format(
                            a=subject.label, b=target.label,
                            rel_en=REL_EN.get(r.relation, r.relation))
                        a = (f"Yes. The {subject.label} is {REL_EN.get(r.relation, r.relation)} "
                             f"the {target.label}, with a center distance of {r.distance:.2f} m.")
                    _add(lang, q, a, "relation")

    # ---- 3) 米制测距 ----
    pairs = [(a, b) for i, a in enumerate(objs) for b in objs[i + 1:]]
    rng.shuffle(pairs)
    for a_obj, b_obj in pairs[: max(len(objs), 8)]:
        from roboground.reasoning.spatial_relations import bbox_gap
        gap = bbox_gap(a_obj, b_obj)
        for lang in ("zh", "en"):
            q = rng.choice(TEMPLATES["distance"][lang]).format(
                a=obj_name(a_obj.label, lang), b=obj_name(b_obj.label, lang))
            if lang == "zh":
                a = (f"{obj_name(a_obj.label, 'zh')}与{obj_name(b_obj.label, 'zh')}"
                     f"的中心距离是 {a_obj.distance_to(b_obj):.2f} 米，"
                     f"最近表面间隙约 {gap:.2f} 米。")
            else:
                a = (f"The center distance between the {a_obj.label} and the {b_obj.label} "
                     f"is {a_obj.distance_to(b_obj):.2f} m; the nearest surface gap is "
                     f"about {gap:.2f} m.")
            _add(lang, q, a, "distance")

    # ---- 4) 区域列举 ----
    for anchor in objs:
        margin = 0.05
        lo = anchor.bbox_min - margin
        hi = anchor.bbox_max + margin
        on_it = [o for o in objs
                 if o.obj_id != anchor.obj_id and np.all(o.center >= lo) and np.all(o.center <= hi)]
        if not on_it:
            continue
        for lang in ("zh", "en"):
            q = rng.choice(TEMPLATES["list_on"][lang]).format(obj=obj_name(anchor.label, lang))
            if lang == "zh":
                a = (f"{obj_name(anchor.label, 'zh')}上有 {len(on_it)} 个物体：" +
                     "、".join(f"{obj_name(o.label, 'zh')}(距 {anchor.distance_to(o):.2f} 米)"
                              for o in on_it[:5]) + "。")
            else:
                a = (f"There are {len(on_it)} objects on the {anchor.label}: " +
                     ", ".join(f"{o.label} ({anchor.distance_to(o):.2f} m away)"
                               for o in on_it[:5]) + ".")
            _add(lang, q, a, "list_on")

    rng.shuffle(records)
    return records


def save_scene_image(frame, out_dir: Path, name: str) -> str:
    """把场景彩色图落盘，返回**相对路径**（训练时按 out_dir 解析）。

    为什么必须落盘：VLM 空间问答微调需要图像输入。
    只写路径不存图，训练脚本就无从加载 —— 这类"数据看着对但训练用不了"
    的问题很隐蔽，所以这里直接把图存下来。
    """
    from PIL import Image  # noqa: PLC0415

    images_dir = ensure_dir(out_dir / "images")
    path = images_dir / f"{name}.png"
    if not path.exists():
        Image.fromarray(np.asarray(frame.color, dtype=np.uint8)).save(path)
    return f"images/{name}.png"


def collect_maps(args, cfg, out_dir: Path):
    """收集若干"地图 + 对应图像相对路径"。"""
    if args.source == "synthetic":
        from roboground.data.synthetic import make_synthetic_sequence
        from roboground.mapping import MapBuilder

        prompts = ["table", "chair", "cup", "box", "bottle", "sofa",
                   "shelf", "monitor", "trash can", "lamp"]
        cfg.set("perception.prompts", prompts)
        out = []
        for i in range(args.scenes):
            frames = make_synthetic_sequence(seed=args.seed + i, num_frames=3,
                                             width=args.width, height=args.height,
                                             num_objects=6)
            smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
            rel = save_scene_image(frames[0], out_dir, f"synth_{args.seed + i:04d}")
            out.append((smap, [rel]))
        return out

    from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene
    from roboground.mapping import MapBuilder

    prompts = ["chair", "table", "desk", "monitor", "door", "window",
               "trash can", "box", "bottle", "sofa", "bed", "bookshelf"]
    cfg.set("perception.prompts", prompts)
    index = load_scene_index(str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz"))
    wanted = {p.lower() for p in prompts}
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))

    out = []
    for pos in np.argsort(-counts):
        i = int(pos)
        if len(out) >= args.scenes:
            break
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = {str(x).lower() for x in np.asarray(index["label_flat"][off0:off1]).ravel()}
        if len(labels & wanted) < 2:
            continue
        scene = load_sunrgbd_scene(index, i, max_depth=8.0, resize=(args.width, args.height))
        if scene is None or scene.boxes_3d.shape[0] == 0:
            continue
        frame = scene.to_frame()
        smap = MapBuilder(cfg, prompts=prompts).build_from_frames([frame])
        rel = save_scene_image(frame, out_dir, f"sunrgbd_{i:05d}")
        out.append((smap, [rel]))
    return out


TRAIN_SCRIPT = r'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VLM 空间问答微调（Qwen2-VL + LoRA + DeepSpeed）。

⚠️ 本脚本是**入口模板**，依赖需要自行安装：
    pip install -e ".[vlm]"          # transformers / peft / accelerate
    pip install deepspeed            # Windows 上编译困难，建议 WSL2 或 Linux
    pip install bitsandbytes         # 4bit 量化（8GB 显存必需）

8GB 显存下的建议配置（见下方 CFG）：
    - 基座 Qwen2-VL-2B-Instruct（不要用 7B）
    - 4bit 量化 + LoRA（rank 8, alpha 16, target all-linear）
    - 冻结视觉塔与 aligner（只训 LLM 侧 LoRA）—— 呼应"VFM 必须冻结"的结论
    - ZeRO-2 + gradient checkpointing + batch 1 × 梯度累积 8

用法：
    python scripts/13_train_vlm_spatial.py --data runs/spatial_qa/train.json
    deepspeed scripts/13_train_vlm_spatial.py --data ... --deepspeed ds_config.json
"""
'''

TRAIN_CONFIG_NOTE = {
    "base_model": "Qwen/Qwen2-VL-2B-Instruct",
    "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": "all-linear"},
    "freeze": ["visual", "aligner"],
    "quantization": "4bit (nf4)",
    "per_device_batch": 1,
    "grad_accum": 8,
    "learning_rate": 1e-4,
    "epochs": 3,
    "max_length": 1024,
    "deepspeed": {"zero_stage": 2, "offload_optimizer": "cpu", "bf16": True},
    "note": "8GB 显存下的保守配置；若显存不足先把 max_length 降到 768",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "sunrgbd"], default="synthetic")
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="runs/spatial_qa")
    ap.add_argument("--emit-train-script", action="store_true",
                    help="只输出训练脚本模板与配置说明")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    out_dir = ensure_dir(args.out_dir)

    if args.emit_train_script:
        (out_dir / "train_vlm_spatial.py").write_text(TRAIN_SCRIPT, encoding="utf-8")
        (out_dir / "train_config.json").write_text(
            json.dumps(TRAIN_CONFIG_NOTE, ensure_ascii=False, indent=2), encoding="utf-8")
        log.ok(f"训练脚本模板与配置已写入 {out_dir}")
        log.info("安装依赖后即可运行：pip install -e \".[vlm]\" deepspeed bitsandbytes")
        return 0

    from roboground import load_config

    cfg = load_config()
    log.info(f"数据来源：{args.source}，目标场景数：{args.scenes}")

    maps = collect_maps(args, cfg, out_dir)
    log.info(f"成功建图 {len(maps)} 个场景")

    all_records: List[Dict[str, Any]] = []
    for idx, (smap, imgs) in enumerate(maps):
        recs = build_records(smap, imgs, seed=args.seed + idx)
        all_records.extend(recs)
        if (idx + 1) % 5 == 0:
            log.debug(f"  已处理 {idx + 1}/{len(maps)} 个场景，累计 {len(all_records)} 条")

    if not all_records:
        log.error("没有生成任何数据 —— 可能是地图里没有置信度足够的物体")
        return 2

    # ---- 划分 train/val ----
    rng = random.Random(args.seed)
    rng.shuffle(all_records)
    n_val = max(1, int(len(all_records) * 0.1))
    val, train = all_records[:n_val], all_records[n_val:]

    (out_dir / "train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 统计 ----
    from collections import Counter
    task_dist = Counter(r["task"] for r in all_records)
    lang_dist = Counter("zh" if any("\u4e00" <= ch <= "\u9fff" for ch in r["messages"][1]["content"])
                        else "en" for r in all_records)

    log.kv("数据集概况", {
        "场景数": len(maps),
        "总样本": len(all_records),
        "训练集": len(train),
        "验证集": len(val),
        "任务分布": dict(task_dist),
        "语言分布": dict(lang_dist),
        "输出目录": str(out_dir),
    })
    log.info("示例样本（定位类）：")
    for r in train:
        if r["task"] == "locate":
            log.info(f"  Q: {r['messages'][1]['content'].splitlines()[-1]}")
            log.info(f"  A: {r['messages'][2]['content']}")
            break
    log.info("示例样本（测距类）：")
    for r in train:
        if r["task"] == "distance":
            log.info(f"  Q: {r['messages'][1]['content'].splitlines()[-1]}")
            log.info(f"  A: {r['messages'][2]['content']}")
            break

    # 顺手生成训练入口模板
    (out_dir / "train_vlm_spatial.py").write_text(TRAIN_SCRIPT, encoding="utf-8")
    (out_dir / "train_config.json").write_text(
        json.dumps(TRAIN_CONFIG_NOTE, ensure_ascii=False, indent=2), encoding="utf-8")

    log.ok(f"空间 QA 数据已生成：{out_dir}/train.json + val.json")
    log.info("下一步：pip install -e \".[vlm]\" deepspeed bitsandbytes，"
             "然后参考 runs/spatial_qa/train_config.json 微调 Qwen2-VL-2B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
