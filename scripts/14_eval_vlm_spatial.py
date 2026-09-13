#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""14 · VLM 空间问答推理评测（微调 vs 未微调基座）。

为什么这个评测的设计本身很重要
============================
我训完 LoRA 之后意识到一个**必须先想清楚的问题**：

我们的 GT 答案里带**米制距离**（"中心距 0.42 米"），这些数值来自 3D 语义地图
（有深度）；而 VLM 只看到**单张 RGB**。

> **从单张 RGB 恢复米制距离在数学上是不可解的**（尺度歧义），
> 除非模型学到了很强的深度先验。所以"VLM 答不准米制数值"不是训练不足，
> 而是**任务本身超出了模态能力边界**。

这个认知反而**正好论证了本项目的架构**：
**语义归 VLM，度量归几何** —— 这也正是我在设计文档里写的"VLM 不该替代几何"。

所以评测必须**分层**，否则会得出错误结论：

| 任务类型 | 需要什么 | VLM 能否做到 | 预期 |
|---|---|---|---|
| 相对关系（杯子在桌子上面吗） | 纯视觉 | ✅ 可以 | 微调应有提升 |
| 物体存在性（有杯子吗） | 纯视觉 | ✅ 可以 | 微调应有提升 |
| **米制距离**（离多远） | **深度** | ❌ 不可解 | 微调也难，误差大 |
| **3D 坐标**（在哪） | **深度 + 标定** | ❌ 不可解 | 微调也难，误差大 |

评测就把这四类分开报，并对比三条基线：
1. **未微调基座**（Qwen2-VL-2B 原版）
2. **微调后**（+ LoRA）
3. （参考上界）规则引擎 —— 它有 3D 地图，米制数值是精确的

用法::

    # 评测微调后的模型
    python scripts/14_eval_vlm_spatial.py --adapter runs/vlm_spatial_lora_full/epoch2

    # 只评测未微调基座（对比基线）
    python scripts/14_eval_vlm_spatial.py --adapter none

    # 两者都跑并输出对比表
    python scripts/14_eval_vlm_spatial.py --compare
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from roboground.utils.io import ensure_dir, save_json           # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity   # noqa: E402

log = get_logger("eval_vlm")

DEFAULT_MODEL = r"G:\RoboGround\weights\Qwen2-VL-2B-Instruct"

# 数值解析：中英文 + 米/m
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_numbers(text: str) -> List[float]:
    """把文本里的数字抠出来（用于算米制误差）。"""
    out = []
    for m in _NUM.finditer(text):
        try:
            out.append(float(m.group()))
        except ValueError:
            continue
    return out


def parse_coords(text: str) -> Optional[List[float]]:
    """从 "(-0.14, 2.00, 0.16)" 这类文本里抠出 3D 坐标。"""
    m = re.search(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)", text)
    if not m:
        return None
    return [float(m.group(1)), float(m.group(2)), float(m.group(3))]


def parse_yesno(text: str) -> Optional[bool]:
    """判断回答是肯定还是否定（中英）。"""
    t = text.strip().lower()
    # 否定优先（"不是" 里也含 "是"，必须先判否定）
    if re.search(r"^(no|不对|不是|没有|否)\b", t) or t.startswith("不是") or t.startswith("不对"):
        return False
    if re.search(r"^(yes|对|是的|是|有)\b", t) or t.startswith("是的") or t.startswith("对"):
        return True
    if "不是" in t[:6] or "不对" in t[:6]:
        return False
    if "是的" in t[:6] or "对的" in t[:6]:
        return True
    return None


def extract_gt(gt_text: str, task: str) -> Dict[str, Any]:
    """从 GT 答案里抽出可比对的目标值。"""
    out: Dict[str, Any] = {"raw": gt_text}
    if task in ("distance",):
        nums = parse_numbers(gt_text)
        out["numbers"] = nums
        out["primary"] = nums[0] if nums else None          # 第一个数通常是中心距
    elif task == "locate":
        out["coords"] = parse_coords(gt_text)
    elif task == "relation":
        out["yes"] = True                                   # GT 生成时一律是肯定式陈述
    elif task == "list_on":
        out["numbers"] = parse_numbers(gt_text)
    return out


def extract_pred(pred_text: str, task: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"raw": pred_text}
    if task == "distance":
        nums = parse_numbers(pred_text)
        out["numbers"] = nums
        out["primary"] = nums[0] if nums else None
    elif task == "locate":
        out["coords"] = parse_coords(pred_text)
        if out["coords"] is None:
            nums = parse_numbers(pred_text)
            if len(nums) >= 3:
                out["coords"] = nums[:3]
    elif task == "relation":
        out["yes"] = parse_yesno(pred_text)
    elif task == "list_on":
        out["numbers"] = parse_numbers(pred_text)
    return out


def score_sample(gt: Dict[str, Any], pred: Dict[str, Any], task: str) -> Dict[str, float]:
    """单条样本的打分。返回若干 0/1 或误差值。"""
    res: Dict[str, float] = {}
    text = pred.get("raw", "") or ""
    res["nonempty"] = float(bool(text.strip()))
    res["mentions_unit"] = float(bool(re.search(r"米|\bm\b|meter", text)))

    if task == "distance":
        g, p = gt.get("primary"), pred.get("primary")
        res["parseable"] = float(p is not None)
        if g is not None and p is not None:
            res["abs_err"] = abs(p - g)
            res["rel_err"] = abs(p - g) / max(abs(g), 1e-6)
            res["within_0.1m"] = float(abs(p - g) <= 0.10)
            res["within_0.3m"] = float(abs(p - g) <= 0.30)
    elif task == "locate":
        gc, pc = gt.get("coords"), pred.get("coords")
        res["parseable"] = float(pc is not None)
        if gc is not None and pc is not None:
            res["coord_err"] = float(np.linalg.norm(np.asarray(gc) - np.asarray(pc)))
            res["within_0.3m"] = float(res["coord_err"] <= 0.30)
    elif task == "relation":
        gy, py_ = gt.get("yes"), pred.get("yes")
        res["parseable"] = float(py_ is not None)
        if gy is not None and py_ is not None:
            res["correct"] = float(py_ == gy)
    elif task == "list_on":
        res["parseable"] = float(bool(pred.get("numbers")))

    return res


def build_chat(processor, system: str, question: str, image_path: Path):
    from PIL import Image

    msgs = []
    if system:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    content: List[Dict[str, Any]] = []
    if image_path.exists():
        content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})
    content.append({"type": "text", "text": question})
    msgs.append({"role": "user", "content": content})
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model(model_path: str, adapter: Optional[str]):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, quantization_config=quant,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=640 * 480)

    if adapter and adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        log.ok(f"已加载 LoRA 适配器：{adapter}")
    else:
        log.info("使用未微调基座（对比基线）")
    model.eval()
    return model, processor


def run_eval(args, val_records: List[Dict[str, Any]], data_root: Path,
             adapter: Optional[str]) -> Dict[str, Any]:
    import torch

    model, processor = load_model(args.model, adapter)
    tag = "finetuned" if (adapter and adapter != "none") else "base"
    log.info(f"[{tag}] 开始评测 {len(val_records)} 条样本")

    per_task: Dict[str, List[Dict[str, float]]] = {}
    t0 = time.time()

    for i, rec in enumerate(val_records):
        msgs = rec["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_raw = next((m["content"] for m in msgs if m["role"] == "user"), "")
        gt_text = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        task = rec.get("task", "unknown")

        imgs = re.findall(r"<image>(.*?)</image>", user_raw, flags=re.DOTALL)
        question = re.sub(r"<image>.*?</image>", "", user_raw, flags=re.DOTALL).strip()
        img_path = Path(imgs[0]) if imgs else Path("")
        if imgs and not img_path.is_absolute():
            img_path = data_root / img_path

        prompt = build_chat(processor, system, question, img_path)
        if img_path.exists():
            from PIL import Image
            inputs = processor(text=[prompt],
                               images=[Image.open(img_path).convert("RGB")],
                               return_tensors="pt")
        else:
            inputs = processor(text=[prompt], return_tensors="pt")
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(gen, skip_special_tokens=True)[0]

        sc = score_sample(extract_gt(gt_text, task), extract_pred(text, task), task)
        per_task.setdefault(task, []).append(sc)

        if (i + 1) % 10 == 0 or (i + 1) == len(val_records):
            el = time.time() - t0
            log.debug(f"  [{tag}] {i + 1}/{len(val_records)}  {el:.0f}s "
                      f"({el / (i + 1):.2f}s/条)")

    elapsed = time.time() - t0

    # ---- 汇总 ----
    summary: Dict[str, Any] = {"tag": tag, "adapter": adapter,
                               "n_samples": len(val_records),
                               "elapsed_s": round(elapsed, 1),
                               "sec_per_sample": round(elapsed / max(len(val_records), 1), 2),
                               "per_task": {}}

    for task, rows in per_task.items():
        agg: Dict[str, float] = {"n": float(len(rows))}
        keys = set().union(*[set(r.keys()) for r in rows])
        for k in keys:
            vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
            if vals:
                agg[k] = float(np.mean(vals))
        summary["per_task"][task] = agg

    # 全局：格式合规率（所有任务）
    all_rows = [r for rows in per_task.values() for r in rows]
    summary["overall"] = {
        "n": float(len(all_rows)),
        "nonempty_rate": float(np.mean([r.get("nonempty", 0.0) for r in all_rows])),
        "parseable_rate": float(np.mean([r.get("parseable", 0.0) for r in all_rows])),
        "unit_rate": float(np.mean([r.get("mentions_unit", 0.0) for r in all_rows])),
    }
    return summary


def question_of(rec: Dict[str, Any]) -> str:
    """取样本的问题文本（去掉 <image> 占位）。"""
    user_raw = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
    return re.sub(r"<image>.*?</image>", "", user_raw, flags=re.DOTALL).strip()


def sample_key(rec: Dict[str, Any]) -> Tuple[str, str]:
    """样本唯一键 = (图像相对路径, 问题文本)。

    ⚠️ **不能只用问题文本当键。** 同一句话（如 "Is the monitor located to the
    left of the cup?"）会在多个合成场景里重复出现，但答案不同（物体位置不同）。
    只用问题文本会让后面的场景覆盖前面的，匹配到错误的答案 ——
    这正是本基线第一版跑出来"距离 MAE 0.178 m、坐标误差 1.001 m"的原因：
    一个本该满分（GT 就是规则引擎生成的）的 oracle 基线被键冲突污染了。
    """
    user_raw = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
    imgs = re.findall(r"<image>(.*?)</image>", user_raw, flags=re.DOTALL)
    img = imgs[0].strip() if imgs else ""
    return (img, question_of(rec))


def _load_generator():
    """加载 `12_gen_spatial_qa.py`（文件名以数字开头，不能直接 import）。"""
    import importlib.util

    path = Path(__file__).resolve().parent / "12_gen_spatial_qa.py"
    spec = importlib.util.spec_from_file_location("_gen_spatial_qa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_rule_answers(args) -> Dict[Tuple[str, str], str]:
    """重建生成 QA 时的场景，用**规则引擎在精确 3D 地图上**回答同一批问题。

    ⚠️ **这是 oracle 上界，不是独立系统。** GT 答案本来就是这套规则引擎
    在 3D 地图上生成的 —— 所以它**应当**接近满分（前提是匹配键正确）。
    它的价值**不是**"证明规则引擎强"，而是**给出一个天花板**，
    让 VLM 的差距可以归因到具体维度：

    - 若 VLM 在**语义/格式**维度追平上界（关系判断、可解析率、单位）
      而**度量**维度差一个数量级 → 说明瓶颈不在语言能力，而在
      "单目 RGB 恢复不出绝对尺度"这个**病态任务**本身；
    - 这正是本项目"**几何走投影、语义走 VLM**"架构决策的定量依据。

    做法：用与 `12_gen_spatial_qa.py` 完全相同的参数（同样确定性种子）
    重建场景与地图，重新跑一遍 `build_records`，再按 **(图像, 问题)** 匹配回 val 样本。
    匹配率必须接近 100% —— 否则说明生成过程不确定，那本身就是个发现。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from roboground import load_config
    from roboground.data.synthetic import make_synthetic_sequence
    from roboground.mapping import MapBuilder

    gen = _load_generator()
    prompts = ["table", "chair", "cup", "box", "bottle", "sofa",
               "shelf", "monitor", "trash can", "lamp"]
    cfg = load_config()
    cfg.set("perception.prompts", prompts)

    answers: Dict[Tuple[str, str], str] = {}
    for i in range(args.rule_scenes):
        seed_i = args.rule_seed + i
        frames = make_synthetic_sequence(seed=seed_i, num_frames=3,
                                         width=args.rule_width,
                                         height=args.rule_height, num_objects=6)
        smap = MapBuilder(cfg, prompts=prompts).build_from_frames(frames)
        recs = gen.build_records(smap, [f"images/synth_{seed_i:04d}.png"], seed=seed_i)
        for r in recs:
            answers[sample_key(r)] = next(
                (m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
    return answers


def run_rule_baseline(args, val_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """用规则引擎（精确地图）回答 val 集，按与 VLM 完全相同的口径打分。"""
    t0 = time.time()
    answers = build_rule_answers(args)
    log.info(f"[rule·oracle] 重建出 {len(answers)} 条规则答案")

    per_task: Dict[str, List[Dict[str, float]]] = {}
    matched = 0
    missing: List[str] = []
    for rec in val_records:
        key = sample_key(rec)
        pred_text = answers.get(key)
        if pred_text is None:
            missing.append(key[1])
            pred_text = ""            # 匹配不上 → 当作空回答（会被记成不可解析）
        else:
            matched += 1
        task = rec.get("task", "unknown")
        gt_text = next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), "")
        sc = score_sample(extract_gt(gt_text, task), extract_pred(pred_text, task), task)
        per_task.setdefault(task, []).append(sc)

    match_rate = matched / max(len(val_records), 1)
    if match_rate < 0.99:
        log.warn(f"[rule·oracle] 问题匹配率只有 {match_rate:.1%} —— "
                 f"生成过程可能不确定（未匹配 {len(missing)} 条，例：{missing[:2]}）")

    summary: Dict[str, Any] = {
        "tag": "rule·oracle", "adapter": None,
        "n_samples": len(val_records), "match_rate": round(match_rate, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "per_task": {},
    }
    for task, rows in per_task.items():
        agg: Dict[str, float] = {"n": float(len(rows))}
        keys = set().union(*[set(r.keys()) for r in rows])
        for k in keys:
            vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
            if vals:
                agg[k] = float(np.mean(vals))
        summary["per_task"][task] = agg
    all_rows = [r for rows in per_task.values() for r in rows]
    summary["overall"] = {
        "n": float(len(all_rows)),
        "nonempty_rate": float(np.mean([r.get("nonempty", 0.0) for r in all_rows])),
        "parseable_rate": float(np.mean([r.get("parseable", 0.0) for r in all_rows])),
        "unit_rate": float(np.mean([r.get("mentions_unit", 0.0) for r in all_rows])),
    }
    return summary


def print_report(results: List[Dict[str, Any]], args) -> None:
    """打印对比表 —— 这是整个脚本的核心产出。"""
    print()
    print("=" * 84)
    print("VLM 空间问答评测结果")
    print("=" * 84)

    tasks = sorted({t for r in results for t in r["per_task"]})
    task_zh = {"locate": "物体定位(3D坐标)", "relation": "相对关系(是/否)",
               "distance": "米制距离", "list_on": "区域列举"}
    needs_depth = {"locate", "distance", "list_on"}

    for task in tasks:
        print()
        marker = "【需深度·模态超纲】" if task in needs_depth else "【纯视觉·可学】"
        print(f"── {task_zh.get(task, task)}  {marker} ──")
        header = f"  {'模型':<12}{'n':>4}{'可解析率':>10}"
        if task == "distance":
            header += f"{'距离MAE(m)':>13}{'≤0.3m':>9}"
        elif task == "locate":
            header += f"{'坐标误差(m)':>13}{'≤0.3m':>9}"
        elif task == "relation":
            header += f"{'关系准确率':>12}"
        else:
            header += f"{'可解析率':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in results:
            agg = r["per_task"].get(task)
            if not agg:
                continue
            line = f"  {r['tag']:<12}{int(agg['n']):>4}{agg.get('parseable', 0):>10.1%}"
            if task == "distance":
                line += f"{agg.get('abs_err', float('nan')):>13.3f}{agg.get('within_0.3m', 0):>9.1%}"
            elif task == "locate":
                line += f"{agg.get('coord_err', float('nan')):>13.3f}{agg.get('within_0.3m', 0):>9.1%}"
            elif task == "relation":
                line += f"{agg.get('correct', float('nan')):>12.1%}"
            else:
                line += f"{agg.get('parseable', 0):>10.1%}"
            print(line)

    print()
    print("── 全局格式合规率 ──")
    for r in results:
        o = r["overall"]
        print(f"  {r['tag']:<12} 非空={o['nonempty_rate']:.1%}  "
              f"可解析={o['parseable_rate']:.1%}  含单位={o['unit_rate']:.1%}")

    print()
    print("=" * 84)
    print("结论解读")
    print("=" * 84)
    print("""
  1) **纯视觉任务**（相对关系）应看到微调带来的提升 —— 这是微调真正学到的东西。
  2) **米制任务**（距离/坐标）即使微调后误差仍然很大，这**不是训练失败**，
     而是单张 RGB 在数学上无法恢复绝对尺度（尺度歧义）。
     → 这正是本项目架构的论据：**语义归 VLM，度量归几何**。
  3) 工程上正确的用法是：
     VLM 负责"把自然语言解析成结构化意图 + 指代消解"，
     几何/规则引擎负责"给出精确的米制数值"，两者通过结构化输出对接
     （本项目 `reasoning/vlm.py` 的 HybridReasoner 就是这么做的）。
""")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="runs/spatial_qa/val.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default="runs/vlm_spatial_lora_full/epoch2",
                    help="LoRA 路径；传 none 表示评测未微调基座")
    ap.add_argument("--compare", action="store_true", help="同时评测基座与微调后")
    ap.add_argument("--limit", type=int, default=48, help="评测样本数（VLM 生成较慢）")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="只评测指定任务类型，如 relation distance")
    ap.add_argument("--out", default="runs/vlm_eval.json")
    # ---- 规则引擎（oracle 上界）基线 ----
    ap.add_argument("--rule-baseline", action="store_true",
                    help="额外评测规则引擎在**精确 3D 地图**上的表现（oracle 上界，不需要模型）")
    ap.add_argument("--rule-baseline-only", action="store_true",
                    help="只跑规则引擎基线，不加载 VLM（几秒钟出结果）")
    ap.add_argument("--rule-scenes", type=int, default=20,
                    help="重建场景数，必须与 12_gen_spatial_qa.py 一致")
    ap.add_argument("--rule-seed", type=int, default=7,
                    help="重建种子，必须与 12_gen_spatial_qa.py 一致，否则匹配率会掉")
    ap.add_argument("--rule-width", type=int, default=640)
    ap.add_argument("--rule-height", type=int, default=480)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    if args.rule_baseline_only:
        args.rule_baseline = True

    if not Path(args.model).exists() and not args.rule_baseline_only:
        log.error(f"模型不存在：{args.model}")
        return 2

    data_path = Path(args.data)
    if not data_path.exists():
        log.error(f"验证数据不存在：{data_path}")
        log.info("请先：python scripts/12_gen_spatial_qa.py --source synthetic --scenes 20")
        return 2

    records = json.loads(data_path.read_text(encoding="utf-8"))
    if args.tasks:
        keep = set(args.tasks)
        records = [r for r in records if r.get("task") in keep]
    # 按任务分层抽样，保证四类都有
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_task.setdefault(r.get("task", "unknown"), []).append(r)
    per_task_n = max(1, args.limit // max(len(by_task), 1))
    sampled: List[Dict[str, Any]] = []
    for t, rows in sorted(by_task.items()):
        sampled.extend(rows[:per_task_n])
    log.info(f"评测样本：{len(sampled)} 条（原始 {len(records)}），"
             f"任务分布 " + str({t: min(len(v), per_task_n) for t, v in sorted(by_task.items())}))

    data_root = data_path.parent
    results: List[Dict[str, Any]] = []

    adapters = [None] if not args.compare else [None, args.adapter]
    if not args.compare and args.adapter and args.adapter != "none":
        adapters = [args.adapter]
    if args.rule_baseline_only:
        adapters = []                 # 只跑规则引擎，不加载模型

    for adapter in adapters:
        r = run_eval(args, sampled, data_root, adapter)
        results.append(r)

    # ---- 规则引擎（oracle 上界）基线 ----
    # 放在 VLM 之后跑，且**不需要模型** —— 它纯粹用重建的 3D 地图回答。
    if args.rule_baseline:
        results.append(run_rule_baseline(args, sampled))

    print_report(results, args)

    if args.out:
        ensure_dir(Path(args.out).parent)
        save_json({"results": results, "config": vars(args)}, args.out)
        log.ok(f"评测结果已保存：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
