#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""13 · Qwen2-VL 空间问答微调（LoRA + 4bit，8GB 显存可跑）。

配套 `12_gen_spatial_qa.py` 生成的数据。设计要点
----------------------------------------------
1. **4bit 量化**（bitsandbytes nf4）：2B 模型权重从 ~4.4GB 压到 ~1.3GB；
2. **只训 LLM 侧 LoRA，冻结视觉塔与 merger**
   —— 呼应论文里"VFM 必须冻结"的结论：视觉塔的通用表征不该被
   小规模空间问答数据带偏；
3. **梯度检查点 + 梯度累积 + batch=1**：8GB 显存下的必要组合；
4. **不用 HF Trainer**：手写循环，行数更少、对 transformers 版本变更更鲁棒。

用法::

    # 冒烟测试（20 条样本，验证链路能跑）
    python scripts/13_train_vlm_spatial.py --limit 20 --epochs 1 --smoke

    # 正式训练
    python scripts/13_train_vlm_spatial.py --epochs 2 --out runs/vlm_spatial_lora
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.io import ensure_dir, save_json               # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity       # noqa: E402

log = get_logger("train_vlm")

DEFAULT_MODEL = r"G:\RoboGround\weights\Qwen2-VL-2B-Instruct"


def parse_image_paths(content: str, data_root: Path) -> tuple[str, List[str]]:
    """把 `<image>rel/path.png</image>` 抽出来，返回 (纯文本, 绝对路径列表)。"""
    import re

    paths = re.findall(r"<image>(.*?)</image>", content, flags=re.DOTALL)
    text = re.sub(r"<image>.*?</image>", "", content, flags=re.DOTALL).strip()
    abs_paths = []
    for p in paths:
        p = p.strip()
        if not p:
            continue
        q = Path(p)
        abs_paths.append(str(q if q.is_absolute() else (data_root / q)))
    return text, abs_paths


class SpatialQADataset:
    """空间问答数据集：把 ShareGPT 记录转成 Qwen2-VL 的输入。"""

    def __init__(self, records: List[Dict[str, Any]], data_root: Path) -> None:
        self.records = records
        self.data_root = Path(data_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.records[i]
        msgs = rec["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_raw = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        user_text, images = parse_image_paths(user_raw, self.data_root)
        return {
            "system": system, "user": user_text, "assistant": assistant,
            "images": images, "task": rec.get("task", "unknown"),
        }


def build_collator(processor):
    """把 batch 转成模型输入；**只对 assistant 段计算 loss**。

    这里刻意**只编码一次**（把 prompt + answer 拼成完整文本后送进 processor），
    然后用"从尾部反推"的方式定位答案区段来构造 labels。

    为什么不用"分别编码 prompt 和 full"的写法
    ----------------------------------------
    那样要把图像传两遍，Qwen2-VL 的 processor 会各自算一次 `image_grid_thw`，
    一旦两次的 `pixel_values` 排布不完全一致，就会在视觉塔里炸掉：
    `RuntimeError: shape '[-1, 3, 2, 14, 14]' is invalid for input of size 800768`。
    （实测踩过这个坑。）

    "从尾部反推"是精确的：完整序列 = `prompt + answer + eos`，
    所以答案 token 就是倒数第 `len(answer_tokens) + 1` 个到末尾。
    """
    import torch
    from PIL import Image

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts, full_texts, image_lists, n_answer_tokens = [], [], [], []

        for item in batch:
            msgs = []
            if item["system"]:
                msgs.append({"role": "system",
                             "content": [{"type": "text", "text": item["system"]}]})
            user_content: List[Dict[str, Any]] = []
            imgs = []
            for p in item["images"]:
                try:
                    img = Image.open(p).convert("RGB")
                    user_content.append({"type": "image", "image": img})
                    imgs.append(img)
                except Exception:
                    continue
            user_content.append({"type": "text", "text": item["user"]})
            msgs.append({"role": "user", "content": user_content})

            prompt_text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            texts.append(prompt_text)
            image_lists.append(imgs)
            full_texts.append(prompt_text + item["assistant"] + processor.tokenizer.eos_token)
            # 答案段长度（不含特殊 token）—— 用于反推 labels 的屏蔽边界
            n_answer_tokens.append(
                len(processor.tokenizer(item["assistant"], add_special_tokens=False)["input_ids"])
            )

        # ---- 逐样本编码，再手动 padding ----
        # 为什么不用 processor 的批处理：当 images 是"列表的列表"时，
        # Qwen2-VL 的 processor 会把 pixel_values 与 image_grid_thw 配错，
        # 视觉塔里报 `shape '[-1,3,2,14,14]' is invalid for input of size ...`。
        # 逐样本编码 + 手动拼，虽然代码多几行，但结果一定自洽。
        per_sample = []
        for i, (full_text, imgs) in enumerate(zip(full_texts, image_lists)):
            kwargs: Dict[str, Any] = {"text": [full_text], "return_tensors": "pt"}
            if imgs:
                kwargs["images"] = [imgs[0]]        # 一次一张图，避免嵌套
            enc = processor(**kwargs)
            per_sample.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "pixel_values": enc.get("pixel_values"),
                "image_grid_thw": enc.get("image_grid_thw"),
                "n_answer": n_answer_tokens[i],
            })

        max_len = max(s["input_ids"].shape[0] for s in per_sample)
        pad_id = processor.tokenizer.pad_token_id or 0

        input_ids = torch.full((len(per_sample), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(per_sample), max_len), dtype=torch.long)
        labels = torch.full((len(per_sample), max_len), -100, dtype=torch.long)

        pv_list, thw_list = [], []
        for i, s in enumerate(per_sample):
            n = s["input_ids"].shape[0]
            input_ids[i, :n] = s["input_ids"]
            attn[i, :n] = s["attention_mask"]
            # 只保留最后 n_answer + 1 个 token（答案 + eos），其余屏蔽
            keep_from = max(0, n - (s["n_answer"] + 1))
            labels[i, keep_from:n] = s["input_ids"][keep_from:]
            if s["pixel_values"] is not None:
                pv_list.append(s["pixel_values"])
            if s["image_grid_thw"] is not None:
                thw_list.append(s["image_grid_thw"])

        inputs: Dict[str, Any] = {
            "input_ids": input_ids, "attention_mask": attn, "labels": labels,
        }
        if pv_list:
            inputs["pixel_values"] = torch.cat(pv_list, dim=0)
            inputs["image_grid_thw"] = torch.cat(thw_list, dim=0)
        return inputs

    return collate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="runs/spatial_qa/train.json")
    ap.add_argument("--val", default="runs/spatial_qa/val.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="runs/vlm_spatial_lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（调试）")
    ap.add_argument("--no-4bit", action="store_true", help="关闭 4bit 量化（费显存）")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：少量步数后停止")
    ap.add_argument("--max-steps", type=int, default=0, help="硬性限制优化步数")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor

    model_path = Path(args.model)
    if not model_path.exists():
        log.error(f"模型目录不存在：{model_path}")
        log.info("请先下载：python -c \"from huggingface_hub import snapshot_download; "
                 "snapshot_download('Qwen/Qwen2-VL-2B-Instruct', "
                 "local_dir=r'G:\\RoboGround\\weights\\Qwen2-VL-2B-Instruct')\"")
        return 2

    data_path = Path(args.data)
    if not data_path.exists():
        log.error(f"训练数据不存在：{data_path}")
        log.info("请先生成：python scripts/12_gen_spatial_qa.py --source synthetic --scenes 20")
        return 2

    data_root = data_path.parent
    records = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]
    val_records = []
    if Path(args.val).exists():
        val_records = json.loads(Path(args.val).read_text(encoding="utf-8"))[: max(8, len(records) // 10)]

    log.info(f"训练样本 {len(records)} 条，验证 {len(val_records)} 条，图像根目录 {data_root}")

    # ---------------- 模型 ----------------
    log.info(f"加载 {model_path.name}（4bit={not args.no_4bit}）...")
    t0 = time.time()
    quant_cfg = None
    torch_dtype = torch.bfloat16
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype = torch.bfloat16

    try:
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path), quantization_config=quant_cfg,
            torch_dtype=torch_dtype, device_map={"": 0},
            attn_implementation="eager",
        )
    except Exception as exc:
        log.error(f"模型加载失败：{type(exc).__name__}: {exc}")
        return 2

    processor = AutoProcessor.from_pretrained(str(model_path), max_pixels=640 * 480)
    log.ok(f"模型就绪，耗时 {time.time() - t0:.1f}s，"
           f"显存 {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ---------------- LoRA：只训 LLM 侧，冻结视觉塔 ----------------
    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    # 冻结视觉塔 / merger —— 视觉表征不该被小规模空间数据带偏
    frozen = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ("visual", "merger")):
            param.requires_grad = False
            frozen += 1
    log.info(f"已冻结视觉塔与 merger 参数组（{frozen} 个张量）")

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.kv("LoRA 配置", {
        "r": args.lora_r, "alpha": args.lora_alpha,
        "可训练参数": f"{trainable / 1e6:.2f} M",
        "总参数": f"{total / 1e6:.1f} M",
        "可训练占比": f"{trainable / max(total, 1):.4%}",
    })
    model.print_trainable_parameters()

    # ---------------- 数据 ----------------
    collate = build_collator(processor)
    train_ds = SpatialQADataset(records, data_root)
    val_ds = SpatialQADataset(val_records, data_root)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if args.smoke:
        total_steps = min(total_steps, 6)
    log.info(f"计划优化步数：{total_steps}（每 epoch {steps_per_epoch} 步，"
             f"batch={args.batch_size} × accum={args.grad_accum}）")

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        warmup = max(1, int(total_steps * 0.05))
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ---------------- 训练循环 ----------------
    out_dir = ensure_dir(args.out)
    model.train()
    global_step = 0
    history: List[Dict[str, float]] = []
    accum_loss = 0.0
    t_start = time.time()
    stop = False

    for epoch in range(args.epochs):
        if stop:
            break
        batch_buf: List[Dict[str, Any]] = []
        for i in range(len(train_ds)):
            batch_buf.append(train_ds[i])
            if len(batch_buf) < args.batch_size:
                continue

            try:
                enc = collate(batch_buf)
                enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc.items()}
                if enc["input_ids"].shape[1] > args.max_length:
                    enc = {k: (v[:, : args.max_length] if v.dim() == 2 else v)
                           for k, v in enc.items()}
                out = model(**enc)
                loss = out.loss / args.grad_accum
                loss.backward()
                accum_loss += float(out.loss.detach())
            except torch.cuda.OutOfMemoryError:
                log.warn("显存不足，跳过该 batch（建议减小 --max-length）")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                batch_buf = []
                continue

            batch_buf = []
            if ((i + 1) // args.batch_size) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 5 == 0 or global_step == 1:
                    elapsed = time.time() - t_start
                    mem = torch.cuda.memory_allocated() / 1e9
                    log.info(f"  step {global_step}/{total_steps}  "
                             f"loss={accum_loss / 5:.4f}  "
                             f"lr={scheduler.get_last_lr()[0]:.2e}  "
                             f"mem={mem:.2f}GB  {elapsed:.0f}s")
                    history.append({"step": global_step, "loss": accum_loss / 5,
                                    "lr": scheduler.get_last_lr()[0], "mem_gb": mem})
                    accum_loss = 0.0

                if global_step >= total_steps:
                    stop = True
                    break

        # 每个 epoch 存一次
        ckpt = out_dir / f"epoch{epoch + 1}"
        model.save_pretrained(str(ckpt))
        processor.save_pretrained(str(ckpt))
        log.ok(f"已保存 LoRA 权重：{ckpt}")

    # ---------------- 收尾 ----------------
    elapsed = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 1e9
    summary = {
        "train_samples": len(records),
        "val_samples": len(val_records),
        "epochs": args.epochs,
        "global_steps": global_step,
        "elapsed_s": round(elapsed, 1),
        "final_loss": history[-1]["loss"] if history else None,
        "first_loss": history[0]["loss"] if history else None,
        "peak_vram_gb": round(peak, 3),
        "trainable_params_m": round(trainable / 1e6, 3),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
        "quantization": "none" if args.no_4bit else "4bit-nf4",
        "frozen": ["visual", "merger"],
        "history": history,
        "output_dir": str(out_dir),
    }
    save_json(summary, out_dir / "train_summary.json")

    log.info("=" * 60)
    log.kv("训练汇总", {k: v for k, v in summary.items() if k != "history"})
    if summary["first_loss"] and summary["final_loss"]:
        drop = summary["first_loss"] - summary["final_loss"]
        log.info(f"loss 变化：{summary['first_loss']:.4f} → {summary['final_loss']:.4f} "
                 f"（{'下降' if drop > 0 else '上升'} {abs(drop):.4f}）")
    log.ok(f"LoRA 权重与训练日志已保存到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
