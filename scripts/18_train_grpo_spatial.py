#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""18 · GRPO + 规则引擎可验证奖励（RLVR）训练空间问答。

核心思想
=======
RLVR（RL with Verifiable Rewards）在数学/代码领域能work，是因为有**确定的答案校验器**。
多模态空间推理通常没有 —— 只能训 reward model，而 reward model 会带来 reward hacking。

**但本项目有一样别人没有的东西**：几何规则引擎能从 3D 语义地图算出**精确答案**。
于是"VLM 输出的空间答案对不对"可以**被程序精确判定**：

    3D 地图 ──规则引擎──> 精确真值 (1.70, 0.00, 0.85)
    VLM 输出 "杯子在 (1.74, 0.04, 0.85)"
    reward = exp(-||Δp||/τ) = 0.96          ← 可验证、不可 hack、无需训 reward model

这就是把"主观生成 RL 的 reward 怎么设计"落地的答案：
**能自动验证的任务，就不要训 reward model。**

本脚本有两种模式
==============
- `--dry-run`：用 `FakePolicy` 离线跑通整条 RL 循环（**不用 GPU、几秒钟**），
  用来验证组内优势 / IS 比率 / clip / KL 是否接对了。
  这是"先证明逻辑对，再花 GPU 时间"的做法。
- 默认：加载 Qwen2-VL-2B + LoRA，跑真实 GRPO。

用法::

    python scripts/18_train_grpo_spatial.py --dry-run            # 离线自检
    python scripts/18_train_grpo_spatial.py --steps 20 --g 4     # 真训练
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.rl import (                                          # noqa: E402
    FakePolicy, GrpoConfig, GrpoTrainer, RewardConfig, SpatialOracle,
    TrainerConfig, build_rl_samples, group_advantages, reward_stats,
    spatial_reward,
)
from roboground.utils.io import ensure_dir                            # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity        # noqa: E402

log = get_logger("train_grpo")

DEFAULT_MODEL = r"G:\RoboGround\weights\Qwen2-VL-2B-Instruct"
PROMPTS = ["table", "chair", "cup", "box", "bottle", "sofa",
           "shelf", "monitor", "trash can", "lamp"]

SYSTEM_PROMPT = (
    "你是服务机器人的空间感知模块。你会看到一张室内场景照片，并被问到空间问题。\n"
    "回答要求：**只输出一个 JSON 对象**，不要任何解释文字。\n"
    "  · 问相对位置 → {\"answer\": \"yes\"} 或 {\"answer\": \"no\"}\n"
    "  · 问距离 → {\"distance_m\": <数字>}\n"
    "  · 问坐标 → {\"coords\": [x, y, z]}\n"
    "  · 问列举 → {\"labels\": [\"...\"], \"count\": <整数>}\n"
    "所有长度单位为米，保留两位小数。"
)


# =============================================================================
# 数据：从 3D 地图构造带**结构化 oracle** 的 RL 样本
# =============================================================================
def build_dataset(args) -> List[Any]:
    from roboground import load_config
    from roboground.data.synthetic import make_synthetic_sequence
    from roboground.mapping import MapBuilder

    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("perception.prompts", PROMPTS)
    cfg.set("project.verbose", False)

    out_dir = Path("runs/rl_spatial/images")
    ensure_dir(out_dir)

    samples: List[Any] = []
    for i in range(args.scenes):
        seed = args.seed + i
        frames = make_synthetic_sequence(seed=seed, num_frames=3,
                                         width=args.width, height=args.height,
                                         num_objects=args.objects)
        smap = MapBuilder(cfg, prompts=PROMPTS).build_from_frames(frames)
        if not smap.objects:
            continue
        name = f"rl_{seed:04d}.png"
        _save_image(frames[0], out_dir / name)
        rel = f"images/{name}"
        recs = build_rl_samples(smap, rel, seed=seed,
                                max_per_task=args.max_per_task,
                                system_prompt=SYSTEM_PROMPT)
        samples.extend(recs)
    log.info(f"数据集：{len(samples)} 条 RL 样本"
             f"（任务分布 {_task_counts(samples)}）")
    return samples


def _task_counts(samples: Sequence[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in samples:
        out[s.oracle.task] = out.get(s.oracle.task, 0) + 1
    return out


def _save_image(frame, path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(frame.color, dtype=np.uint8)).save(path)


# =============================================================================
# 真策略：Qwen2-VL-2B + LoRA（4bit）
# =============================================================================
class QwenVlPolicy:
    """Qwen2-VL-2B + LoRA 策略。

    关键实现点
    ----------
    1. **参考策略 = 关掉 adapter**，不额外加载一份模型
       （4bit 下 2B 约 1.8GB，再加载一份直接翻倍）。
    2. **逐 token log 概率**：前向算 logits，用 `log_softmax` 取
       生成 token 位置的值。注意要**右移一位**对齐因果语言模型的标签。
    3. 生成时用 `do_sample=True + temperature` —— RL 需要**多样性**，
       贪心解码会让一组 G 个样本完全相同 → 组内优势恒 0 → 学不动。
    """

    def __init__(self, model_path: str, *, lora_r: int = 8, lr: float = 1e-5,
                 device: str = "cuda", init_adapter: Optional[str] = None) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        self.torch = torch
        self.device = device
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, quantization_config=bnb, torch_dtype=torch.float16,
            device_map={"": 0} if device == "cuda" else None,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = prepare_model_for_kbit_training(self.model)
        # ★ `init_adapter`：从 **SFT 好的 LoRA** 继续，而不是从基座开始。
        #
        # 为什么这一步是必需的（实测踩出来的）：
        # 直接拿基座跑 GRPO，**每一步的组内优势都是 0、梯度恒为 0** ——
        # 因为基座没在这个输出格式上训过，采样 G 次得到的是**同一种答案**，
        # 组内奖励完全同分 → 优势退化 → 学不动。
        #
        # 这就是面试里问的「GRPO 的**冷启动问题**」：
        # 策略必须先有"会做但不确定"的分布，才能靠组内相对好坏来优化。
        # 标准管线因此是 **SFT → RL** 两段式，而不是直接对基座做 RL。
        if init_adapter and Path(init_adapter).exists():
            self.model = PeftModel.from_pretrained(
                self.model, init_adapter, is_trainable=True)
            log.info(f"从 SFT adapter 继续：{init_adapter}")
        else:
            lcfg = LoraConfig(
                r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.0, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lcfg)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.reference_mode = False
        self.pending_grads: List[Tuple[Any, Any]] = []

    # ---- 内部：把样本编码成模型输入 ----
    def _encode(self, sample, response: Optional[str]):
        from PIL import Image

        img_path = Path("runs/rl_spatial") / sample.image_path
        messages = []
        if sample.system_prompt:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": sample.system_prompt}]})
        content = []
        if img_path.exists():
            content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})
        content.append({"type": "text", "text": sample.question})
        messages.append({"role": "user", "content": content})
        if response is not None:
            messages.append({"role": "assistant",
                             "content": [{"type": "text", "text": response}]})
        return messages, img_path

    def generate(self, samples, *, temperature: float = 1.0,
                 max_new_tokens: int = 48) -> List[str]:
        torch = self.torch
        out: List[str] = []
        self.model.eval()
        for s in samples:
            messages, img_path = self._encode(s, None)
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images = None
            if img_path.exists():
                from PIL import Image
                images = [Image.open(img_path).convert("RGB")]
            inputs = self.processor(text=[prompt], images=images,
                                    return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=max(temperature, 1e-2), top_p=0.95,
                    # ★★ `top_k=0` **必须显式给**，否则整个 RL 训练完全无效。
                    #
                    # Qwen2-VL 自带的 `generation_config.json` 是：
                    #     {"do_sample": true, "temperature": 0.01,
                    #      "top_p": 0.001, "top_k": 1}
                    # `top_k=1` = **只保留概率最高的那一个 token**，
                    # 采样直接退化成贪心解码，而且传 temperature / top_p
                    # **覆盖不掉它**。
                    #
                    # 后果（实测）：同一 prompt 采 G=4 次得到**逐字符相同**的输出
                    # → 组内奖励完全同分 → 组内优势恒 0 → **loss=grad=0**，
                    # 一个字都学不到。日志上却一切正常（reward 数值也在合理区间），
                    # 只有把"逐组原始回答"打出来才看得见。
                    #
                    # 验证：换不同随机种子各跑一次，看输出是否变化 ——
                    #   不设 top_k → 1/3 种输出（确定）
                    #   设 top_k=0 → 3/3 种输出（真随机）
                    top_k=0,
                )
            text = self.processor.batch_decode(
                gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            out.append((text or "").strip())
        return out

    def logprobs(self, samples, responses, *, requires_grad: bool = False):
        """返回 log 概率。

        Parameters
        ----------
        requires_grad
            **这个开关是整条训练链路的关键。**

            - `False`（用于 `logp_old` / `logp_ref`）：detach 成 numpy，
              省显存、也不需要梯度。
            - `True`（用于 `logp_new`）：**保留计算图并返回 torch.Tensor**，
              否则梯度传不回模型。

            ⚠️ 踩过的坑：第一版无论哪种情况都 `.detach().cpu().numpy()`，
            于是训练循环里"看起来在跑、loss 也在变"，但**参数一动没动** ——
            这和直接写个假 backward 是同一种病，而且更隐蔽
            （因为 numpy 版损失函数本身是对的、单测也全过）。
            所以 `tests/test_grpo.py` 里专门有一条 `test_torch_loss_actually_produces_gradients`。

        ⚠️ 因果语言模型要**右移一位**：位置 t 的 logits 预测的是 token t+1。
        这是最容易写错、且错了不报错（只是梯度方向乱）的地方。
        """
        torch = self.torch
        rows: List[Any] = []
        lens: List[int] = []
        use_grad = bool(requires_grad) and not self.reference_mode
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        for s, resp in zip(samples, responses):
            messages, img_path = self._encode(s, resp)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            images = None
            if img_path.exists():
                from PIL import Image
                images = [Image.open(img_path).convert("RGB")]
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            ids = inputs["input_ids"]
            with ctx:
                logits = self.model(**inputs).logits          # (1, T, V)
            resp_ids = self.processor.tokenizer(
                resp, add_special_tokens=False, return_tensors="pt")["input_ids"]
            n_resp = int(resp_ids.shape[1])
            n_resp = max(1, min(n_resp, ids.shape[1] - 1))
            logp_full = torch.log_softmax(logits.float(), dim=-1)
            tgt = ids[0, -n_resp:]
            src = logp_full[0, -n_resp - 1:-1, :]
            lp = src.gather(-1, tgt[:, None]).squeeze(-1)
            rows.append(lp if use_grad else lp.detach().float().cpu().numpy())
            lens.append(n_resp)

        if use_grad:
            # 拼成带梯度的张量：真实段用 torch.cat，padding 位用零（会被 mask 掉）
            t_max = max(lens)
            padded = []
            for r, L in zip(rows, lens):
                if L < t_max:
                    pad = torch.zeros(t_max - L, dtype=r.dtype, device=r.device)
                    r = torch.cat([r, pad])
                padded.append(r)
            lp_t = torch.stack(padded, dim=0)
            return lp_t, torch.ones_like(lp_t)

        # numpy 路径：padding 到等长
        t_max = max(lens)
        n = len(rows)
        lp_arr = np.zeros((n, t_max), dtype=np.float32)
        mask = np.zeros((n, t_max), dtype=np.float32)
        for i, (r, L) in enumerate(zip(rows, lens)):
            lp_arr[i, :L] = r
            mask[i, :L] = 1.0
        return lp_arr, mask

    def set_reference_mode(self, on: bool) -> None:
        if on and not self.reference_mode:
            self.model.disable_adapter_layers()
            self.model.eval()
            self.reference_mode = True
        elif not on and self.reference_mode:
            self.model.enable_adapter_layers()
            self.reference_mode = False

    def train_mode(self) -> None:
        self.model.train()

    def optimizer_step(self) -> None:
        self.torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()


# =============================================================================
# 真实训练（带梯度的版本）
# =============================================================================
def run_real_training(args, samples) -> Dict[str, Any]:
    import torch as _torch
    from roboground.rl.torch_loss import clip_grad_norm, grpo_loss_torch

    def T(x):
        """numpy → torch（放在 GPU 上，float32 以匹配损失里的计算）。"""
        return _torch.as_tensor(np.asarray(x), dtype=_torch.float32,
                                device="cuda" if _torch.cuda.is_available() else "cpu")

    policy = QwenVlPolicy(args.model, lora_r=args.lora_r, lr=args.lr,
                           init_adapter=args.init_adapter)
    policy.model.print_trainable_parameters()

    grpo_cfg = GrpoConfig(
        group_size=args.g, clip_eps=args.clip_eps,
        clip_eps_high=(None if args.no_clip_higher else args.clip_eps_high),
        beta_kl=args.beta_kl, kl_estimator=args.kl_estimator,
        level=args.level, normalize_advantage=not args.no_adv_norm,
    )
    reward_cfg = RewardConfig()
    hist: List[Dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    for step in range(args.steps):
        idx = rng.choice(len(samples), size=min(args.prompts, len(samples)),
                         replace=False)
        prompts = [samples[int(i)] for i in idx]
        g = args.g
        batch = [s for s in prompts for _ in range(g)]

        policy.set_reference_mode(False)
        responses = policy.generate(batch, temperature=args.temperature,
                                    max_new_tokens=args.max_new_tokens)
        bds = [spatial_reward(t, s.oracle, reward_cfg)
               for t, s in zip(responses, batch)]
        rewards = np.array([b.total for b in bds])
        rstats = reward_stats(bds)
        adv = np.zeros_like(rewards)
        degen_groups = 0
        group_stds: List[float] = []
        for i in range(len(prompts)):
            sl = slice(i * g, (i + 1) * g)
            adv[sl], deg = group_advantages(rewards[sl],
                                            normalize=not args.no_adv_norm)
            degen_groups += int(deg)
            group_stds.append(float(np.std(rewards[sl])))

        policy.set_reference_mode(False)
        lp_old, mask = policy.logprobs(batch, responses)          # numpy，无需梯度
        lp_ref = None
        if grpo_cfg.beta_kl > 0:
            policy.set_reference_mode(True)
            lp_ref, _ = policy.logprobs(batch, responses)
            policy.set_reference_mode(False)
        lengths = np.asarray(mask).sum(axis=1)

        policy.train_mode()
        ran = 0
        gnorm = 0.0
        for _ in range(max(1, args.inner_epochs)):
            # ★ requires_grad=True → 返回带计算图的 torch.Tensor
            lp_new, m_t = policy.logprobs(batch, responses, requires_grad=True)
            loss, diag = grpo_loss_torch(
                lp_new,
                T(lp_old), (T(lp_ref) if lp_ref is not None else None),
                T(adv), T(lengths), m_t, grpo_cfg)
            policy.optimizer.zero_grad()
            loss.backward()                      # ← 真正的反向传播
            gnorm = clip_grad_norm(policy.model, 1.0)
            policy.optimizer.step()
            ran += 1
        hist.append({
            "step": step,
            "reward_mean": float(rewards.mean()),
            "correct_mean": rstats["correct_mean"],
            "format_rate": rstats["format_rate"],
            # ★ 与 format_rate 分开看：正则兜底会把自由文本也判成"可解析"，
            # 只有 json_rate 才真实反映模型有没有按格式输出。
            "json_rate": rstats["json_rate"],
            "zero_adv_frac": float(np.mean(np.abs(adv) < 1e-9)),
            "degenerate_group_frac": degen_groups / max(len(prompts), 1),
            #: 逐组奖励明细 —— 判断"退化"是模型真同分还是代码有 bug 的关键证据
            "group_reward_stds": [round(s, 6) for s in group_stds],
            "reward_per_sample": [round(float(r), 4) for r in rewards],
            "n_distinct_rewards": int(len(np.unique(np.round(rewards, 6)))),
            # ★ 直接看"同一个 prompt 的 G 个回答是否不同" —— 这是冷启动问题的判据
            "group0_responses": [r[:60] for r in responses[:g]],
            "loss": diag.loss, "clip_frac": diag.clip_frac,
            "entropy": diag.entropy, "grad_norm": gnorm,
            "n_inner_updates": ran,
            "sample_response": (responses[0][:120] if responses else ""),
            "elapsed_s": round(time.time() - t0, 1),
        })
        log.info(f"step {step + 1}/{args.steps} "
                 f"reward={hist[-1]['reward_mean']:.4f} "
                 f"correct={hist[-1]['correct_mean']:.3f} "
                 f"fmt={hist[-1]['format_rate']:.2f} "
                 f"json={hist[-1]['json_rate']:.2f} "
                 f"loss={diag.loss:.4f} grad={gnorm:.3f} "
                 f"clip={diag.clip_frac:.2f} ent={diag.entropy:.3f} "
                 f"degenerate={hist[-1]['degenerate_group_frac']:.0%}")

    return {"mode": "real", "history": hist,
            "elapsed_s": round(time.time() - t0, 1)}


# =============================================================================
# 离线自检（不用 GPU）
# =============================================================================
def run_dry_run(args, samples) -> Dict[str, Any]:
    """用 FakePolicy 跑通整条循环，验证接线是否正确。"""
    print()
    print("=" * 84)
    print("离线自检：用假策略验证 GRPO 接线（不加载模型）")
    print("=" * 84)

    results: Dict[str, Any] = {}

    # ---- 对照 1：难度偏差（正确率 1.0 → 组内全对 → 优势退化）----
    print()
    print("① 难度偏差：把假策略的正确率调到极端，看组内优势是否退化")
    print(f"   {'正确概率':>10}{'奖励均值':>12}{'优势退化组比':>16}{'学得到吗':>12}")
    print("   " + "-" * 54)
    for p in (0.0, 0.5, 1.0):
        # drift=False：不模拟策略更新，得到"难度偏差"的纯净对照
        policy = FakePolicy(correct_prob=p, seed=0, drift=False)
        tr = GrpoTrainer(policy, samples,
                         trainer_cfg=TrainerConfig(steps=4, prompts_per_step=4,
                                                   group_size=4, seed=0, log_every=0))
        summary = tr.train()
        degen = float(np.mean([h["degenerate_group_frac"] for h in tr.history]))
        can = "否（无梯度）" if degen >= 1.0 else "是"
        print(f"   {p:>10.1f}{summary['reward_mean']:>12.4f}{degen:>16.1%}{can:>14}")
    results["difficulty_bias"] = "正确率 0 或 1 时组内同分 → 优势全 0 → 无梯度"

    # ---- 对照 2：inner_epochs 让 IS 比率偏离 1 ----
    print()
    print("② Importance Sampling：内层更新次数决定比率是否偏离 1")
    print(f"   {'inner_epochs':>14}{'mean_ratio':>14}{'clip_frac':>12}")
    print("   " + "-" * 42)
    for ie in (1, 3):
        policy = FakePolicy(correct_prob=0.5, seed=1)
        tr = GrpoTrainer(policy, samples,
                         trainer_cfg=TrainerConfig(steps=6, prompts_per_step=4,
                                                   group_size=4, inner_epochs=ie,
                                                   seed=1, log_every=0))
        tr.train()
        mr = float(np.mean([h["mean_ratio"] for h in tr.history]))
        cf = float(np.mean([h["clip_frac"] for h in tr.history]))
        print(f"   {ie:>14}{mr:>14.4f}{cf:>12.4f}")
        results[f"inner_epochs_{ie}"] = {"mean_ratio": mr, "clip_frac": cf}
    print("   → inner_epochs=1 时策略没变过，ratio 恒为 1，**clip 是死代码**；")
    print("     ≥2 次内层更新才会让 ratio 偏离 1，clip 与 IS 才真正生效。")

    # ---- 对照 3：token 级 vs sequence 级 IS ----
    print()
    print("③ Token 级 vs Sequence 级 IS（GSPO 的动机）")
    from roboground.rl import token_ratios, sequence_ratios
    import math
    lp_old = np.zeros((1, 4))
    lp_new = np.array([[math.log(2.0), math.log(2.0), 0.0, 0.0]])
    print(f"   序列内逐 token 比率（token 级）：{np.round(token_ratios(lp_new, lp_old)[0], 3).tolist()}")
    print(f"   整条序列一个比率（sequence 级）：{np.round(sequence_ratios(lp_new, lp_old, np.array([4])), 3).tolist()}")
    print("   → token 级下同一序列里比率不一致，clip 之后目标函数不再对应"
          "『整条回答的好坏』；")
    print("     sequence 级做长度归一化后共享一个比率，语义一致、对 MoE 更稳。")

    # ---- 对照 4：奖励分量诊断 ----
    print()
    print("④ 奖励分解诊断（识别 reward hacking）")
    from roboground.rl import reward_stats
    oracle = SpatialOracle(task="relation", statement_true=True)
    for name, txt in (("只会写 JSON 但答错", '{"answer": "no"}'),
                      ("答对但格式自由", "Yes, it is to the left."),
                      ("完全跑偏", "我不知道")):
        bd = spatial_reward(txt, oracle)
        print(f"   {name:<20} total={bd.total:.3f} 格式={bd.format_ok:.1f} "
              f"正确={bd.correct:.3f} 解析途径={(bd.parsed or {}).get('_via', '-')}")
    print("   → 只看 total 无法发现问题；**必须逐项记录**。")

    results["checks"] = ["难度偏差", "IS 比率随 inner_epochs 偏离 1",
                         "token vs sequence 级 IS", "奖励分解诊断"]
    return results


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="离线自检（不加载模型）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--objects", type=int, default=5)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--max-per-task", type=int, default=4)
    ap.add_argument("--seed", type=int, default=101)
    # 训练
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--g", type=int, default=4, help="组大小 G")
    ap.add_argument("--inner-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--init-adapter", default=None,
                    help="SFT 好的 LoRA 路径；从它继续做 RL（推荐，否则冷启动学不动）")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--clip-eps-high", type=float, default=0.28)
    ap.add_argument("--no-clip-higher", action="store_true")
    ap.add_argument("--beta-kl", type=float, default=0.0)
    ap.add_argument("--kl-estimator", default="k3", choices=["k1", "k2", "k3"])
    ap.add_argument("--level", default="token", choices=["token", "sequence"])
    ap.add_argument("--no-adv-norm", action="store_true")
    ap.add_argument("--output", default="runs/grpo_spatial.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)

    samples = build_dataset(args)
    if not samples:
        log.error("没有构造出任何 RL 样本")
        return 2

    report: Dict[str, Any] = {"config": vars(args),
                              "n_samples": len(samples),
                              "task_counts": _task_counts(samples)}

    if args.dry_run:
        report["dry_run"] = run_dry_run(args, samples)
    else:
        report["train"] = run_real_training(args, samples)

    print()
    print("=" * 84)
    print("面试答题稿：可验证奖励（RLVR）怎么设计")
    print("=" * 84)
    print("""
  Q：主观生成的 RL，Reward 怎么设计？

  → 我的思路是**先问一句"这个任务能不能自动验证"**。
    能验证的，就不要训 reward model —— reward model 会带来 reward hacking，
    而且要和策略一起迭代，工程成本高得多。

    空间问答恰好是**能验证**的：我项目里的几何规则引擎能从 3D 地图
    算出精确答案（距离、坐标、关系真值）。于是：

        reward = 格式门控 × 正确性（塑形）

    三个设计要点：

    1. **格式门控**：输出不是可解析结构就直接 0 分，
       不给"乱写也有分"的空间。但必须留正则兜底 ——
       否则 RL 早期整组格式分全 0 → 组内优势退化 → **完全学不动**（冷启动）。

    2. **数值任务用塑形奖励，不用二值**。
       二值奖励（误差 <0.3m 给 1）在早期几乎恒为 0：
       基座坐标误差 2.76m，离 0.3m 十万八千里 → 整组全 0 → 优势全 0 → 梯度为 0。
       这就是**稀疏奖励**问题。塑形 exp(-|Δ|/τ) 虽然分都不高，但**排序正确**，
       组内就有相对优势。τ 要和可接受误差同量级。

    3. **奖励必须逐项记录**。只看 total 无法诊断 reward hacking ——
       "格式分涨、正确性分不动"意味着模型学会了写 JSON 但没学会答对，
       这时该调权重而不是继续训。

    另外一个诚实的边界：塑形奖励会引入**偏好偏差**
    （模型可能学会"输出看起来接近的垃圾"），所以它必须配格式门控，
    而且 τ 不能乱设。
""")

    if args.output:
        ensure_dir(Path(args.output).parent)
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        log.ok(f"结果已保存：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
