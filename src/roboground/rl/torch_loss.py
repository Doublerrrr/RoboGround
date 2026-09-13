"""GRPO 损失函数的 **torch 实现**（可反向传播）。

与 `grpo.py` 的关系
==================
`grpo.py` 里的 `grpo_loss` 是**参考实现**（纯 numpy）：
公式写在一处、能离线单测、能画图分析，但**不可反向传播**。

本模块是**可微实现**，供真实训练使用。

为什么要两份而不是只留一份
------------------------
只留 numpy 就训不了；只留 torch 就没法脱离 GPU 验证数学。
更重要的是：两份实现之间做**数值等价测试**
（`tests/test_grpo.py::test_torch_loss_matches_numpy_reference`），
可以在"参考实现改对了但优化实现没跟上"时立刻发现 ——
这比只保留一份、然后靠肉眼核对要可靠得多。

⚠️ 曾经的错误做法：我在训练脚本里写了个"用 numpy 算 loss，
再包一个 torch 标量去 backward"的假反向 —— 那样**梯度根本传不到模型**，
训练会"看起来在跑、loss 也在变"，但参数一动没动。
（这类 bug 靠看 loss 曲线是发现不了的，必须验证参数是否真的变了。）
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from roboground.rl.grpo import GrpoConfig, GrpoLoss

__all__ = ["grpo_loss_torch", "clip_grad_norm"]


def grpo_loss_torch(logp_new, logp_old, logp_ref, advantages,
                    lengths, mask, cfg: Optional[GrpoConfig] = None
                    ) -> Tuple[object, GrpoLoss]:
    """可微的 GRPO 损失；返回 `(loss_tensor, 诊断量)`。

    与 numpy 版逐项对应，公式完全一致：

        level == "token"    : r_{i,t} = exp(logp_new - logp_old)
        level == "sequence" : r_i    = exp( (Σ_t Δ_{i,t}) / |o_i| )   ← GSPO
        obj = min(r·A, clip(r, 1-ε, 1+ε_high)·A)
        loss = -Σ obj / Σ mask  + β · KL

    Parameters
    ----------
    其余参数同 `grpo.py::grpo_loss`，但都是 torch.Tensor。
    """
    import torch

    cfg = cfg or GrpoConfig()
    m = mask.float()
    n_tok = m.sum()
    if float(n_tok) <= 0:
        zero = (logp_new.sum() * 0.0)
        return zero, GrpoLoss()

    # ---- IS 比率 ----
    if cfg.level == "sequence":
        ln = lengths.float().clamp(min=1.0)
        diff = (logp_new - logp_old).sum(dim=-1) / ln
        r_seq = torch.exp(diff.clamp(-20.0, 20.0))
        ratios = r_seq.unsqueeze(-1).expand_as(logp_new)
    else:
        ratios = torch.exp((logp_new - logp_old).clamp(-20.0, 20.0))

    # ---- 优势广播到 token ----
    adv = advantages.float().unsqueeze(-1)

    # ---- Clip 目标 ----
    hi = cfg.clip_eps if cfg.clip_eps_high is None else float(cfg.clip_eps_high)
    r_clipped = torch.clamp(ratios, 1.0 - cfg.clip_eps, 1.0 + hi)
    obj = torch.minimum(ratios * adv, r_clipped * adv) * m
    policy_loss = -(obj.sum() / n_tok)

    # ---- KL ----
    kl_val = None
    if logp_ref is not None and cfg.beta_kl > 0:
        d = logp_ref - logp_new
        if cfg.kl_estimator == "k1":
            kl_tok = -d
        elif cfg.kl_estimator == "k2":
            kl_tok = 0.5 * d * d
        else:                                  # k3：无偏且方差最小
            kl_tok = torch.exp(d.clamp(-20.0, 20.0)) - 1.0 - d
        kl_val = (kl_tok * m).sum() / n_tok
    loss = policy_loss + (cfg.beta_kl * kl_val if kl_val is not None else 0.0)

    # ---- 诊断量（detach 后转 numpy，不参与梯度）----
    with torch.no_grad():
        clip_frac = ((ratios - r_clipped).abs() > 1e-9).float().mul(m).sum() / n_tok
        mean_ratio = (ratios * m).sum() / n_tok
        entropy = (-logp_new * m).sum() / n_tok
        zero_adv = (advantages.abs() < 1e-9).float().mean() if advantages.numel() else \
            torch.zeros((), device=logp_new.device)
        diag = GrpoLoss(
            loss=float(loss.detach()),
            policy_loss=float(policy_loss.detach()),
            kl=float(kl_val.detach()) if kl_val is not None else 0.0,
            entropy=float(entropy),
            clip_frac=float(clip_frac),
            zero_adv_frac=float(zero_adv),
            mean_ratio=float(mean_ratio),
            n_tokens=int(n_tok),
        )
    return loss, diag


def clip_grad_norm(model, max_norm: float = 1.0) -> float:
    """梯度裁剪并返回裁剪前的总范数（用于监控梯度爆炸）。"""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    total = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total)
