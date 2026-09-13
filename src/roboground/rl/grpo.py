"""GRPO 目标函数与训练循环（**手写实现**，不依赖 trl）。

为什么手写
=========
面经显示这个岗位的基础拷打里 **RL 占 5/9**：
GRPO 目标函数、组内优势估计、Importance Sampling、Clip 的作用与对象、
KL 散度、熵坍塌、GSPO 的 sequence-level IS。
这些只有自己推过、调过、看它崩过，才能答出层次。

因此本模块把每一个部件都拆成**可单测的纯函数**，
并且把面试问到的那几个"变体开关"都做成参数：

| 面试问到的 | 本模块对应的开关 |
|---|---|
| Clip 的作用与**对象** | `clip_eps` / `clip_eps_high`（非对称 = clip-higher） |
| Token / Sequence 不一致 | `level="token"｜"sequence"`（后者即 GSPO 的 sequence-level IS） |
| KL 散度 | `kl_estimator="k1"｜"k2"｜"k3"` |
| 熵坍塌 | `clip_eps_high` + `entropy` 监控 |
| 长度归一化 | `length_normalize` |
| 稀疏奖励 | 塑形奖励（见 `reward.py`） |

GRPO 与 PPO 的本质区别
=====================
PPO 需要一个 **value network** 估计基线 `V(s)` 来算优势；
GRPO **砍掉 value network**，改用"**同一 prompt 采一组 G 个样本，用组内均值当基线**"：

    A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

代价：每个 prompt 要多采样 G 次（算力换显存与复杂度）。
收益：省掉一个和策略同规模的 value network，且**天然适配可验证奖励**
（因为不需要学 value，只要能把答案判对错）。

⚠️ 这也解释了 GRPO 的一个已知缺陷 —— **难度偏差**：
如果一组 G 个样本**全对或全错**，`std=0` → 除以 eps 后优势仍为 0 →
**这道题贡献不了任何梯度**。太简单的题和太难的题都是"白采"。
所以训练时要监控 `zero_advantage_ratio`（见 `TrainStats`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("rl.grpo")


# =============================================================================
# 1) 组内优势估计（GRPO 的心脏）
# =============================================================================
def group_advantages(rewards: np.ndarray, *, eps: float = 1e-4,
                     normalize: bool = True) -> Tuple[np.ndarray, bool]:
    """组内相对优势。

    Parameters
    ----------
    rewards
        同一 prompt 下 G 个样本的奖励，形状 `(G,)`。
    normalize
        是否除以组内标准差。设 False 则只减均值（**无偏但尺度不一致**）——
        消融用，因为除 std 会让"组内差异小"的题目被放大，
        相当于给简单题更大的梯度权重。

    Returns
    -------
    (advantages, degenerate)
        `degenerate=True` 表示这组**全对或全错**（std≈0），
        优势全 0、贡献不了梯度。这是 GRPO 难度偏差的直接体现，
        必须计数上报而不是静默丢弃。
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return np.zeros(0, dtype=np.float64), True
    adv = r - r.mean()
    if normalize:
        std = r.std()
        if std < eps:
            # ★ 全班同分 → 没有相对好坏 → 优势恒 0
            return np.zeros_like(adv), True
        adv = adv / (std + eps)
    else:
        if np.abs(adv).max() < eps:
            return np.zeros_like(adv), True
    return adv, False


# =============================================================================
# 2) Importance Sampling 比率（Token 级 vs Sequence 级）
# =============================================================================
def token_ratios(logp_new: np.ndarray, logp_old: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> np.ndarray:
    """**Token 级** IS 比率 `π_θ(o_t) / π_old(o_t)`。

    ⚠️ 这是 GRPO 与 GSPO 分歧的起点：
    逐 token 算比率意味着**同一个序列里的不同 token 会有不同的比率**，
    于是"序列级的重要性权重"被拆成了多个不一致的局部权重。
    面经里问的"Token/Sequence 不一致"就是指这个。
    """
    ratio = np.exp(np.clip(logp_new - logp_old, -20.0, 20.0))
    if mask is not None:
        ratio = ratio * np.asarray(mask, dtype=np.float64)
    return ratio


def sequence_ratios(logp_new: np.ndarray, logp_old: np.ndarray,
                    lengths: np.ndarray) -> np.ndarray:
    """**Sequence 级** IS 比率（GSPO 的做法）：长度归一化后再取指数。

        s_i = exp( (1/|o_i|) * Σ_t log π_θ(o_t) - (1/|o_i|) * Σ_t log π_old(o_t) )

    为什么 GSPO 要这么做（三个动机，面试常问）：
    1. **Token/Sequence 不一致**：token 级比率下，一个序列里各 token 的比率
       可能一个 1.5 一个 0.3，clip 之后目标函数不再对应"整条回答的好坏"；
       长度归一化后整条序列共享一个比率，语义一致。
    2. **对 MoE 友好**：MoE 路由会放大逐 token 的方差，
       序列级比率把方差摊平，训练更稳。
    3. **长度归一化**：长回答的 logp 和天然更负，
       不做长度归一化会让**长回答的比率系统性偏小**，被 clip 误伤。
    """
    lp_new = np.asarray(logp_new, dtype=np.float64)
    lp_old = np.asarray(logp_old, dtype=np.float64)
    n = np.maximum(np.asarray(lengths, dtype=np.float64), 1.0)
    diff = (lp_new.sum(axis=-1) - lp_old.sum(axis=-1)) / n
    return np.exp(np.clip(diff, -20.0, 20.0))


# =============================================================================
# 3) 带 Clip 的目标函数
# =============================================================================
def clipped_surrogate(ratios: np.ndarray, advantages: np.ndarray, *,
                      clip_eps: float = 0.2,
                      clip_eps_high: Optional[float] = None,
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
    """逐元素的 `min(r·A, clip(r)·A)`（PPO/GRPO 的 clip 目标）。

    Clip 的作用（三层，面试按这个顺序答）
    ------------------------------------
    1. **限制单步更新幅度**：`r` 偏离 1 太远说明新旧策略差太多，
       此时梯度方向不再可信；clip 把这种样本的目标函数变成常数，
       **梯度为 0** —— 等价于"这一步先不学它"。
    2. **对象是"新旧策略的概率比"，不是奖励也不是优势**。
       常见错误答法是"clip 奖励"，那是错的。
    3. **min 让它悲观** —— 注意两个方向**都会被截**，截的都是"乐观方向"：

       | 优势 | 比率情形 | 结果 |
       |---|---|---|
       | A > 0 | `r` 超过 `1+ε`（好动作概率涨太猛） | 截成 `(1+ε)·A` |
       | A > 0 | `r` 低于 1（好动作概率在跌） | **不截**，照常给梯度 |
       | A < 0 | `r` 低于 `1-ε`（坏动作概率跌太猛） | 截成 `(1-ε)·A` |
       | A < 0 | `r` 高于 1（坏动作概率在涨） | **不截**，照常惩罚 |

       一句话：**只在"策略已经很努力往正确方向走"时停止加码**，
       而在"策略在往错的方向走"时继续给梯度。
       这保证了单调改进的下界（PPO 论文的核心论证）。
       ⚠️ 容易说反 —— 我第一版把"A<0 时不截断"写进了注释，
       是被单测里 `obj = (1-ε)·A = -0.8` 而不是 `-0.5` 抓出来的。

    Parameters
    ----------
    clip_eps_high
        **非对称上界（clip-higher / DAPO）**。
        动机是**熵坍塌**：标准对称 clip 下，低概率 token 的比率
        很容易被上界截住 → 这些 token 永远拿不到正梯度 →
        策略越来越确定 → 熵塌掉、输出单调重复。
        把**上界放宽**（如 0.28 vs 下界 0.2），给低概率 token 留出上升通道。
        注意是放宽**上界**而不是下界 —— 放宽下界会让坏样本更容易被学。
    """
    r = np.asarray(ratios, dtype=np.float64)
    a = np.asarray(advantages, dtype=np.float64)
    hi = clip_eps if clip_eps_high is None else float(clip_eps_high)
    r_clipped = np.clip(r, 1.0 - clip_eps, 1.0 + hi)
    obj = np.minimum(r * a, r_clipped * a)
    if mask is not None:
        obj = obj * np.asarray(mask, dtype=np.float64)
    return obj


# =============================================================================
# 4) KL 散度估计
# =============================================================================
def kl_penalty(logp: np.ndarray, logp_ref: np.ndarray, *,
               estimator: str = "k3") -> np.ndarray:
    """逐 token 的 KL(π_θ ‖ π_ref) 估计。

    三种估计器（面试常追问"你用哪个、为什么"）
    ----------------------------------------
    设 `d = logp_ref - logp`（即 log(π_ref/π_θ)）：

    | 估计器 | 公式 | 特点 |
    |---|---|---|
    | `k1` | `-d` | **无偏但方差大**，且单样本可以为负（KL 本不该为负） |
    | `k2` | `d² / 2` | 恒非负、方差小，但**有偏**（小 d 时系统性偏高） |
    | **`k3`** | `exp(d) - 1 - d` | **无偏且方差最小**，是 RLHF 实践默认 |

    ⚠️ 关键事实（容易被说错）：**k1 与 k3 都是无偏估计**，
    差别在**方差**而不在期望 —— `E_{π_θ}[k1] = KL`，
    且因 `E_{π_θ}[exp(d)] = Σ π_θ·(π_ref/π_θ) = 1`，
    有 `E[k3] = 1 - 1 - E[d] = KL`。
    选 k3 是因为它的方差小得多（实测蒙特卡洛里 k3 的标准差约为 k1 的 1/3），
    训练时梯度噪声更小。k1 还有个直观毛病：单样本上会出现**负的 KL**。

    k3 恒非负这个性质也很重要：KL 惩罚永远只往
    "别偏离参考太远"的方向推，不会出现自相矛盾的引导。
    """
    lp = np.asarray(logp, dtype=np.float64)
    lr = np.asarray(logp_ref, dtype=np.float64)
    d = lr - lp
    if estimator == "k1":
        return -d
    if estimator == "k2":
        return 0.5 * d * d
    if estimator == "k3":
        return np.exp(np.clip(d, -20.0, 20.0)) - 1.0 - d
    raise ValueError(f"未知 KL 估计器 {estimator!r}（可选 k1/k2/k3）")


# =============================================================================
# 5) 完整 GRPO 损失
# =============================================================================
@dataclass
class GrpoConfig:
    """GRPO 目标函数配置。"""

    #: 组大小 G：同一 prompt 采几个样本
    group_size: int = 4
    clip_eps: float = 0.2
    #: clip-higher：放宽上界，缓解熵坍塌。None = 对称
    clip_eps_high: Optional[float] = 0.28
    #: KL 惩罚系数 β。0 表示不做 KL 约束（纯 RLVR 常见做法）
    beta_kl: float = 0.01
    kl_estimator: str = "k3"
    #: IS 比率层级：token（GRPO 原版）/ sequence（GSPO）
    level: str = "token"
    #: 是否做长度归一化
    length_normalize: bool = True
    #: 优势是否除以组内标准差
    normalize_advantage: bool = True


@dataclass
class GrpoLoss:
    """损失 + 全部诊断量（**诊断量必须一起返回**，否则无法定位问题）。"""

    loss: float = 0.0
    policy_loss: float = 0.0
    kl: float = 0.0
    #: 策略熵（nats/token）—— 掉到很低就是**熵坍塌**
    entropy: float = 0.0
    #: 被 clip 的 token 比例。过高说明新旧策略差太多（学习率过大）
    clip_frac: float = 0.0
    #: 优势为 0 的样本比例（GRPO **难度偏差**的直接度量）
    zero_adv_frac: float = 0.0
    mean_ratio: float = 1.0
    n_tokens: int = 0


def grpo_loss(logp_new: np.ndarray, logp_old: np.ndarray,
              logp_ref: Optional[np.ndarray],
              advantages: np.ndarray,
              lengths: np.ndarray,
              mask: np.ndarray,
              cfg: Optional[GrpoConfig] = None) -> GrpoLoss:
    """计算 GRPO 损失与全部诊断量。

    Parameters
    ----------
    logp_new, logp_old, logp_ref
        `(N, T)` 新旧策略与参考策略的逐 token log 概率。
        `logp_ref` 可以为 None（β=0 时不需要参考模型，**省一半显存**）。
    advantages
        `(N,)` 组内优势。
    lengths
        `(N,)` 每条样本的有效 token 数（用于长度归一化与 GSPO 比率）。
    mask
        `(N, T)` 有效 token 掩码（padding 为 0）。

    Returns
    -------
    GrpoLoss
    """
    cfg = cfg or GrpoConfig()
    lp_new = np.asarray(logp_new, dtype=np.float64)
    lp_old = np.asarray(logp_old, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64).reshape(-1)
    n_tok_total = float(m.sum())
    if n_tok_total <= 0:
        return GrpoLoss()

    # ---- IS 比率 ----
    if cfg.level == "sequence":
        r_seq = sequence_ratios(lp_new, lp_old, lengths)          # (N,)
        ratios = np.repeat(r_seq[:, None], lp_new.shape[1], axis=1)
    else:
        ratios = token_ratios(lp_new, lp_old)                      # (N, T)

    # ---- 广播优势到 token ----
    adv_tok = adv[:, None]

    # ---- Clip 目标 ----
    obj = clipped_surrogate(ratios, adv_tok, clip_eps=cfg.clip_eps,
                            clip_eps_high=cfg.clip_eps_high, mask=m)
    if cfg.length_normalize:
        policy_loss = -float(obj.sum() / n_tok_total)
    else:
        per_seq = obj.sum(axis=1) / np.maximum(m.sum(axis=1), 1.0)
        policy_loss = -float(per_seq.mean())

    # ---- KL ----
    kl_val = 0.0
    if logp_ref is not None and cfg.beta_kl > 0:
        kl_tok = kl_penalty(lp_new, logp_ref, estimator=cfg.kl_estimator) * m
        kl_val = float(kl_tok.sum() / n_tok_total)

    # ---- 诊断量 ----
    # 熵：用 -logp 的期望近似（真正的熵需要完整分布；这里是常用的代理，
    # 趋势判断足够，绝对值不要当真）
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = float(np.nansum(-lp_new * m) / n_tok_total)
    r_clipped = np.clip(ratios, 1.0 - cfg.clip_eps,
                        1.0 + (cfg.clip_eps if cfg.clip_eps_high is None
                               else cfg.clip_eps_high))
    clip_frac = float(((np.abs(ratios - r_clipped) > 1e-9) & (m > 0)).sum()
                      / max(n_tok_total, 1.0))
    zero_adv = float((np.abs(adv) < 1e-9).mean()) if adv.size else 0.0
    mean_ratio = float((ratios * m).sum() / n_tok_total)

    return GrpoLoss(
        loss=policy_loss + cfg.beta_kl * kl_val,
        policy_loss=policy_loss,
        kl=kl_val,
        entropy=ent,
        clip_frac=clip_frac,
        zero_adv_frac=zero_adv,
        mean_ratio=mean_ratio,
        n_tokens=int(n_tok_total),
    )


# =============================================================================
# 6) 训练统计与健康检查
# =============================================================================
@dataclass
class TrainStats:
    """累计训练统计 + 健康告警。"""

    steps: int = 0
    zero_adv_frac: List[float] = field(default_factory=list)
    clip_frac: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    reward_mean: List[float] = field(default_factory=list)
    kl: List[float] = field(default_factory=list)

    def update(self, loss: GrpoLoss, reward_mean: float) -> None:
        self.steps += 1
        self.zero_adv_frac.append(loss.zero_adv_frac)
        self.clip_frac.append(loss.clip_frac)
        self.entropy.append(loss.entropy)
        self.reward_mean.append(reward_mean)
        self.kl.append(loss.kl)

    def warnings(self) -> List[str]:
        """健康检查 —— 把"面试常问的几个病"做成自动告警。"""
        out: List[str] = []
        if not self.steps:
            return out
        za = float(np.mean(self.zero_adv_frac))
        cf = float(np.mean(self.clip_frac))
        if za > 0.5:
            out.append(
                f"**难度偏差**：{za:.0%} 的样本优势为 0（组内全对或全错），"
                "这些题贡献不了梯度。对策：过滤过易/过难的题、增大 G、"
                "或改用塑形奖励让奖励不至于全 0。"
            )
        if cf > 0.3:
            out.append(
                f"**clip 比例过高**：{cf:.0%} 的 token 被截断，"
                "说明新旧策略差异过大。对策：降学习率、减小更新步数。"
            )
        if len(self.entropy) >= 4:
            half = len(self.entropy) // 2
            drop = float(np.mean(self.entropy[:half]) - np.mean(self.entropy[half:]))
            if drop > 0.5:
                out.append(
                    f"**熵坍塌**：策略熵下降 {drop:.2f} nats，输出在变得单调。"
                    "对策：开 clip-higher（放宽上界）、提高 KL 系数、"
                    "或对高概率 token 做裁剪。"
                )
        if len(self.reward_mean) >= 4:
            half = len(self.reward_mean) // 2
            gain = float(np.mean(self.reward_mean[half:]) - np.mean(self.reward_mean[:half]))
            if gain < 1e-4:
                out.append("**奖励没涨**：检查奖励设计（是否全 0 或全满）、"
                           "优势是否退化、学习率是否过小。")
        return out

    def summary(self) -> Dict[str, float]:
        if not self.steps:
            return {"steps": 0}
        return {
            "steps": self.steps,
            "reward_mean": float(np.mean(self.reward_mean)),
            "reward_first_half": float(np.mean(self.reward_mean[:max(1, self.steps // 2)])),
            "reward_last_half": float(np.mean(self.reward_mean[self.steps // 2:])),
            "zero_adv_frac": float(np.mean(self.zero_adv_frac)),
            "clip_frac": float(np.mean(self.clip_frac)),
            "entropy_first": float(self.entropy[0]),
            "entropy_last": float(self.entropy[-1]),
            "kl_mean": float(np.mean(self.kl)),
        }
