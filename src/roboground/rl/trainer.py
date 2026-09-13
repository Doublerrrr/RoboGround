"""GRPO 训练循环：把策略模型接到目标函数上。

设计要点（都是为了"能讲清楚 + 能验证"）
=====================================
1. **策略抽象成 `PolicyBase` 接口**
   → 整条 RL 循环可以用 `FakePolicy` 离线跑通，不用 GPU 也能单测
   （组内优势、IS 比率、clip、KL 是否接对了）。
   真实模型只是这个接口的一个实现。

2. **参考模型用 `disable_adapter()`，不额外加载一份**
   → 4bit 下 Qwen2-VL-2B 约 1.8GB，再加载一份参考模型直接翻倍。
   因为只训 LoRA，把 adapter 关掉就是参考策略，**省掉一整份模型显存**。
   代价：算 `logp_ref` 时要单独前向一次（时间换显存）。

3. **`inner_epochs` 让 Importance Sampling 真正有意义**
   → 生成后立刻算 `logp_old`（此时 ratio≡1），若只做 1 次内层更新，
   ratio 永远等于 1，**clip 和 IS 全都是死代码**。
   ≥2 次内层更新才会让策略偏离生成时的策略，clip 才被激活。
   所以 `inner_epochs=1` 是很好的消融对照（见脚本里的说明）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from roboground.rl.grpo import GrpoConfig, GrpoLoss, TrainStats, group_advantages, grpo_loss
from roboground.rl.reward import (
    RewardBreakdown, RewardConfig, SpatialOracle, reward_stats, spatial_reward,
)
from roboground.utils.logging import get_logger

logger = get_logger("rl.trainer")


# =============================================================================
# 策略接口
# =============================================================================
class PolicyBase(Protocol):
    """策略模型需要提供的最小能力。"""

    def generate(self, samples: Sequence[Dict[str, Any]], *,
                 temperature: float, max_new_tokens: int) -> List[str]:
        """对每条样本生成一个回答。"""
        ...

    def logprobs(self, samples: Sequence[Dict[str, Any]],
                 responses: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """返回 `(logp, mask)`，形状都是 `(N, T)`。

        mask 标出有效（非 padding）位置。
        """
        ...

    def set_reference_mode(self, on: bool) -> None:
        """切换到参考策略（关掉 LoRA adapter）或策略模式。"""
        ...


@dataclass
class RlSample:
    """一条 RL 样本：图像 + 问题 + **可验证真值**。"""

    image_path: str
    question: str
    oracle: SpatialOracle
    system_prompt: str = ""

    def to_prompt(self) -> str:
        return f"<image>{self.image_path}</image>\n{self.question}"


# =============================================================================
# 假策略（离线验证整条循环）
# =============================================================================
class FakePolicy:
    """离线假策略：按"正确概率 p"从预设答案里挑，用于验证训练循环。

    它模拟一个"有一定正确率的模型"，并且**支持人为退化**：
    `correct_prob` 越低，组内奖励差异越大（优势不退化）；
    接近 1 或 0 时组内同分 → 优势退化 → 正好用来验证难度偏差的告警。
    """

    def __init__(self, *, correct_prob: float = 0.5, seed: int = 0,
                 vocab_size: int = 32, seq_len: int = 8,
                 drift: bool = True) -> None:
        self.correct_prob = float(correct_prob)
        self.rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.reference_mode = False
        #: 是否模拟"策略在更新"（关掉可做难度偏差的纯净对照）
        self.drift = bool(drift)
        #: 记录调用次数，便于断言"参考模式确实被切过"
        self.calls: Dict[str, int] = {"generate": 0, "logprobs": 0, "ref": 0}
        #: 当前"策略权重"（用 logits 偏移模拟学习过程）
        self._shift = 0.0

    def _answers_for(self, sample: RlSample) -> List[str]:
        """构造"对/错"两种回答（用规则可解析的形式）。"""
        o = sample.oracle
        if o.task == "relation":
            right = '{"answer": "%s"}' % ("yes" if o.statement_true else "no")
            wrong = '{"answer": "%s"}' % ("no" if o.statement_true else "yes")
        elif o.task == "distance":
            right = '{"distance_m": %.2f}' % o.distance_m
            wrong = '{"distance_m": %.2f}' % (o.distance_m + 1.5)
        elif o.task == "locate":
            right = '{"coords": [%.2f, %.2f, %.2f]}' % o.coords
            wrong = '{"coords": [%.2f, %.2f, %.2f]}' % (o.coords[0] + 1.5,
                                                        o.coords[1], o.coords[2])
        else:
            right = '{"labels": %s, "count": %d}' % (
                list(o.on_labels), len(o.on_labels))
            wrong = '{"labels": [], "count": 0}'
        return [right, wrong]

    def generate(self, samples, *, temperature: float = 1.0,
                 max_new_tokens: int = 48) -> List[str]:
        self.calls["generate"] += 1
        out: List[str] = []
        p = float(np.clip(self.correct_prob + self._shift, 0.02, 0.98))
        for s in samples:
            right, wrong = self._answers_for(s)
            out.append(right if self.rng.random() < p else wrong)
        return out

    def logprobs(self, samples, responses) -> Tuple[np.ndarray, np.ndarray]:
        self.calls["logprobs"] += 1
        n = len(samples)
        t = self.seq_len
        mask = np.ones((n, t), dtype=np.float32)
        if self.reference_mode:
            self.calls["ref"] += 1
            base = -2.0                       # 参考策略：固定、略有不同的分布
        else:
            # 策略：随训练漂移。系数 2.0 是刻意调大的 ——
            # 太小的话 ratio 永远落在 clip 区间内，dry-run 演示不出 clip 生效。
            base = -1.5 + self._shift * 2.0
        lp = np.full((n, t), base, dtype=np.float32)
        # 加一点 token 间差异，让"sequence 级 vs token 级 IS"有区别
        lp[:, 1::2] += 0.1
        return lp, mask

    def set_reference_mode(self, on: bool) -> None:
        self.reference_mode = bool(on)

    #: 供训练器模拟一次"策略更新"
    def apply_update(self, delta: float = 0.08) -> None:
        if self.drift:
            self._shift += delta


# =============================================================================
# 训练器
# =============================================================================
@dataclass
class TrainerConfig:
    steps: int = 20
    prompts_per_step: int = 4
    group_size: int = 4
    inner_epochs: int = 2
    lr: float = 1e-5
    temperature: float = 1.0
    max_new_tokens: int = 48
    seed: int = 0
    #: 每隔多少步打印一次
    log_every: int = 1
    #: 是否只在"优势未退化"的组上更新（难度偏差的一种处置）
    skip_degenerate_groups: bool = False


class GrpoTrainer:
    """GRPO 训练循环。**与具体模型解耦**，只依赖 `PolicyBase`。"""

    def __init__(self, policy: PolicyBase, samples: Sequence[RlSample], *,
                 grpo_cfg: Optional[GrpoConfig] = None,
                 reward_cfg: Optional[RewardConfig] = None,
                 trainer_cfg: Optional[TrainerConfig] = None,
                 on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None,
                 ) -> None:
        self.policy = policy
        self.samples = list(samples)
        self.grpo_cfg = grpo_cfg or GrpoConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.cfg = trainer_cfg or TrainerConfig()
        self.on_step = on_step
        self.rng = np.random.default_rng(self.cfg.seed)
        self.stats = TrainStats()
        #: 逐步的原始记录（供脚本落盘）
        self.history: List[Dict[str, Any]] = []

    # ---------------- 一步 ----------------
    def step(self, prompts: Sequence[RlSample]) -> Dict[str, Any]:
        """一次参数更新：采样 → 打分 → 组内优势 → 多次内层更新。"""
        g = self.cfg.group_size
        # ---- 1) 采样 G 个回答 ----
        batch: List[RlSample] = []
        responses: List[str] = []
        for s in prompts:
            batch.extend([s] * g)
        self.policy.set_reference_mode(False)
        gen = self.policy.generate(batch, temperature=self.cfg.temperature,
                                   max_new_tokens=self.cfg.max_new_tokens)
        responses = list(gen)

        # ---- 2) 可验证奖励（规则引擎当 verifier）----
        bds: List[RewardBreakdown] = [
            spatial_reward(txt, s.oracle, self.reward_cfg)
            for txt, s in zip(responses, batch)
        ]
        rewards = np.array([b.total for b in bds], dtype=np.float64)
        rstats = reward_stats(bds)
        rstats["parse_via_json"] = float(np.mean(
            [1.0 if (b.parsed or {}).get("_via") == "json" else 0.0 for b in bds]))

        # ---- 3) 组内优势（GRPO 砍掉 value network 的关键）----
        n_groups = len(prompts)
        adv = np.zeros_like(rewards)
        degenerate_groups = 0
        for i in range(n_groups):
            sl = slice(i * g, (i + 1) * g)
            a, deg = group_advantages(rewards[sl],
                                      normalize=self.grpo_cfg.normalize_advantage)
            adv[sl] = a
            degenerate_groups += int(deg)

        # ---- 4) logp_old（生成时的策略）与 logp_ref ----
        self.policy.set_reference_mode(False)
        lp_old, mask = self.policy.logprobs(batch, responses)
        lp_ref = None
        if self.grpo_cfg.beta_kl > 0:
            self.policy.set_reference_mode(True)
            lp_ref, _ = self.policy.logprobs(batch, responses)
            self.policy.set_reference_mode(False)

        lengths = mask.sum(axis=1)
        group_adv_frac = degenerate_groups / max(n_groups, 1)

        # ---- 5) 多次内层更新（让 IS 比率真正偏离 1）----
        losses: List[GrpoLoss] = []
        for _ in range(max(1, self.cfg.inner_epochs)):
            self.policy.set_reference_mode(False)
            lp_new, _ = self.policy.logprobs(batch, responses)
            if self.cfg.skip_degenerate_groups and group_adv_frac >= 1.0:
                break
            out = grpo_loss(lp_new, lp_old, lp_ref, adv, lengths, mask,
                            self.grpo_cfg)
            losses.append(out)
            # 假策略用它模拟一次 SGD；真策略由脚本注入 optimizer 回调
            if hasattr(self.policy, "apply_update"):
                self.policy.apply_update()

        last = losses[-1] if losses else GrpoLoss()
        self.stats.update(last, float(rewards.mean()))
        rec = {
            "reward": rstats,
            "loss": last.loss,
            "policy_loss": last.policy_loss,
            "kl": last.kl,
            "entropy": last.entropy,
            "clip_frac": last.clip_frac,
            "zero_adv_frac": last.zero_adv_frac,
            "mean_ratio": last.mean_ratio,
            "degenerate_group_frac": group_adv_frac,
            "n_inner_updates": len(losses),
        }
        self.history.append(rec)
        return rec

    # ---------------- 训练 ----------------
    def train(self) -> Dict[str, Any]:
        n = len(self.samples)
        if n == 0:
            raise ValueError("没有训练样本")
        mc = self.cfg
        for step in range(mc.steps):
            idx = self.rng.choice(n, size=min(mc.prompts_per_step, n),
                                  replace=False)
            prompts = [self.samples[int(i)] for i in idx]
            rec = self.step(prompts)
            if self.on_step is not None:
                self.on_step(step, rec)
            if mc.log_every and (step + 1) % mc.log_every == 0:
                logger.info(
                    f"step {step + 1}/{mc.steps} "
                    f"reward={rec['reward']['reward_mean']:.4f} "
                    f"correct={rec['reward']['correct_mean']:.3f} "
                    f"fmt={rec['reward']['format_rate']:.2f} "
                    f"loss={rec['loss']:.4f} clip={rec['clip_frac']:.2f} "
                    f"ent={rec['entropy']:.3f} zeroAdv={rec['zero_adv_frac']:.2f}"
                )
        return self.stats.summary()


# =============================================================================
# 数据构造（把 3D 地图变成可验证的 RL 样本）
# =============================================================================
def build_rl_samples(smap, image_path: str, *, seed: int = 0,
                     max_per_task: int = 8,
                     system_prompt: str = "") -> List[RlSample]:
    """从 3D 语义地图构造 RL 样本（**带结构化 oracle**）。

    与 `scripts/12_gen_spatial_qa.py` 用的是**同一套几何逻辑**
    （`RuleEngine` + `compute_relation`），区别是这里保留**结构化真值**，
    而不是把答案提前渲染成自然语言 —— 这样才能做 verifiable reward。
    """
    import random  # noqa: PLC0415

    from roboground.reasoning.spatial_relations import (  # noqa: PLC0415
        bbox_gap, compute_relation,
    )

    rng = random.Random(seed)
    objs = [o for o in getattr(smap, "objects", []) if o.label and o.confidence > 0.3]
    if not objs:
        return []
    out: List[RlSample] = []
    per_task: Dict[str, int] = {}

    def _add(task: str, q: str, oracle: SpatialOracle) -> None:
        if per_task.get(task, 0) >= max_per_task:
            return
        per_task[task] = per_task.get(task, 0) + 1
        out.append(RlSample(image_path=image_path, question=q, oracle=oracle,
                            system_prompt=system_prompt))

    # 1) 关系（含**反例**：把假陈述也放进来，模型必须学会说不）
    pairs = [(a, b) for i, a in enumerate(objs) for b in objs[i + 1:]]
    rng.shuffle(pairs)
    for a_obj, b_obj in pairs[:12]:
        subj, tgt = (a_obj, b_obj) if rng.random() < 0.5 else (b_obj, a_obj)
        r = compute_relation(subj, tgt)
        rel_en = {"above": "above", "below": "below", "left": "to the left of",
                  "right": "to the right of", "in front of": "in front of",
                  "behind": "behind", "near": "near", "inside": "inside",
                  "overlapping": "overlapping"}.get(r.relation, r.relation)
        q = (f"Is the {subj.label} {rel_en} the {tgt.label}? "
             f"Answer with JSON {{\"answer\": \"yes\" or \"no\"}}.")
        _add("relation", q, SpatialOracle(
            task="relation", subject=subj.label, object_=tgt.label,
            relation=r.relation, statement_true=True, distance_m=float(r.distance)))
        # 反例：换一个不成立的关系
        other = next((x for x in ("above", "below", "to the left of")
                      if x != rel_en), "behind")
        q2 = (f"Is the {subj.label} {other} the {tgt.label}? "
              f"Answer with JSON {{\"answer\": \"yes\" or \"no\"}}.")
        _add("relation", q2, SpatialOracle(
            task="relation", subject=subj.label, object_=tgt.label,
            relation=other, statement_true=False, distance_m=float(r.distance)))

    # 2) 距离
    for a_obj, b_obj in pairs[:10]:
        d = float(a_obj.distance_to(b_obj))
        g = float(bbox_gap(a_obj, b_obj))
        q = (f"What is the center distance between the {a_obj.label} and the "
             f"{b_obj.label} in meters? "
             f"Answer with JSON {{\"distance_m\": <number>}}.")
        _add("distance", q, SpatialOracle(task="distance", distance_m=d, gap_m=g))

    # 3) 定位
    for o in objs[:8]:
        c = tuple(float(x) for x in np.asarray(o.center).reshape(-1)[:3])
        e = tuple(float(x) for x in np.asarray(o.extent).reshape(-1)[:3])
        q = (f"Where is the {o.label}? Give its 3D center as JSON "
             f"{{\"coords\": [x, y, z]}} in meters.")
        _add("locate", q, SpatialOracle(task="locate", coords=c, extent=e))

    # 4) 区域列举
    for anchor in objs:
        margin = 0.05
        lo = anchor.bbox_min - margin
        hi = anchor.bbox_max + margin
        on_it = [o for o in objs if o.obj_id != anchor.obj_id
                 and np.all(o.center >= lo) and np.all(o.center <= hi)]
        if not on_it:
            continue
        labels = tuple(sorted({o.label for o in on_it}))
        q = (f"List the object labels on the {anchor.label}. "
             f"Answer with JSON {{\"labels\": [...], \"count\": <n>}}.")
        _add("list_on", q, SpatialOracle(task="list_on", anchor=anchor.label,
                                         on_labels=labels))

    rng.shuffle(out)
    return out
