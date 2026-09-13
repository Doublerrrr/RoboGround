# -*- coding: utf-8 -*-
"""GRPO 目标函数与可验证奖励的测试（纯数学，离线可跑）。

这个文件对应面经基础拷打里的 5 道 RL 题：
GRPO 目标函数 / 组内优势 / Importance Sampling / Clip 的作用与对象 /
KL 散度 / GSPO 的 sequence-level IS / 熵坍塌。
每一道都落成一条可执行的断言 —— 面试时能说"我手写过并单测过"，
和"我看过论文"是两个层次。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from roboground.rl import (
    GrpoConfig, RewardConfig, SpatialOracle, correctness, group_advantages,
    grpo_loss, kl_penalty, parse_answer, reward_stats, sequence_ratios,
    spatial_reward, token_ratios, clipped_surrogate,
)


# ==========================================================================
# 1. 组内优势估计（GRPO 的心脏）
# ==========================================================================
def test_group_advantage_is_zero_mean_unit_std():
    """组内优势必须**零均值、近似单位标准差** —— 这是"组内相对"的定义。

    注意分母是 `std + eps`，所以标准差会略小于 1（eps=1e-4 时约 0.9997），
    这是标准 GRPO 实现的做法，不是 bug。
    """
    r = np.array([0.1, 0.5, 0.9, 0.3])
    adv, degenerate = group_advantages(r)
    assert not degenerate
    assert adv.mean() == pytest.approx(0.0, abs=1e-9)
    assert adv.std() == pytest.approx(1.0, rel=1e-3)


def test_group_advantage_degenerate_when_all_equal():
    """★ 全组同分 → 优势恒 0 → **贡献不了梯度**（GRPO 难度偏差的直接体现）。

    面经里问"GRPO 的缺点"，难度偏差就是第一个。
    这里把它落成可断言的行为：返回 `degenerate=True` 让上层能计数上报，
    而不是静默产生一堆 0 梯度。
    """
    r = np.array([0.5, 0.5, 0.5, 0.5])
    adv, degenerate = group_advantages(r)
    assert degenerate
    assert np.allclose(adv, 0.0)

    # 全对（奖励全 1）也一样退化 —— "太简单"和"太难"都是白采
    adv2, deg2 = group_advantages(np.ones(4))
    assert deg2 and np.allclose(adv2, 0.0)


def test_group_advantage_without_normalization():
    """不除标准差时只减均值 —— 尺度不同但符号一致。"""
    r = np.array([0.1, 0.5, 0.9])
    adv_n, _ = group_advantages(r, normalize=True)
    adv_raw, deg = group_advantages(r, normalize=False)
    assert not deg
    assert adv_raw.mean() == pytest.approx(0.0, abs=1e-9)
    # 排序必须一致（这是 RL 真正依赖的性质）
    assert list(np.argsort(adv_n)) == list(np.argsort(adv_raw))


# ==========================================================================
# 2. Importance Sampling 比率
# ==========================================================================
def test_token_ratios_are_exp_of_log_diff():
    """Token 级比率 = exp(logp_new - logp_old)。"""
    lp_new = np.array([[math.log(0.5), math.log(0.25)]])
    lp_old = np.array([[math.log(0.25), math.log(0.25)]])
    r = token_ratios(lp_new, lp_old)
    assert r[0, 0] == pytest.approx(2.0)
    assert r[0, 1] == pytest.approx(1.0)


def test_sequence_ratio_is_length_normalized():
    """★ GSPO 的 sequence-level IS：**整条序列共享一个比率**，且做长度归一化。

    为什么必须长度归一化：长回答的 logp 和天然更负，
    不归一化会让长回答的比率系统性偏小、被 clip 误伤。
    这里验证"同分布下，长序列与短序列得到相同比率"。
    """
    # 每条 token 的 log 概率比都是 log(1.2)，序列长度不同
    d = math.log(1.2)
    lp_new = np.array([[d, d, d, d], [d, d, 0.0, 0.0]])
    lp_old = np.zeros((2, 4))
    r = sequence_ratios(lp_new, lp_old, lengths=np.array([4, 2]))
    assert r[0] == pytest.approx(1.2, rel=1e-6)
    assert r[1] == pytest.approx(1.2, rel=1e-6), "长度不同的序列应得到同一比率"

    # 对照：token 级下，同一个序列内部各 token 比率可以不同（这就是不一致的根源）
    tok = token_ratios(lp_new, lp_old)
    assert tok[0, 0] == pytest.approx(1.2)
    assert tok[1, 2] == pytest.approx(1.0), "padding 位置比率为 1，序列内不一致"


# ==========================================================================
# 3. Clip 的作用与对象
# ==========================================================================
def test_clip_object_is_probability_ratio():
    """★ Clip 的**对象是"新旧策略概率比"**，不是奖励、不是优势。

    这是面试最常见的错误答案。用四个象限把行为钉死：

    | 优势 | 比率 | 结果 |
    |---|---|---|
    | A>0 | r=1.1（在区间内） | 照常 = r·A = 1.1 |
    | A>0 | r=3.0（超上界） | 截成 (1+ε)·A = 1.2 |
    | A>0 | r=0.5（低于 1，好动作概率在跌） | **不截** = 0.5 |
    | A<0 | r=0.5（低于下界，坏动作概率跌太猛） | 截成 (1-ε)·A = **-0.8** |

    ⚠️ 最后一行极易说反。我第一版注释写成"A<0 时不截断"，
    被这里的断言（实际是 -0.8 而不是 -0.5）抓出来了 ——
    真实语义是：**两个方向都截，截的都是"乐观方向"**。
    """
    # A>0，区间内 → 不截
    assert clipped_surrogate(np.array([1.1]), np.array([1.0]),
                             clip_eps=0.2)[0] == pytest.approx(1.1)
    # A>0，超上界 → 截成 1.2
    assert clipped_surrogate(np.array([3.0]), np.array([1.0]),
                             clip_eps=0.2)[0] == pytest.approx(1.2)
    # A>0，比率低于 1（好动作概率在跌）→ 不截，继续给梯度
    assert clipped_surrogate(np.array([0.5]), np.array([1.0]),
                             clip_eps=0.2)[0] == pytest.approx(0.5)
    # A<0，比率低于下界 → 截成 (1-ε)·A
    assert clipped_surrogate(np.array([0.5]), np.array([-1.0]),
                             clip_eps=0.2)[0] == pytest.approx(-0.8)


def test_clip_is_pessimistic_min():
    """min 让目标函数**悲观**：往好的方向走太猛会被截，往坏的方向不被截。

    这保证了单调改进的下界（PPO 论文的核心论证）。
    """
    r = np.array([2.0, 0.5])
    a = np.array([1.0, -1.0])
    obj = clipped_surrogate(r, a, clip_eps=0.2)
    unclipped = r * a
    assert np.all(obj <= unclipped + 1e-12), "min 结果必须 ≤ 未截断值"


def test_clip_higher_asymmetric_bounds():
    """★ clip-higher：上界放宽（0.28）而**下界不变**（0.2）。

    动机是**熵坍塌**：对称 clip 下低概率 token 的比率容易被上界截住，
    永远拿不到正梯度 → 策略越来越确定。
    放宽**上界**给低概率 token 留上升通道；放宽下界则会让坏样本更容易被学。
    """
    r = np.array([1.5])
    a = np.array([1.0])
    sym = clipped_surrogate(r, a, clip_eps=0.2)[0]
    higher = clipped_surrogate(r, a, clip_eps=0.2, clip_eps_high=0.28)[0]
    assert sym == pytest.approx(1.2)
    assert higher == pytest.approx(1.28), "上界放宽后不应再截到 1.2"
    # 下界不受影响
    low = clipped_surrogate(np.array([0.5]), np.array([1.0]),
                            clip_eps=0.2, clip_eps_high=0.28)[0]
    assert low == pytest.approx(0.5)


# ==========================================================================
# 4. KL 散度估计
# ==========================================================================
def test_kl_is_zero_when_identical():
    lp = np.array([-1.0, -2.0, -0.5])
    for est in ("k1", "k2", "k3"):
        assert np.allclose(kl_penalty(lp, lp, estimator=est), 0.0, atol=1e-12)


def test_kl_k3_is_always_nonnegative():
    """★ k3 估计器**恒非负** —— 这是它成为 RLHF 默认选择的核心理由。

    KL 本不该为负，但 k1（= log π_ref - log π_θ）在单样本上可以为负，
    会给出"负的惩罚"这种自相矛盾的引导。
    """
    rng = np.random.default_rng(0)
    lp = rng.normal(-3.0, 1.5, size=500)
    ref = rng.normal(-3.0, 1.5, size=500)
    k1 = kl_penalty(lp, ref, estimator="k1")
    k3 = kl_penalty(lp, ref, estimator="k3")
    assert (k1 < 0).any(), "k1 确实会出现负值"
    assert (k3 >= -1e-12).all(), "k3 必须恒非负"


def test_kl_estimators_are_unbiased_by_monte_carlo():
    """★ 三个估计器都是**无偏**的 —— 差别在**方差**，不在期望。

    这一点极易搞错（我第一版就写成了"三者应逐样本相等"，
    被测试直接打脸）。真实的关系是：

        E_{x~π_θ}[k1] = KL，E_{x~π_θ}[k3] = KL     ← 都无偏
        Var(k3) ≪ Var(k1)                          ← 这才是选 k3 的理由

    做法：构造一对离散分布 (p 策略, q 参考)，从 p 采样，
    用蒙特卡洛验证两个估计器的**均值**都收敛到真实 KL，
    且 k3 的**标准差显著更小**。
    """
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.2, 0.3, 0.3, 0.2])
    true_kl = float(np.sum(p * np.log(p / q)))

    rng = np.random.default_rng(0)
    n = 200_000
    idx = rng.choice(len(p), size=n, p=p)
    d = np.log(q[idx]) - np.log(p[idx])          # log(π_ref/π_θ)

    k1 = kl_penalty(np.zeros(n), d, estimator="k1")   # -d
    k3 = kl_penalty(np.zeros(n), d, estimator="k3")   # exp(d)-1-d

    # 均值都应逼近真实 KL（相对误差 <3%）
    assert k1.mean() == pytest.approx(true_kl, rel=0.03)
    assert k3.mean() == pytest.approx(true_kl, rel=0.03)
    # k3 方差显著更小 —— 这是它成为实践默认的真正原因
    assert k3.std() < k1.std(), f"k3 方差应更小：k3={k3.std():.4f} k1={k1.std():.4f}"


def test_unknown_kl_estimator_raises():
    with pytest.raises(ValueError):
        kl_penalty(np.array([-1.0]), np.array([-1.1]), estimator="k9")


# ==========================================================================
# 5. 完整 GRPO 损失
# ==========================================================================
def _make_batch(n=4, t=5, seed=0):
    rng = np.random.default_rng(seed)
    lp_old = rng.normal(-1.5, 0.3, size=(n, t))
    lp_new = lp_old + rng.normal(0.0, 0.05, size=(n, t))
    lp_ref = lp_old - rng.normal(0.0, 0.05, size=(n, t))
    mask = np.ones((n, t))
    lengths = np.full(n, t)
    adv = np.array([1.0, 1.0, -1.0, -1.0])
    return lp_new, lp_old, lp_ref, adv, lengths, mask


def test_grpo_loss_runs_and_reports_diagnostics():
    """损失函数必须同时给出**全部诊断量**（否则无法定位问题）。"""
    lp_new, lp_old, lp_ref, adv, lengths, mask = _make_batch()
    out = grpo_loss(lp_new, lp_old, lp_ref, adv, lengths, mask)
    assert np.isfinite(out.loss)
    assert out.n_tokens == 20
    assert 0.0 <= out.clip_frac <= 1.0
    assert 0.0 <= out.zero_adv_frac <= 1.0
    assert np.isfinite(out.entropy)
    assert out.mean_ratio > 0


def test_grpo_loss_zero_when_advantages_all_zero():
    """★ 优势全 0 → 策略损失必须为 0（难度偏差的后果）。"""
    lp_new, lp_old, lp_ref, _, lengths, mask = _make_batch()
    adv = np.zeros(4)
    out = grpo_loss(lp_new, lp_old, lp_ref, adv, lengths, mask,
                    GrpoConfig(beta_kl=0.0))
    assert out.policy_loss == pytest.approx(0.0, abs=1e-12)


def test_grpo_loss_sequence_level_differs_from_token_level():
    """★ Token 级与 Sequence 级 IS **会给出不同的损失** —— 这就是 GSPO 的动机。

    构造一个"序列内部比率不一致"的 batch（部分 token 比率很高、部分很低），
    两种层级的差异就会被放大。
    """
    n, t = 4, 6
    lp_old = np.zeros((n, t))
    lp_new = np.zeros((n, t))
    # 让同一序列内比率差异很大（一半 token 比率 2.0，一半 1.0）
    lp_new[:, : t // 2] = math.log(2.0)
    mask = np.ones((n, t))
    lengths = np.full(n, t)
    adv = np.array([1.0, 1.0, -1.0, -1.0])
    tok = grpo_loss(lp_new, lp_old, None, adv, lengths, mask,
                    GrpoConfig(level="token", beta_kl=0.0, clip_eps_high=None))
    seq = grpo_loss(lp_new, lp_old, None, adv, lengths, mask,
                    GrpoConfig(level="sequence", beta_kl=0.0, clip_eps_high=None))
    assert tok.policy_loss != pytest.approx(seq.policy_loss), \
        "两种层级的损失应当不同，否则说明 level 开关没生效"


def test_grpo_loss_kl_only_when_reference_given():
    """β>0 但没有参考模型时不应崩，且 KL 记 0。"""
    lp_new, lp_old, _, adv, lengths, mask = _make_batch()
    out = grpo_loss(lp_new, lp_old, None, adv, lengths, mask,
                    GrpoConfig(beta_kl=0.05))
    assert out.kl == 0.0
    assert np.isfinite(out.loss)


def test_grpo_loss_handles_all_masked():
    """全被 mask 掉的 batch 必须安全返回，而不是 NaN。"""
    n, t = 2, 3
    zeros = np.zeros((n, t))
    out = grpo_loss(zeros, zeros, None, np.array([1.0, -1.0]),
                    np.zeros(n), np.zeros((n, t)))
    assert out.loss == 0.0 and out.n_tokens == 0


# ==========================================================================
# 6. 可验证奖励（规则引擎当 verifier）
# ==========================================================================
def test_parse_json_answer():
    got = parse_answer('{"answer": "yes"}', "relation")
    assert got and got["answer"] == "yes" and got["_via"] == "json"


def test_parse_json_survives_markdown_fence_and_prose():
    txt = 'Sure! Here it is:\n```json\n{"distance_m": 2.53}\n```\nHope that helps.'
    got = parse_answer(txt, "distance")
    assert got and got["distance_m"] == pytest.approx(2.53)
    assert got["_via"] == "json"


def test_parse_regex_fallback_for_relation():
    """★ 正则兜底路径：让"说人话但说对了"也能拿分。

    如果只认 JSON，RL 早期整组格式分都是 0 → **组内优势全 0 → 学不动**。
    这是"冷启动"问题的具体处置。
    """
    got = parse_answer("Yes, the cup is to the left of the table.", "relation")
    assert got and got["answer"] == "yes"
    assert got["_via"] == "regex"


def test_regex_fallback_prefers_negation():
    """'is not to the left' 里也含 'left'，必须先判否定。"""
    got = parse_answer("No, the cup is not to the left of the table.", "relation")
    assert got and got["answer"] == "no"


def test_parse_returns_none_for_garbage():
    assert parse_answer("", "relation") is None
    assert parse_answer("嗯……我不知道。", "distance") is None


def test_reward_relation_binary():
    oracle = SpatialOracle(task="relation", statement_true=True)
    good = spatial_reward('{"answer":"yes"}', oracle)
    bad = spatial_reward('{"answer":"no"}', oracle)
    assert good.correct == 1.0 and bad.correct == 0.0
    assert good.total > bad.total


def test_reward_distance_is_shaped_not_binary():
    """★ 数值任务必须用**塑形**奖励，否则 RL 早期梯度全 0（稀疏奖励问题）。

    验证：误差 2.7m / 1.3m / 0.4m 得到的奖励**单调递增**，
    而不是"只有 <0.3m 才给分"。这正是"能不能学得动"的关键。
    """
    oracle = SpatialOracle(task="distance", distance_m=2.0)
    r_far = spatial_reward('{"distance_m": 4.7}', oracle).correct   # 误差 2.7
    r_mid = spatial_reward('{"distance_m": 3.3}', oracle).correct   # 误差 1.3
    r_close = spatial_reward('{"distance_m": 2.4}', oracle).correct  # 误差 0.4
    assert 0.0 < r_far < r_mid < r_close < 1.0
    # 全部**非零** —— 这是塑形奖励相对二值奖励的关键优势
    assert r_far > 0.0, "误差大也应该拿到非零奖励，否则没有梯度"


def test_reward_coord_uses_euclidean_distance():
    oracle = SpatialOracle(task="locate", coords=(1.0, 0.0, 0.85))
    same = spatial_reward('{"coords":[1.0,0.0,0.85]}', oracle)
    off = spatial_reward('{"coords":[1.0,0.0,1.85]}', oracle)
    assert same.correct == pytest.approx(1.0)
    assert off.correct == pytest.approx(math.exp(-1.0 / 1.0), rel=1e-6)


def test_reward_format_gate_zeroes_unparseable():
    """格式门控：不可解析 → 总奖励置 0（不给"乱写也有分"的空间）。"""
    oracle = SpatialOracle(task="relation", statement_true=True)
    bd = spatial_reward("完全不相关的胡言乱语", oracle)
    assert bd.format_ok == 0.0 and bd.total == 0.0
    assert bd.reason == "格式不可解析"


def test_reward_list_on_uses_f1_and_count_bonus():
    oracle = SpatialOracle(task="list_on", anchor="table",
                           on_labels=("cup", "bottle"))
    full = spatial_reward('{"labels":["cup","bottle"],"count":2}', oracle)
    partial = spatial_reward('{"labels":["cup"],"count":1}', oracle)
    assert full.correct >= 0.9 > partial.correct


def test_reward_stats_flags_format_hacking():
    """★ 诊断 reward hacking：格式分高、正确性分低 → 疑似在刷格式分。"""
    oracle = SpatialOracle(task="relation", statement_true=True)
    hacking = [spatial_reward('{"answer":"no"}', oracle) for _ in range(8)]
    s = reward_stats(hacking)
    assert s["format_rate"] == pytest.approx(1.0)
    assert s["correct_mean"] == pytest.approx(0.0)
    assert s["format_correct_ratio"] > 1.0  # 哨兵指标

    learning = [spatial_reward('{"answer":"yes"}', oracle) for _ in range(8)]
    s2 = reward_stats(learning)
    assert s2["correct_mean"] == pytest.approx(1.0)


def test_reward_zero_advantage_scenario():
    """★ 把"稀疏奖励 → 优势退化"整条链路走通一遍。

    场景：全组回答格式都对但答案都错（或都错得一样），
    正确性奖励全 0 → 组内优势退化 → 梯度为 0。
    这是"稀疏奖励"与"难度偏差"两个问题的交汇点。
    """
    oracle = SpatialOracle(task="relation", statement_true=True)
    rewards = np.array([spatial_reward('{"answer":"no"}', oracle).correct
                        for _ in range(4)])
    adv, degenerate = group_advantages(rewards)
    assert degenerate and np.allclose(adv, 0.0)

    # 但如果用塑形奖励（数值任务），即使都错得离谱也会有不同的分 → 有梯度
    oracle_d = SpatialOracle(task="distance", distance_m=2.0)
    shaped = np.array([
        spatial_reward(f'{{"distance_m": {v}}}', oracle_d).correct
        for v in (5.0, 4.0, 3.0, 2.5)
    ])
    adv2, deg2 = group_advantages(shaped)
    assert not deg2, "塑形奖励下不应退化"
    assert adv2[0] < adv2[-1], "误差小的应拿到更高的优势"


# ==========================================================================
# 7. 参考实现 vs 可微实现（防止两份实现漂移）
# ==========================================================================
torch = pytest.importorskip("torch", reason="等价性测试需要 torch")


@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("clip_high", [None, 0.28])
@pytest.mark.parametrize("beta_kl", [0.0, 0.05])
def test_torch_loss_matches_numpy_reference(level, clip_high, beta_kl):
    """★ 可微实现必须与 numpy 参考实现**数值一致**。

    这个测试的价值在于：`grpo.py` 是"给人看、给测试验证"的参考实现，
    `torch_loss.py` 是真正训练用的可微实现。
    两份实现一旦漂移（常见于"改了参考实现忘了同步优化实现"），
    训练出来的东西就和文档里写的不一样了，而且**不报错**。
    """
    import torch as T

    from roboground.rl import grpo_loss
    from roboground.rl.torch_loss import grpo_loss_torch

    rng = np.random.default_rng(7)
    n, t = 5, 6
    lp_old = rng.normal(-1.5, 0.3, size=(n, t))
    lp_new = lp_old + rng.normal(0.0, 0.08, size=(n, t))
    lp_ref = lp_old - rng.normal(0.0, 0.05, size=(n, t))
    mask = np.ones((n, t), dtype=np.float32)
    mask[0, 4:] = 0.0                       # 制造 padding 差异
    lengths = mask.sum(axis=1)
    adv = np.array([1.2, 0.9, -0.7, -1.1, 0.3])

    cfg = GrpoConfig(level=level, clip_eps=0.2, clip_eps_high=clip_high,
                     beta_kl=beta_kl, kl_estimator="k3")

    ref = grpo_loss(lp_new, lp_old, 
                    (lp_ref if beta_kl > 0 else None), adv, lengths, mask, cfg)

    tt = lambda x: T.tensor(x, dtype=T.float32, requires_grad=True)  # noqa: E731
    loss_t, diag = grpo_loss_torch(
        tt(lp_new), tt(lp_old),
        (tt(lp_ref) if beta_kl > 0 else None),
        T.tensor(adv, dtype=T.float32), T.tensor(lengths, dtype=T.float32),
        T.tensor(mask, dtype=T.float32), cfg)

    assert diag.policy_loss == pytest.approx(ref.policy_loss, rel=1e-5), \
        f"策略损失不一致：torch={diag.policy_loss} numpy={ref.policy_loss}"
    assert diag.loss == pytest.approx(ref.loss, rel=1e-5)
    assert diag.kl == pytest.approx(ref.kl, abs=1e-6)
    assert diag.clip_frac == pytest.approx(ref.clip_frac, abs=1e-6)
    assert diag.mean_ratio == pytest.approx(ref.mean_ratio, rel=1e-5)
    assert loss_t.requires_grad, "可微实现必须保留计算图"


def test_torch_loss_actually_produces_gradients():
    """★ 损失必须能把梯度传回输入 —— 防"假反向"。

    踩过的坑：训练脚本里写成"用 numpy 算 loss，再包一个 torch 标量 backward"，
    梯度根本传不回模型。表现是**训练看起来在跑、loss 也在变，但参数一动没动**，
    靠看曲线发现不了。
    """
    import torch as T

    from roboground.rl.torch_loss import grpo_loss_torch

    n, t = 4, 5
    lp_old = T.zeros(n, t)
    lp_ref = T.zeros(n, t)
    lp_new = T.full((n, t), -1.0, requires_grad=True)
    mask = T.ones(n, t)
    lengths = T.full((n,), float(t))
    adv = T.tensor([1.0, 1.0, -1.0, -1.0])

    loss, _ = grpo_loss_torch(lp_new, lp_old, lp_ref, adv, lengths, mask,
                              GrpoConfig(beta_kl=0.01))
    loss.backward()
    assert lp_new.grad is not None, "梯度必须传到输入"
    assert float(lp_new.grad.abs().sum()) > 0.0, "梯度不能全 0"
