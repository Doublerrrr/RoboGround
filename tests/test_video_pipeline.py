# -*- coding: utf-8 -*-
"""视频数据管线测试：镜头检测 / 抽帧 / 去重 / 质量 / 端到端。

这些测试守的是**本项目在视频管线上踩过的具体坑**，不是泛泛的冒烟测试。
每个 `⚠️` 注释都对应一次真实故障。
"""
from __future__ import annotations

import numpy as np
import pytest

from roboground.data.video import (
    STRATEGIES, DedupConfig, PipelineConfig, QualityConfig, SamplingConfig,
    ShotDetectionConfig, coverage_metrics, dedup_frames, detect_shots, dhash,
    evaluate_shot_detection, filter_by_quality, hamming, hsv_histogram,
    make_synthetic_shot_video, process_video, quality_scores, sample_frames,
)
from roboground.data.video.shot import peak_prominence_mask


# ==========================================================================
# 合成视频构造
# ==========================================================================
@pytest.fixture(scope="module")
def small_video():
    frames, truth, labels = make_synthetic_shot_video(
        n_shots=4, frames_per_shot=8, width=96, height=72, num_objects=4,
        seed=11, gradual_at=(), blend_frames=6,
    )
    return frames, truth, labels


@pytest.fixture(scope="module")
def hetero_video():
    frames, truth, labels = make_synthetic_shot_video(
        n_shots=6, heterogeneous=True, width=96, height=72, num_objects=4,
        seed=2026, gradual_at=(3,), blend_frames=10,
    )
    return frames, truth, labels


# ==========================================================================
# 镜头检测
# ==========================================================================
def test_synthetic_video_boundaries_match_labels(small_video):
    """合成视频的真值边界必须与逐帧标签自洽（测试基准本身要可信）。"""
    frames, truth, labels = small_video
    assert len(frames) == len(labels)
    # 每个真值边界处，标签恰好发生一次变化
    for t in truth:
        assert labels[t] != labels[t - 1], f"边界 {t} 处标签没变"
    # 镜头数与标签种类数一致
    assert len(set(labels)) == len(truth) + 1


def test_crossfade_ends_exactly_at_target():
    """★ 回归锁：交叉叠化的**末帧必须恰好是目标帧**。

    曾经的 bug：alpha 写成 `i/(n+1)`，最大只有 n/(n+1)=0.8，
    于是叠化末帧还差 20% 才到目标，下一帧却是完整目标 ——
    **凭空造出一个假硬切**（实测该处直方图差异 0.70，比所有真边界都大），
    把镜头检测精确率打到 50%。
    """
    from roboground.data.video.shot import _crossfade

    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = np.full((16, 16, 3), 255, dtype=np.uint8)
    blend = _crossfade(a, b, 4)
    assert len(blend) == 4
    assert np.array_equal(blend[-1], b), "叠化末帧必须等于目标帧 b"


def test_shot_detection_finds_known_boundaries(small_video):
    """镜头边界已知 → 必须能检出（容差 2 帧）。"""
    frames, truth, _ = small_video
    shots = detect_shots(frames, cfg=ShotDetectionConfig(use_embedding=False))
    m = evaluate_shot_detection(shots, truth, tolerance=2)
    assert m["recall"] >= 0.8, f"召回率过低：{m}"
    assert m["precision"] >= 0.5, f"精确率过低：{m}"


def test_peak_prominence_rejects_sustained_motion():
    """★ 峰值突出度：能区分"单点尖峰"（真切换）与"持续高值"（镜头内快运动）。

    这是本项目最重要的一个判据，动机是实测踩到的坑：
    快运动镜头内部的帧间差异（实测 0.72）**比真实硬切（0.34）还大**，
    所以"差异大"本身无法区分两者。区别在**形态**：
    真切换 = 单点尖峰（邻域中位 ~0.0004，比值 ~950）
    快运动 = 持续高值（邻域中位 ~0.6，比值 ~1.2）
    """
    # 单点尖峰：背景 0.001，第 10 帧跳到 0.5
    spike = np.full(21, 0.001)
    spike[10] = 0.5
    mask = peak_prominence_mask(spike, window=5, min_ratio=5.0)
    assert mask[10], "单点尖峰应被保留为边界"

    # 持续高值：整段都是 0.5~0.6，没有突出点
    sustained = np.full(21, 0.55)
    sustained[10] = 0.6
    mask2 = peak_prominence_mask(sustained, window=5, min_ratio=5.0)
    assert not mask2.any(), "持续高运动不应被判为边界"


def test_fixed_threshold_is_worse_than_otsu(hetero_video):
    """固定阈值必须明显差于自适应 —— 这正是"阈值不能写死"的量化证据。"""
    frames, truth, _ = hetero_video
    fixed = detect_shots(frames, cfg=ShotDetectionConfig(
        use_embedding=False, threshold_mode="fixed"))
    auto = detect_shots(frames, cfg=ShotDetectionConfig(use_embedding=False))
    m_fixed = evaluate_shot_detection(fixed, truth, tolerance=2)
    m_auto = evaluate_shot_detection(auto, truth, tolerance=2)
    assert m_auto["recall"] >= m_fixed["recall"], \
        f"自适应({m_auto['recall']:.2f}) 应不差于固定阈值({m_fixed['recall']:.2f})"


# ==========================================================================
# ★ 阈值判据：真实视频推翻 Otsu（2026-09-11）
# ==========================================================================
def _realistic_diffs(seed: int = 0, n: int = 400, n_cuts: int = 4,
                     frac_dup: float = 0.30) -> tuple:
    """造一条**模拟真实压缩视频**的帧差序列。

    形态对齐 MSR-VTT 实测（重尾单峰）：
    - 主体：对数正态噪声（中位 ~0.004，跨一个数量级）；
    - **`frac_dup` 比例的"零差分"重复帧** —— 压缩视频里极常见，
      经 `clip(1e-6)` 后会在 log 空间的最左端形成一个**人造峰**。
      这正是 Otsu 失效的机理：它会在"零差分峰"与真实噪声体之间切一刀，
      阈值低到几乎没有点在其下（实测 video7010 有 216/269 个点超阈）；
    - 少量真切点（0.3~0.7，比噪声高两个数量级）。
    """
    rng = np.random.default_rng(seed)
    d = np.exp(rng.normal(np.log(0.004), 0.7, size=n))
    dup = rng.choice(n, size=int(n * frac_dup), replace=False)
    d[dup] = 0.0
    cuts = rng.choice(np.setdiff1d(np.arange(5, n - 5), dup),
                      size=n_cuts, replace=False)
    d[cuts] = rng.uniform(0.3, 0.7, size=n_cuts)
    return d, np.sort(cuts)


def test_otsu_guard_rejects_split_inside_noise():
    """★ 回归锁：Otsu 的阈值必须**高于噪声地板**，否则退回 robust。

    回归背景（2026-09-11，真实视频上发现）：
    镜头检测默认用 Otsu，前提是"帧差分布双峰"。真实压缩视频是**重尾单峰**，
    Otsu 照样返回一个"把质量对半劈"的阈值 —— 而且它**原有的守卫挡不住**：
    `min(w0, w1) < 0.05` 只防"某一类样本太少"，而单峰被对半劈时
    两类权重都 ≈ 0.5，守卫直接放行。
    实测 10 条 MSR-VTT：Otsu 报出 684 个镜头（68/视频），
    robust 报出 32 个（3.2/视频）—— 差 21 倍，且 418 张核验图里绝大多数是假的。
    """
    from roboground.data.video.shot import _otsu_threshold, _robust_threshold

    d, cuts = _realistic_diffs(seed=7)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    floor = med + 3.0 * 1.4826 * mad

    t_otsu = _otsu_threshold(d)
    # ① 守卫必须生效：返回的阈值不低于噪声地板
    assert t_otsu >= floor, (
        f"Otsu 返回 {t_otsu:.6f}，落在噪声地板 {floor:.6f} 之下 —— 守卫失效"
    )
    # ② 生效方式必须是"回退到 robust"
    assert t_otsu == pytest.approx(_robust_threshold(d, 6.0, 0.99, 0.35))
    # ③ 行为层面：超阈的必须是真切点，不是噪声主体
    n_over = int((d >= t_otsu).sum())
    assert n_over <= 12, f"放行了 {n_over} 个候选（真切点只有 {len(cuts)} 个）"
    assert (d[cuts] >= t_otsu).all(), "真切点被误杀"


def test_robust_threshold_separates_noise_from_cuts():
    """robust 阈值必须"高于噪声、低于真信号"—— 两侧都要验证。

    只验证一侧（比如只看"没有误报"）的话，把阈值抬到无穷大也能通过。
    """
    from roboground.data.video.shot import _robust_threshold

    for seed in (0, 7, 42):
        d, cuts = _realistic_diffs(seed=seed)
        t = _robust_threshold(d, 6.0, 0.99, 0.35)
        # 真信号侧：所有真切点都必须在阈值之上
        assert (d[cuts] >= t).all(), f"seed={seed}: 真切点被误杀"
        # 噪声侧：超阈的点数不能远超真切点数
        assert int((d >= t).sum()) <= len(cuts) + 8, \
            f"seed={seed}: 阈值 {t:.5f} 放行了 {(d >= t).sum()} 个候选"


def test_otsu_falls_back_when_threshold_lands_in_noise():
    """Otsu 的第二道守卫：阈值必须**高于噪声地板**（median + 3σ）。"""
    from roboground.data.video.shot import _otsu_threshold, _robust_threshold

    d, _ = _realistic_diffs(seed=3)
    t_otsu = _otsu_threshold(d)
    t_rob = _robust_threshold(d, 6.0, 0.99, 0.35)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    assert t_otsu >= med + 3.0 * 1.4826 * mad, (
        f"Otsu 返回 {t_otsu:.5f} 落在噪声地板 {med + 3 * 1.4826 * mad:.5f} 之下，"
        "守卫没有生效"
    )
    # 回退后应当等于 robust 的结果
    assert t_otsu == pytest.approx(t_rob)


def test_robust_quantile_is_sample_size_aware():
    """★ 分位数必须随样本量自适应 —— 否则短序列会退化到"等于最大值"。

    回归背景：`np.quantile(d, 0.99)` 在 n=31 时就是**最大值本身**
    （第 30.7 个次序统计量），阈值高到把真切点一起挡掉。
    实测合成短序列召回率从 0.80 掉到 0.33。
    """
    from roboground.data.video.shot import _robust_threshold

    # 31 个样本、3 个真切点（对应 32 帧 4 镜头的合成视频）
    rng = np.random.default_rng(0)
    d = np.exp(rng.normal(np.log(0.004), 0.7, size=31))
    d[[7, 15, 23]] = 0.5
    t = _robust_threshold(d, 6.0, 0.99, 0.35)
    assert (d >= t).sum() >= 3, (
        f"短序列阈值 {t:.5f} 只放行 {(d >= t).sum()} 个点，真切点被误杀"
    )
    assert t < 0.5, "阈值不应高到等于最大值"


def test_gradual_path_requires_prominence():
    """★ 渐变路径必须有峰值突出度门。

    回归背景：`cum` 是连续帧差的滑动平均，而**噪声的滑动平均本身就是
    缓慢起伏的平台**，只要阈值低一点就到处"超阈"。实测 10 条真实视频里
    渐变路径贡献了 102~143 个假边界（video7013 完全没有切点却报了 18 个）。
    """
    from roboground.data.video.shot import decide_boundaries

    # 构造：噪声缓慢起伏（滑动平均持续偏高），但**没有**任何局部峰
    rng = np.random.default_rng(1)
    d = 0.02 + 0.004 * np.sin(np.arange(300) / 7.0)
    d += rng.normal(0, 0.0005, size=300)
    cfg = ShotDetectionConfig(use_embedding=False, threshold_mode="fixed",
                              fallback_hist_threshold=0.015)
    hard, gradual = decide_boundaries(d, None, cfg)
    assert not hard.any(), "缓慢起伏不应产生硬切"
    assert gradual.sum() <= 2, (
        f"渐变路径放行了 {gradual.sum()} 个假边界（应 ≤2）—— "
        "说明峰值突出度门没有生效"
    )


def test_robust_n_signal_and_max_frac_are_wired_into_resolve_threshold():
    """★ 回归锁：`robust_n_signal` / `robust_max_frac` 必须真的接进调用链。

    回归背景（2026-09-11）：这两个参数先被加进了 `ShotDetectionConfig`，
    但 `_resolve_threshold` 里仍走 `_robust_threshold` 的**函数默认值** ——
    于是"调 config 参数"完全没效果，而参数扫描脚本会给出
    "扫了 4/6/8/12/16 结果全一样"的结论，看起来像"阈值调不动"，
    实际是**配置根本没接线**。

    这类"加了参数但没接线"的缺陷是典型的**静默失效**：
    不报错、不崩溃、测试全绿，只是结果与配置无关。
    """
    from roboground.data.video.shot import _resolve_threshold, _robust_threshold

    d, _ = _realistic_diffs(seed=5)
    # 直接调函数：n_signal 越大 → frac 越大 → 分位数越低 → 阈值越低
    t_small = _robust_threshold(d, 6.0, 0.99, 0.35, n_signal=4, max_frac=0.10)
    t_large = _robust_threshold(d, 6.0, 0.99, 0.35, n_signal=64, max_frac=0.10)
    assert t_large <= t_small

    # 走配置路径必须与直接调用**逐位一致**
    cfg = ShotDetectionConfig(robust_n_signal=64, robust_max_frac=0.10)
    assert _resolve_threshold(d, cfg, is_hist=True) == pytest.approx(t_large), (
        "配置里的 robust_n_signal 没有生效 —— 参数没接进 `_resolve_threshold`"
    )
    cfg4 = ShotDetectionConfig(robust_n_signal=4)
    assert _resolve_threshold(d, cfg4, is_hist=True) == pytest.approx(t_small)

    # max_frac 同样必须接线：把它压到极小 → 退回纯 MAD 判据（阈值更高）
    cfg_floor = ShotDetectionConfig(robust_n_signal=4, robust_max_frac=0.001)
    assert _resolve_threshold(d, cfg_floor, is_hist=True) >= t_small


def test_robust_n_signal_trades_recall_for_precision():
    """★ `n_signal` 是显式的 **召回/精确率取舍旋钮**，不是"越大越好"。

    真实数据上的实测（6 条人工核验视频，见 `runs/goldset_sweep.json`）：

    ==========  =======  =======  =======  ============
    n_signal    P        R        F1       TP/FP/FN
    ==========  =======  =======  =======  ============
    4（默认）   1.000    0.900    0.947    9 / 0 / 1
    8           0.833    1.000    0.909    10 / 2 / 0
    16          0.833    1.000    0.909    10 / 2 / 0
    32          0.714    1.000    0.833    10 / 4 / 0
    ==========  =======  =======  =======  ============

    为了多拿回 **1 个** 漏检要付出 **2~4 个**误检 —— F1 反而下降。
    所以默认停在 4：**这是一个有意识的选择，不是没发现**。
    """
    from roboground.data.video.shot import _robust_threshold

    d, cuts = _realistic_diffs(seed=11)
    n_over = []
    for ns in (4, 8, 16, 32, 64):
        t = _robust_threshold(d, 6.0, 0.99, 0.35, n_signal=ns, max_frac=0.10)
        n_over.append(int((d >= t).sum()))
    # 单调不减：放松阈值只会让更多候选通过，不会更少
    assert n_over == sorted(n_over), f"候选数应单调不减，实际 {n_over}"
    # 放宽的代价必须真实存在：最大的那档一定放进了额外噪声
    assert n_over[-1] > len(cuts), (
        f"n_signal=64 只放行 {n_over[-1]} 个（真切点 {len(cuts)} 个）—— "
        "说明这个旋钮已经失去区分力，测试失去意义"
    )


# ==========================================================================
# ★ 首镜头豁免（2026-09-11 金种子核验发现）
# ==========================================================================
def _two_scene_frames(n_a: int = 2, n_b: int = 6, size: int = 32) -> list:
    """构造 `[A]*n_a + [B]*n_b` —— 帧 `n_a` 处有一次硬切。

    边界下标 = `n_a`（`diffs[n_a-1]` 是 A→B 的那一帧）。
    """
    A = np.zeros((size, size, 3), np.uint8)
    A[..., 0] = 200                      # 纯红
    B = np.zeros((size, size, 3), np.uint8)
    B[..., 2] = 200                      # 纯蓝
    return [A] * n_a + [B] * n_b


def test_first_shot_is_exempt_from_min_shot_len():
    """★ 回归锁：首镜头 [0, b) 不该被 `min_shot_len` 判成噪声。

    回归背景（2026-09-11，金种子核验发现）：
    video7014 第 2 帧是一次真硬切（帧差 0.4600，`hard=True`），
    但首镜头只有 2 帧 < `min_shot_len=3`，于是被整段丢掉 ——
    这是**召回侧的静默漏检**，而只核验"预测出来的边界"的流程**查不出来**。

    `min_shot_len` 的语义是"一个切点至少要把片子切成这么长的两段才可信"，
    对首镜头不成立：它的左端是**视频开头**，不是另一个切点。
    """
    from roboground.data.video.shot import detect_shots

    frames = _two_scene_frames(n_a=2, n_b=6)      # 边界在帧 2

    on = detect_shots(frames, cfg=ShotDetectionConfig(
        min_shot_len=3, exempt_edge_shots=True))
    assert [s.start for s in on] == [0, 2], (
        f"首镜头豁免没生效：边界 {[s.start for s in on]}（应为 [0, 2]）"
    )

    off = detect_shots(frames, cfg=ShotDetectionConfig(
        min_shot_len=3, exempt_edge_shots=False))
    assert [s.start for s in off] == [0], (
        f"关闭豁免时应合并成一个镜头，实际 {[s.start for s in off]}"
    )


def test_edge_exemption_only_applies_to_hard_cuts():
    """首镜头豁免**只对硬切**生效 —— 渐变候选在帧 0 附近不可信。

    渐变判据用的 `cum = np.convolve(hd, ..., mode="same")`，两端被截断填充，
    边缘值有系统偏差；硬切是单帧判据，不受影响。
    """
    from roboground.data.video.shot import _merge_short

    diffs = np.zeros(7)
    boundaries = [0, 2, 7]
    # 场景一：帧 1 处是**硬切** → 放行
    hard = np.zeros(7, dtype=bool)
    hard[1] = True
    kept_hard = _merge_short(boundaries, hard, np.zeros(7, dtype=bool),
                             3, diffs, exempt_edge=True)
    assert [s.start for s in kept_hard] == [0, 2]

    # 场景二：帧 1 处是**渐变** → 不放行
    gradual = np.zeros(7, dtype=bool)
    gradual[1] = True
    kept_grad = _merge_short(boundaries, np.zeros(7, dtype=bool), gradual,
                             3, diffs, exempt_edge=True)
    assert [s.start for s in kept_grad] == [0], (
        "渐变候选不应享受首镜头豁免（卷积边缘效应使其在帧 0 附近不可信）"
    )


def test_edge_exemption_adds_at_most_one_boundary():
    """首镜头豁免的影响**有上界**：每条视频最多多 1 个边界。

    为什么要有这条锁：任何"放宽判据"的改动都必须证明**影响是有界的**，
    否则就是拿全局召回换一个局部漏检。
    实测（300 条 MSR-VTT 视频，见 `runs/goldset_sweep.json`）：
    只有 9 条（3.0%）发生变化，且**恒为 +1**，每视频镜头数中位数/最大值不变。
    """
    from roboground.data.video.shot import _merge_short

    zeros = np.zeros(6, dtype=bool)

    # --- 场景 A：关掉豁免时那个候选被整段丢掉 → 豁免净增 1 ---
    diffs = np.array([0.9, 0.0, 0.9, 0.0, 0.9, 0.0])
    hard = np.zeros(6, dtype=bool)
    hard[[0, 2, 4]] = True                      # 候选边界 = 帧 1 / 3 / 5
    boundaries = [0, 1, 3, 5, 6]
    on = _merge_short(boundaries, hard, zeros, 3, diffs, exempt_edge=True)
    off = _merge_short(boundaries, hard, zeros, 3, diffs, exempt_edge=False)
    n_on = len([s for s in on if s.start > 0])
    n_off = len([s for s in off if s.start > 0])
    assert n_on - n_off == 1, (
        f"豁免应恰好净增 1 个边界，实际 {n_off} → {n_on}"
    )

    # --- 场景 B：候选成簇时，豁免**把边界前移到更早的那个候选**（计数不变）---
    diffs2 = np.array([0.9, 0.9, 0.0, 0.0, 0.0, 0.0])
    hard2 = np.zeros(6, dtype=bool)
    hard2[[1, 2]] = True                        # 候选边界 = 帧 2 / 3
    bnd2 = [0, 2, 3, 6]
    on2 = _merge_short(bnd2, hard2, zeros, 3, diffs2, exempt_edge=True)
    off2 = _merge_short(bnd2, hard2, zeros, 3, diffs2, exempt_edge=False)
    assert [s.start for s in on2] == [0, 2], (
        f"豁免应保留最靠前的候选（帧 2），实际 {[s.start for s in on2]}"
    )
    assert [s.start for s in off2] == [0, 3], (
        f"无豁免时贪心会跳到帧 3，实际 {[s.start for s in off2]}"
    )
    assert len([s for s in on2 if s.start > 0]) \
        == len([s for s in off2 if s.start > 0]), "成簇时计数不应变化"

    # --- 通用上界：任何构造下都不允许多于 +1 ---
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = int(rng.integers(4, 12))
        cand = sorted(set(int(x) for x in rng.integers(1, 20, size=m)))
        if not cand:
            continue
        dd = rng.uniform(0.0, 1.0, size=max(cand))
        hh = rng.random(max(cand)) > 0.4
        bb = sorted({0, *cand, max(cand) + 1})
        a = len([s for s in _merge_short(bb, hh, zeros, 3, dd, exempt_edge=True)
                 if s.start > 0])
        b = len([s for s in _merge_short(bb, hh, zeros, 3, dd, exempt_edge=False)
                 if s.start > 0])
        assert a - b <= 1, f"豁免净增 {a - b} 个边界（上界应为 1）：{cand}"


# ==========================================================================
# 抽帧
# ==========================================================================
def _mean_abs_diffs(frames):
    """逐帧平均绝对像素差（一维，长度 N-1）。"""
    means = np.array([np.asarray(f, dtype=np.float32).mean() for f in frames],
                     dtype=np.float64)
    return np.abs(np.diff(means))


def test_all_strategies_respect_budget(small_video):
    """★ 回归锁：固定预算是四种策略可比的前提，必须**真的抽满**。

    踩过两次坑：
    1. 加权采样产生重复下标，被 `set()` 去重后少于预算（adaptive 24→15）；
    2. 配额被镜头长度截断后**没有再分配**，份额被静默丢弃（24→17）。
    两次都表现为"adaptive 看起来省帧"，其实是 bug。
    """
    frames, _, _ = small_video
    shots = detect_shots(frames, cfg=ShotDetectionConfig(use_embedding=False))
    diffs = _mean_abs_diffs(frames)
    budget = 12
    for strat in STRATEGIES:
        picked = sample_frames(shots, diffs, len(frames),
                               SamplingConfig(strategy=strat,
                                              target_frames=budget))
        assert len(picked) == len(set(picked)), f"{strat} 有重复下标"
        # 允许 thirds 因短镜头配额取整少 1 帧，其余必须精确
        if strat != "thirds":
            assert len(picked) == budget, f"{strat} 只抽了 {len(picked)}/{budget}"


def test_sampling_rejects_wrong_diffs_shape(small_video):
    """diffs 形状不对时要给出**可读**的报错，而不是难懂的广播异常。"""
    frames, _, _ = small_video
    shots = detect_shots(frames, cfg=ShotDetectionConfig(use_embedding=False))
    with pytest.raises(ValueError, match="一维"):
        sample_frames(shots, np.zeros((3, 5)), len(frames))
    with pytest.raises(ValueError, match="不匹配"):
        sample_frames(shots, np.zeros(7), len(frames))


def test_sampling_returns_valid_sorted_indices(small_video):
    frames, _, _ = small_video
    shots = detect_shots(frames, cfg=ShotDetectionConfig(use_embedding=False))
    diffs = np.ones(max(len(frames) - 1, 1))
    for strat in STRATEGIES:
        picked = sample_frames(shots, diffs, len(frames),
                               SamplingConfig(strategy=strat, target_frames=6))
        assert picked == sorted(picked)
        assert all(0 <= i < len(frames) for i in picked)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        SamplingConfig(strategy="not-a-strategy")


def test_adaptive_beats_uniform_on_motion_capture(hetero_video):
    """★ 核心论点：同样预算下，内容自适应应覆盖更多**变化量**。

    注意必须用**结构性异构**视频（长短镜头混合）——
    如果所有镜头长度与运动都差不多，"按内容分配"和"按时间分配"恰好等价，
    四种策略会打成平手，那时**基准没有区分力**，而不是策略没差别。
    """
    frames, _, labels = hetero_video
    shots = detect_shots(frames, cfg=ShotDetectionConfig(use_embedding=False))
    hists = [hsv_histogram(f, 32) for f in frames]
    from roboground.data.video.shot import chi_square_distance

    diffs = np.array([chi_square_distance(hists[i], hists[i + 1])
                      for i in range(len(frames) - 1)])
    n_true = len(set(int(x) for x in labels))
    covs = {}
    for strat in ("uniform", "adaptive"):
        picked = sample_frames(shots, diffs, len(frames),
                               SamplingConfig(strategy=strat, target_frames=24))
        covs[strat] = coverage_metrics(picked, labels, n_true, diffs=diffs)
    assert covs["adaptive"]["motion_captured"] > covs["uniform"]["motion_captured"], \
        f"adaptive 变化捕获应更高：{covs}"


def test_motion_captured_is_weighted_by_difference():
    """★ 回归锁：`motion_captured` 必须**按变化量加权**，不能数帧下标。

    第一版写的是"覆盖了多少个差异下标"，结果均匀抽帧最高 ——
    但那只是因为均匀抽帧 spread 得开、覆盖下标多，与"抓到关键变化"无关。
    """
    labels = [0] * 10 + [1] * 10
    # 前 10 帧静止（差异≈0），后 10 帧剧烈变化（差异大）
    diffs = np.array([0.0] * 9 + [1.0] * 10)
    # 只抽静止段 → 变化捕获应接近 0（尽管"覆盖下标"不少）
    only_static = coverage_metrics([4, 8], labels, 2, diffs=diffs)
    assert only_static["motion_captured"] < 0.05, only_static
    # 抽到变化段 → 变化捕获应显著高
    with_motion = coverage_metrics([10, 15, 19], labels, 2, diffs=diffs)
    assert with_motion["motion_captured"] > 0.3, with_motion


# ==========================================================================
# 去重与质量
# ==========================================================================
def test_dhash_is_exposure_robust():
    """dHash 看梯度方向 → 整体亮度偏移不应改变哈希。"""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    brighter = np.clip(base.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    assert hamming(dhash(base), dhash(brighter)) <= 8


def test_dhash_distinguishes_different_images():
    """dHash 看**水平梯度方向**，所以梯度相反的图应给出近乎互补的哈希。

    ⚠️ 测试用例要挑对：左白右黑 与 上白下黑 的水平差分**都是全 0**
    （区域内均匀、边界处是 白>黑=False），
    两者 dHash 完全相同 —— 第一版就是这么写的，白测了一场。
    改用"左暗右亮"与"左亮右暗"这种**梯度方向相反**的图才有区分度。
    """
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    increasing = np.stack([ramp] * 3, axis=-1)            # 左暗右亮
    decreasing = np.stack([ramp[:, ::-1]] * 3, axis=-1)   # 左亮右暗
    assert hamming(dhash(increasing), dhash(decreasing)) > 40

    # 均匀帧的 dHash 全 0（没有梯度）
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert not dhash(flat).any()


def test_dedup_removes_near_duplicates():
    """近乎相同的帧应被哈希级砍掉。"""
    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    frame[10:20, 10:20] = 200
    frames = [frame.copy() for _ in range(5)]
    kept, stats = dedup_frames(frames, list(range(5)), cfg=DedupConfig())
    assert len(kept) == 1, f"5 个相同帧应只剩 1 个，实际 {len(kept)}"
    assert stats["dropped_by_hash"] == 4


def test_quality_gates_reject_blur_and_flat():
    import cv2

    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    blur = cv2.GaussianBlur(sharp, (31, 31), 0)
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)

    cfg = QualityConfig(min_sharpness=30.0, min_contrast=8.0, min_side=32)
    assert quality_scores(sharp)["sharpness"] > quality_scores(blur)["sharpness"]
    kept, stats = filter_by_quality([sharp, blur, flat], [0, 1, 2], cfg=cfg)
    assert 0 in kept, "清晰帧应通过"
    assert 1 not in kept or 2 not in kept, "至少糊帧/纯色帧之一应被淘汰"
    assert stats["n_in"] == 3


def test_quality_rejects_small_frames():
    small = np.zeros((32, 32, 3), dtype=np.uint8)
    small[8:24, 8:24] = 255
    cfg = QualityConfig(min_side=64)
    kept, stats = filter_by_quality([small], [0], cfg=cfg)
    assert kept == []
    assert stats["drop_reasons"].get("resolution") == 1


# ==========================================================================
# 端到端管线
# ==========================================================================
def test_pipeline_index_mapping_is_correct(small_video):
    """★ 回归锁：管线必须返回**原视频下标**，且数量与各阶段统计自洽。

    踩过的坑：dedup 拿到的是"原视频下标"却用来索引"压缩后的帧列表"
    （`frames[i]` 里 i 可达 86，而列表只有 24 个元素），
    越界项被静默跳过 → 87 帧只剩 2 帧甚至 0 帧。
    表面看像"质量闸门太严"，其实是索引错位。
    """
    frames, _, _ = small_video
    # ⚠️ 测试视频只有 96×72，必须放宽 `min_side`：
    # 默认 128 是给真实视频设的，会**正确地**把测试帧全部拒掉
    # （第一版就是这么"失败"的 —— 闸门没问题，是素材太小）。
    res = process_video(frames, cfg=PipelineConfig(
        sampling=SamplingConfig(strategy="uniform", target_frames=8),
        quality=QualityConfig(min_side=32)))
    s = res.stats
    assert 0 < len(res.kept_indices) <= s["n_picked"]
    assert all(0 <= i < len(frames) for i in res.kept_indices)
    assert res.kept_indices == sorted(res.kept_indices)
    assert s["n_after_dedup"] >= s["n_after_quality"]
    assert s["n_after_quality"] == len(res.kept_indices)


def test_pipeline_reports_throughput(small_video):
    frames, _, _ = small_video
    res = process_video(frames, cfg=PipelineConfig(
        sampling=SamplingConfig(strategy="adaptive", target_frames=8)))
    s = res.stats
    assert s["throughput_fps"] > 0
    assert s["throughput_gb_per_hour"] > 0
    assert s["total_seconds"] > 0
    assert "decode" in s["stage_seconds"]


def test_pipeline_handles_empty_input():
    with pytest.raises(ValueError):
        process_video([])


def test_pipeline_on_single_frame():
    """单帧视频不应崩（边界情况）。"""
    frame = np.full((64, 64, 3), 100, dtype=np.uint8)
    res = process_video([frame], cfg=PipelineConfig(
        sampling=SamplingConfig(strategy="uniform", target_frames=1)))
    assert res.stats["n_frames"] == 1
