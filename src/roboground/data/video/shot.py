"""镜头检测（shot boundary detection）。

为什么视频数据管线里**第一步**是切镜头
====================================
因为"抽帧"这件事的语义单位是**镜头**，不是秒。同一个镜头内的帧高度冗余，
跨镜头才是信息跃变。不做镜头切分就直接均匀抽帧，会出现两种典型坏数据：

- **冗余**：一个静止镜头占了 30 秒 → 抽 30 帧几乎一样，训练数据被它刷屏；
- **漏采**：一个 2 秒的短镜头恰好落在采样点之间 → 整个场景被跳过。

所以工业界视频 caption / video-QA 管线的标准顺序是：
**镜头检测 → 分镜头抽帧 → 去重 → 质量过滤 → 打标**。
面试里被问"视频怎么切帧"，第一句就该讲清这个层级关系。

两个判据，为什么不能只用一个
==========================
| 判据 | 敏感于 | 不敏感于 | 失效场景 |
|---|---|---|---|
| **HSV 直方图卡方距离** | 全局颜色/光照突变 | 物体移动 | 同色系换场景（如室内→室内） |
| **帧间嵌入余弦距离** | 语义内容变化 | 相机平移/抖动 | 光照突变但内容不变 |

两者**互补**：直方图抓"画面整体变了"，嵌入抓"画面内容变了"。
本实现取**融合分数** `max(直方图分, 嵌入分归一化)`，任一显著即判为边界 ——
比 AND 更保守（宁可多切一点，也不要把两个镜头混在一起）。

渐变镜头（fade/dissolve）为什么需要额外处理
=========================================
硬切（hard cut）相邻两帧差异巨大，单帧差分就能抓到。
但**渐变**（淡入淡出、叠化）每帧只变一点点，逐帧差分**永远不过阈值** ——
这是纯逐帧法的经典盲区。解法是同时看**累积差分**：
以滑动窗口的和/均值作为第二判据，渐变会在累积量上暴露。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("data.video.shot")


# =============================================================================
# 帧特征
# =============================================================================
def hsv_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    """单帧的 HSV 联合直方图（H 与 S 各 `bins` 档，V 只用来做曝光加权）。

    H/S 联合比单用 H 更能区分"同色系不同材质"；
    不做三维联合是为了保持维度可控（bins² 而不是 bins³）。
    """
    import cv2  # noqa: PLC0415

    img = np.asarray(frame)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = hist.reshape(-1).astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


def chi_square_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """卡方距离 —— 直方图之间的标准度量，对峰值差异比 L2 更敏感。"""
    denom = h1 + h2
    mask = denom > 0
    if not mask.any():
        return 0.0
    diff = h1[mask] - h2[mask]
    return float(0.5 * np.sum(diff * diff / denom[mask]))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


# =============================================================================
# 配置与结果
# =============================================================================
@dataclass
class ShotDetectionConfig:
    """镜头检测配置。

    ⚠️ 阈值**不要写死**。不同视频源（手机拍摄 / 监控 / 影视剪辑）的
    直方图差与嵌入差的分布能差好几倍。本实现默认 `threshold_mode="adaptive"`：
    用全片差异的**中位数 + k×MAD** 定阈值 ——
    MAD（绝对中位差）比标准差抗离群，不会被少数几个切点自己抬高阈值。
    """

    hist_bins: int = 32
    #: 阈值模式：
    #: - `robust`（默认）`max(median + k×1.4826×MAD, p99)`，见 `_robust_threshold`
    #: - `otsu` 大津法（**只适合干净的合成数据**，真实视频会失效，见下）
    #: - `adaptive` median + k×MAD
    #: - `fixed` 用下面的固定阈值
    #:
    #: ⚠️ **默认曾经是 `otsu`，在真实视频上是灾难性的**（2026-09-11 实测）：
    #: Otsu 的前提是"差异分布双峰"，而真实压缩视频的帧差是**重尾单峰**
    #: （噪声主体 + 尾部几个真切点）。Otsu 对单峰分布照样会返回一个
    #: "把质量对半劈开"的阈值，且它自带的"两类样本都不能太少"守卫
    #: **恰好会被单峰分布骗过**（对半劈时两类权重都≈0.5）。
    #: 实测 10 条 MSR-VTT 视频：Otsu 给出 684 个镜头（68/视频），
    #: robust 给出 32 个（3.2/视频）—— 差 21 倍。
    threshold_mode: str = "robust"
    #: `robust` 模式的 MAD 倍率
    robust_k: float = 6.0
    #: `robust` 模式的分位数下限（重尾分布的噪声上界）
    robust_quantile: float = 0.99
    #: 阈值之上**至少**保留多少个样本（见 `_robust_threshold` 的样本量自适应）。
    #:
    #: ⚠️ 这个值直接决定"一个视频里最多能有多少个镜头"。取太小会让阈值贴着
    #: 最大差异（漏检真切点），取太大则放进噪声。实测：300 帧的视频取 4 时，
    #: video7014 的真切点（diff 0.4388）被 0.4391 的阈值挡在门外 —— **差 0.0003**。
    #: 用金种子扫过 4/6/8/12/16，见 docs/大规模多模态数据工程报告.md 第四节。
    robust_n_signal: int = 4
    #: 阈值之上**最多**允许的比例（防止长视频被放进来太多候选）
    robust_max_frac: float = 0.10
    #: 自适应阈值：median + k * MAD
    adaptive_k: float = 6.0
    #: 自适应失效（差异分布过平）时的兜底绝对阈值
    fallback_hist_threshold: float = 0.35
    fallback_emb_threshold: float = 0.20
    #: 渐变镜头：累积窗口长度（帧）
    gradual_window: int = 5
    #: 渐变判据相对硬切的倍率（渐变每帧变化小，阈值应更低）
    gradual_ratio: float = 0.45
    #: 渐变候选的**峰值突出度**倍率（相对平滑信号 `cum` 的邻域中位数）。
    #:
    #: ★ 为什么渐变也要突出度门（2026-09-11 实测）：
    #: `cum` 是连续帧差的滑动平均，而**噪声的滑动平均本身就是缓慢起伏的平台**，
    #: 只要阈值低一点就到处都"超阈"。实测 10 条真实视频里渐变路径贡献了
    #: 102~143 个假边界（其中 video7013 完全没有切点却报了 18 个）。
    #: 真叠化在 `cum` 上是一个**局部峰**（两端平、中间鼓），
    #: 突出度判据正好只留这种形态。
    #:
    #: 倍率比硬切那一路（`min_peak_ratio=5.0`）低，因为平滑会把峰削掉。
    #: 实测 2.0：渐变假边界 102 → 12，而 video7012（有真切点）仍保留 2 个。
    gradual_peak_ratio: float = 2.0
    #: 是否启用渐变的峰值突出度门。**默认必须开**，关掉只是为了让消融实验
    #: 能复现"没有这道门会怎样"（见 docs/大规模多模态数据工程报告.md 的消融表）。
    use_gradual_prominence: bool = True
    #: 最短镜头长度（帧）—— 低于它的"边界"是噪声，直接合并
    min_shot_len: int = 3
    #: 首/尾镜头是否豁免 `min_shot_len`。
    #:
    #: ★ 为什么需要（2026-09-11 金种子核验发现）：
    #: `min_shot_len` 的语义是"一个边界至少要把片子切成这么长的两段才可信"，
    #: 但这个约束对**首镜头**是错的 —— 首镜头 [0, b) 的**左端是视频开头，
    #: 不是另一个切点**，`b - 0 < min_len` 只说明"视频一开始就是一个短镜头"，
    #: 这在真实素材里完全合法（片头一闪而过的字幕卡、广告切播、转场后立即切换）。
    #: 实测 video7014 第 2 帧是一次真硬切（diff 0.4600，`hard=True`），
    #: 却因为首镜头只有 2 帧被丢弃 —— 这是**召回侧的静默漏检**。
    #:
    #: 风险与约束：豁免只在"该边界是**硬切**（`hard[b-1]`）"时生效。
    #: 渐变候选在帧 0 附近**天然不可信** —— 渐变判据用的 `cum` 是
    #: `np.convolve(..., mode="same")`，两端被截断填充，边缘值有系统偏差。
    #: 硬切是单帧判据，不受这个影响。
    #:
    #: 上界影响：每条视频**最多**多出 1 个边界（只有第一个候选能触发），
    #: 且它仍须通过阈值 + 峰值突出度两道门。
    exempt_edge_shots: bool = True
    #: 边界合并窗口（帧）。**默认 1 = 基本关闭**，这是实测扫出来的结论。
    #:
    #: 历史：一开始用它来合并"渐变过渡产生的连续多点候选"（一次 10 帧叠化
    #: 被检出 4 个边界）。但后来加了**峰值突出度判据**（`peak_prominence_mask`），
    #: 它从根上就只留下"单点尖峰"，渐变的多点候选根本不会产生 ——
    #: **两个机制在解同一个问题，于是 NMS 变成纯粹有害**：
    #: 它会把短镜头的两个真边界（相隔 3~4 帧）当渐变尾巴合并掉。
    #:
    #: 实测扫描（异构视频，含 4~5 帧短镜头）：
    #:   nms=1 → 精确率 71.4% / **召回率 100%** / F1 0.833  ← 最好
    #:   nms=3 → 80.0% / 80.0% / 0.800
    #:   nms=6 → **100%** / 60.0% / 0.750
    #: 我们更看召回（漏切会把两个场景混成一段，caption 直接矛盾；
    #: 多切一刀只是多几个碎片），所以取 1。
    #:
    #: 若遇到**完全没有快运动**且渐变很长的视频，可以把它调大试试。
    nms_window: int = 1
    #: 有嵌入就用嵌入（更准）；没有就只用直方图
    use_embedding: bool = True
    # ---- 峰值突出度判据（解决"快运动淹没真边界"）----
    #: 邻域窗口半径（帧）。用来估"局部背景差异水平"。
    peak_window: int = 5
    #: 候选边界的差异必须 ≥ 局部邻域中位数的多少倍。
    #:
    #: ★ 为什么必须有这个判据（实测发现）
    #: 快运动镜头（快速摇镜/手持）的**镜头内**帧间差异能到 0.72，
    #: 而我们合成视频里**真实硬切**只有 0.34 —— 也就是说
    #: **"帧间差异大"这件事本身无法区分"切镜头"和"镜头内快运动"**。
    #:
    #: 但两者的**形态**完全不同：
    #:   真切换 = 单点尖峰（邻域中位 0.0004，比值 ~950）
    #:   快运动 = 持续高值（邻域中位 0.6，比值 ~1.2）
    #: 相差三个数量级，所以用"相对邻域的突出度"一刀就能切开。
    min_peak_ratio: float = 5.0


@dataclass
class Shot:
    """一个镜头（半开区间 [start, end)，单位：帧下标）。"""

    index: int
    start: int
    end: int
    boundary_score: float = 0.0          #: 该镜头**起始处**的边界分数
    is_gradual: bool = False             #: 是否由渐变判据切出来的

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        kind = "渐变" if self.is_gradual else "硬切"
        return (f"Shot#{self.index}[{self.start}:{self.end}] "
                f"len={self.length} {kind} score={self.boundary_score:.3f}")


# =============================================================================
# 主流程
# =============================================================================
def detect_shots(
    frames: Sequence[np.ndarray],
    *,
    embeddings: Optional[np.ndarray] = None,
    cfg: Optional[ShotDetectionConfig] = None,
) -> List[Shot]:
    """检测镜头边界并返回镜头列表。

    Parameters
    ----------
    frames
        帧序列（RGB，uint8）。**逐帧传入而不是视频路径** ——
        这样管线可以是流式的（不需要把整段视频读进内存）。
    embeddings
        可选的逐帧嵌入 `(N, D)`。有它就能用上"语义判据"，
        对"同色系换场景"这类直方图盲区更稳。没有则退化为纯直方图。
    cfg
        检测配置。

    Returns
    -------
    List[Shot]
        至少包含一个镜头（哪怕只有一个边界都没有）。
    """
    cfg = cfg or ShotDetectionConfig()
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        return [Shot(index=0, start=0, end=1)]

    # ---- 1) 逐帧特征 ----
    hists = np.stack([hsv_histogram(f, cfg.hist_bins) for f in frames], axis=0)
    hist_diffs = np.array(
        [chi_square_distance(hists[i], hists[i + 1]) for i in range(n - 1)],
        dtype=np.float64,
    )

    emb_diffs: Optional[np.ndarray] = None
    if cfg.use_embedding and embeddings is not None:
        E = _l2_normalize(np.asarray(embeddings, dtype=np.float64))
        if E.shape[0] == n:
            # 余弦距离 = 1 - 余弦相似度，落在 [0, 2]，正常情况接近 [0, 1]
            emb_diffs = 1.0 - np.sum(E[:-1] * E[1:], axis=1)
        else:
            logger.warn(f"嵌入帧数 {E.shape[0]} 与帧数 {n} 不一致，忽略嵌入判据")

    # ---- 2) 阈值 ----
    t_hist = _resolve_threshold(hist_diffs, cfg)
    t_emb = (_resolve_threshold(emb_diffs, cfg) if emb_diffs is not None else None)

    # ---- 3) 逐帧判定（硬切 + 渐变）----
    # 判据统一在 `decide_boundaries` 里，**不要在这里重写一遍** ——
    # 这个函数与 `pipeline._detect_from_hists` 共用同一份实现。
    hard, gradual = decide_boundaries(hist_diffs, emb_diffs, cfg)

    boundaries = [0] + [i + 1 for i in range(n - 1) if hard[i] or gradual[i]]
    boundaries.append(n)
    # ---- 4) 边界合并（非极大值抑制）----
    # ⚠️ 一个**渐变**过渡会在差异曲线上产生**一串连续超阈值的点**
    # （实测一次 10 帧叠化被检出 4 个边界：42/45/48/51），
    # 不做合并的话精确率会被自己打垮。
    boundaries = _merge_boundaries(boundaries, hist_diffs, cfg.nms_window)
    # ---- 5) 合并过短镜头 ----
    shots = _merge_short(boundaries, hard, gradual, cfg.min_shot_len, hist_diffs,
                         exempt_edge=cfg.exempt_edge_shots)
    logger.debug(f"镜头检测：{n} 帧 → {len(shots)} 个镜头 "
                 f"(t_hist={t_hist:.4f}"
                 + (f", t_emb={t_emb:.4f}" if t_emb is not None else "") + ")")
    return shots


def _adaptive_threshold(diffs: np.ndarray, k: float, fallback: float) -> float:
    """`median + k × MAD` 自适应阈值。

    ⚠️ **这个判据在"同镜头内差异极小"的视频上会失效**（实测踩过）：
    合成视频里同镜头中位差异只有 0.001、MAD 更小，算出来阈值 0.0066，
    而**非边界**差异的 p95 是 0.0213 —— 阈值低于噪声上界，
    于是产生大量误报（精确率掉到 50%）。
    根因是 MAD 衡量的是"分布的离散度"，但这里真正需要的是
    "**双峰之间的谷底**" —— 所以默认改用 `_otsu_threshold`。
    本函数保留用于分布接近单峰的场合。
    """
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 4:
        return fallback
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if mad < 1e-9:
        # 差异分布几乎为常数（比如全静止视频）→ 自适应失效
        return max(fallback, med * 2.0 if med > 0 else fallback)
    thr = med + k * 1.4826 * mad          # 1.4826 把 MAD 换算成 σ 当量
    # 自适应结果太小时兜个底，避免噪声全被当边界
    return float(max(thr, med * 1.5))


def _robust_threshold(diffs: np.ndarray, k: float, quantile: float,
                      fallback: float, *, n_signal: int = 4,
                      max_frac: float = 0.10) -> float:
    """重尾分布的稳健阈值：`max(median + k×1.4826×MAD, p_quantile)`。

    ★ 为什么不用 Otsu（默认曾是它，2026-09-11 在真实视频上实测推翻）
    --------------------------------------------------------------
    Otsu 的前提是"差异分布**双峰**"—— 同镜头内接近 0、切点处 0.1~0.7，
    要的是两峰之间的谷底。这个前提在**合成视频**上成立，在**真实压缩视频**上
    **不成立**：实测 MSR-VTT 的帧差分布是**重尾单峰** ——

    ===========  ========  ========  ========  ========
    视频          中位      MAD       p99       max
    ===========  ========  ========  ========  ========
    video7010    0.0004    0.0003    0.1142    0.2054
    video7013    0.0099    0.0038    0.0369    0.0497
    video7017    0.0290    0.0116    0.0860    0.2044
    ===========  ========  ========  ========  ========

    中位到 p99 跨了两个数量级，**没有第二个峰**。Otsu 对单峰分布照样会
    返回一个"把质量对半劈开"的阈值（video7010 算出 0.0000、video7013 算出
    0.0104 ≈ 中位数本身），于是 216/269 个点被判成边界。
    更糟的是 Otsu 自带的守卫 `min(w0, w1) < 0.05` **恰好会被单峰分布骗过**：
    对半劈时两类权重都 ≈ 0.5，守卫直接放行。

    实测 10 条 MSR-VTT 视频的镜头数：Otsu **684**（68/视频）→ robust **32**（3.2/视频）。

    为什么是 `max(median + k×MAD, p_quantile)`
    ------------------------------------------
    - `median + k×1.4826×MAD`：MAD 换算成 σ 当量的稳健上界，
      对"同镜头内差异极小"的视频也稳（不像 σ 会被少数切点自己抬高）；
    - 但 MAD 只描述**主体**的离散度，**低估重尾**（上表里 median+6σ 只有
      0.0033~0.132，而 p99 是 0.037~0.724）—— 所以再取 `p_quantile`
      作为"噪声主体的上界"。两者取大，保证阈值**高于噪声、低于真信号**。
    - 如果结果高于全片最大差异，说明整段视频**没有任何突出的跳变**，
      即单镜头视频 —— 这是合法结论，`_detect_from_hists` 会自然给出 0 个边界。
    """
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 4:
        return fallback
    n = d.size
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    thr_sigma = med + k * 1.4826 * mad if mad > 1e-12 else med * 2.0
    # ⚠️ 分位数必须**随样本量自适应**，否则小样本会退化：
    # `np.quantile(d, 0.99)` 在 n=31 时就是**最大值本身**
    # （第 30.7 个次序统计量），阈值高到把真切点一起挡掉 ——
    # 实测合成短序列召回率从 0.80 掉到 0.33。
    # 改成"阈值之上至少留 `n_signal` 个样本，且不超过 `max_frac`"。
    frac = min(max_frac, max(1.0 - quantile, n_signal / float(n)))
    thr_q = float(np.quantile(d, 1.0 - frac))
    return float(max(thr_sigma, thr_q, 1e-9))


def _otsu_threshold(diffs: np.ndarray, n_bins: int = 128) -> float:
    """Otsu 大津阈值 —— 把"切点 / 非切点"当成一次二分类来定阈值。

    为什么它比 `median + k×MAD` 合适：镜头差异分布**天然是双峰的**
    （同镜头内接近 0，切点处 0.1~0.7），要的是**两峰之间的谷底**。
    Otsu 正是"最大化类间方差"的那个分割点，不需要调任何超参。

    ⚠️ 两个实现要点：
    1. **在对数尺度上做** —— 差异跨 3 个数量级（1e-4 ~ 1e0），
       线性分桶会让 90% 的样本挤在第一个桶里，Otsu 直接失效；
    2. 分出结果要**夹到观测范围内**，并且要求两类的样本量都不能太少，
       否则退化回 `median + k×MAD`。
    """
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 8:
        return float(np.max(d)) if d.size else 0.0
    eps = 1e-6
    logs = np.log10(np.clip(d, eps, None))
    lo, hi = float(logs.min()), float(logs.max())
    if hi - lo < 1e-6:
        return float(d.max())
    hist, edges = np.histogram(logs, bins=n_bins, range=(lo, hi))
    total = hist.sum()
    if total == 0:
        return float(d.max())
    p = hist.astype(np.float64) / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    idx = int(np.argmax(sigma_b))
    t = float(10 ** centers[idx])
    # ---- 有效性守卫（两道，缺一不可）----
    # ① 两类都要有足够样本，否则分布不是双峰、Otsu 结果不可信。
    #    ⚠️ **这一道挡不住单峰分布** —— 单峰被对半劈时 w0≈w1≈0.5，照样通过。
    w0, w1 = omega[idx], 1.0 - omega[idx]
    if min(w0, w1) < 0.05:
        return _robust_threshold(d, 6.0, 0.99, float(d.max()))
    # ② 阈值必须**高于噪声地板**。真实压缩视频的差异分布是重尾单峰，
    #    Otsu 会把阈值切在噪声主体内部（实测 video7010 算出 0.0000、
    #    video7013 算出 0.0104 —— 后者≈中位数本身），
    #    于是 216/269 个点被判成边界。落在噪声里 → 退回 robust。
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if t < med + 3.0 * 1.4826 * mad:
        return _robust_threshold(d, 6.0, 0.99, float(d.max()))
    return t


def peak_prominence_mask(diffs: np.ndarray, window: int,
                         min_ratio: float) -> np.ndarray:
    """峰值突出度掩码：只保留"相对局部背景显著凸起"的位置。

    ★ 这是本模块最重要的一个判据，动机是实测踩到的坑：

    快运动镜头（快速摇镜、手持）的**镜头内**帧间差异可以达到 0.72，
    而同一段视频里**真实硬切**只有 0.34 —— 即"差异大"本身
    完全无法区分"切镜头"和"镜头内快运动"。实测后果：
    一个 28 帧的快运动镜头内部产生 8 个以上超阈候选，
    经合并后把整段 [56,66] 并成一个假边界，
    同时把真边界 34/39 也并掉，**召回率从 100% 掉到 60%**。

    两者的区别不在**幅度**而在**形态**：

    | | 邻域中位 | 峰值 | 比值 |
    |---|---|---|---|
    | 真切换 | 0.0004 | 0.382 | ~950 |
    | 快运动 | 0.60 | 0.718 | ~1.2 |

    所以判据是 `d[i] >= min_ratio × median(邻域)` ——
    用**相对**突出度而不是绝对幅度，三个数量级的差距一刀切开。

    ⚠️ 邻域中位数要**排除自身**，否则峰值会把自己抬高。
    """
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    w = max(1, int(window))
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        neigh = np.concatenate([d[lo:i], d[i + 1:hi]])
        if neigh.size == 0:
            out[i] = True
            continue
        base = float(np.median(neigh))
        if base <= 0:
            out[i] = d[i] > 0
        else:
            out[i] = d[i] >= min_ratio * base
    return out


def _resolve_threshold(diffs: np.ndarray, cfg: ShotDetectionConfig,
                       is_hist: bool = True) -> float:
    """按 `cfg.threshold_mode` 选阈值。"""
    mode = (cfg.threshold_mode or "robust").lower()
    fallback = (cfg.fallback_hist_threshold if is_hist
                else cfg.fallback_emb_threshold)
    if mode == "fixed":
        return float(fallback)
    if mode == "otsu":
        return _otsu_threshold(diffs)
    if mode == "adaptive":
        return _adaptive_threshold(diffs, cfg.adaptive_k, fallback)
    return _robust_threshold(
        diffs, cfg.robust_k, cfg.robust_quantile, fallback,
        n_signal=cfg.robust_n_signal, max_frac=cfg.robust_max_frac,
    )


def decide_boundaries(
    hist_diffs: np.ndarray,
    emb_diffs: Optional[np.ndarray] = None,
    cfg: Optional[ShotDetectionConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """从帧间差异序列判定 `(hard, gradual)` 两个布尔掩码。

    ★ 抽成公共函数的原因（2026-09-11）：这段判定原先在 `detect_shots`
    （传帧的入口）和 `pipeline._detect_from_hists`（传直方图的入口）
    里**各写了一份**，两份逻辑几乎一样但不完全一样 —— 修一边漏一边
    就是必然的事。真实视频上的镜头检测 bug 正是在两个入口上都存在，
    只因为默认阈值在合成数据上"看起来能用"而长期没被发现。
    **同一份判据只能有一处实现。**

    Returns
    -------
    (hard, gradual)
        长度均为 `len(hist_diffs)` 的布尔数组。`hard[i]` 表示"帧 i 与 i+1
        之间存在硬切"，`gradual[i]` 表示"存在渐变过渡（叠化/划像）"。
    """
    cfg = cfg or ShotDetectionConfig()
    hd = np.asarray(hist_diffs, dtype=np.float64)
    n = hd.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

    t_hist = _resolve_threshold(hd, cfg, is_hist=True)
    t_emb = None
    if emb_diffs is not None:
        ed = np.asarray(emb_diffs, dtype=np.float64)
        if ed.size == n:
            t_emb = _resolve_threshold(ed, cfg, is_hist=False)

    # 融合：直方图与嵌入**任一**超阈值即认为是候选边界（宁多切不少切）
    cand = hd >= t_hist
    if t_emb is not None:
        cand = cand | (np.asarray(emb_diffs, dtype=np.float64) >= t_emb)

    # ★ 峰值突出度过滤：把"镜头内持续快运动"和"真正的单点切换"分开
    hard = cand & peak_prominence_mask(hd, cfg.peak_window, cfg.min_peak_ratio)

    # 渐变：滑动窗口内的累积差异。硬切会在一帧内爆发，渐变则是平摊的。
    gradual = np.zeros(n, dtype=bool)
    w = max(2, int(cfg.gradual_window))
    if n >= w:
        cum = np.convolve(hd, np.ones(w) / w, mode="same")
        m = (cum >= t_hist * cfg.gradual_ratio) & ~hard
        # ★ 渐变也必须"相对邻域突出"：真叠化在平滑信号 `cum` 上是局部峰，
        #   而噪声的平滑信号是缓慢起伏的平台，突出度≈1。
        #   没有这一道时，10 条真实视频的渐变路径贡献了 102~143 个假边界。
        if cfg.use_gradual_prominence:
            m = m & peak_prominence_mask(cum, cfg.peak_window,
                                         cfg.gradual_peak_ratio)
        gradual = _refine_gradual(hd, m, w)
    return hard, gradual


def _merge_boundaries(boundaries: List[int], diffs: np.ndarray,
                      window: int) -> List[int]:
    """把窗口内相邻的候选边界合并成一个（非极大值抑制）。

    为什么必须做：**一个渐变过渡会连续多帧超阈值**。
    实测一次 10 帧交叉叠化被检出 4 个边界（42/45/48/51），
    真值只有 1 个 —— 精确率直接被自己的渐变判据打垮。

    合并位置用**差异加权质心**而不是"取最大值"：
    叠化的真值边界是过渡**中点**（两场景各占一半），
    而最大值常偏在某一侧（两场景颜色分布不对称时尤其明显）。
    加权质心对对称过渡能回到中点，对硬切则退化为那个单点。

    ⚠️ 这也是为什么 `nms_window` 要和渐变长度匹配：
    窗口太小合不掉（叠化尾巴漏出去），太大又会把两个真边界并成一个。
    """
    if len(boundaries) <= 2 or window <= 1:
        return sorted(set(boundaries))
    edges = [b for b in sorted(set(boundaries)) if 0 < b < boundaries[-1]]
    out: List[int] = []
    group: List[int] = [edges[0]]
    for b in edges[1:]:
        if b - group[-1] <= window:
            group.append(b)
        else:
            out.append(_group_center(group, diffs))
            group = [b]
    out.append(_group_center(group, diffs))
    return sorted(set([0] + out + [boundaries[-1]]))


def _group_center(group: List[int], diffs: np.ndarray) -> int:
    """一组候选边界的加权质心（权重 = 该边界处的差异值）。"""
    if len(group) == 1:
        return group[0]
    weights = []
    for b in group:
        idx = b - 1                       # diffs[i] 是帧 i 与 i+1 之间
        weights.append(float(diffs[idx]) if 0 <= idx < len(diffs) else 0.0)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        return int(round(float(np.mean(group))))
    center = int(round(float(np.sum(np.asarray(group, dtype=np.float64) * w) / w.sum())))
    return min(max(center, min(group)), max(group))


def _refine_gradual(diffs: np.ndarray, mask: np.ndarray, w: int) -> np.ndarray:
    """把渐变窗口内的边界对齐到"窗口内差异最大"的那一帧。"""
    out = np.zeros_like(mask)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    half = w // 2
    for i in idx:
        lo, hi = max(0, i - half), min(len(diffs), i + half + 1)
        out[lo + int(np.argmax(diffs[lo:hi]))] = True
    return out


def _merge_short(boundaries: List[int], hard: np.ndarray, gradual: np.ndarray,
                 min_len: int, diffs: np.ndarray, *,
                 exempt_edge: bool = False) -> List[Shot]:
    """把长度 < `min_len` 的镜头并进前一个镜头（消除噪声边界）。

    Parameters
    ----------
    exempt_edge
        首镜头是否豁免 `min_len`（见 `ShotDetectionConfig.exempt_edge_shots`）。
        只在"该边界是硬切"时豁免 —— 渐变判据在帧 0 附近因卷积边缘效应不可信。
    """
    last = max(boundaries)
    cuts = [b for b in boundaries if 0 < b < last]
    kept: List[int] = []
    for b in cuts:
        prev = kept[-1] if kept else 0
        if b - prev >= min_len:
            kept.append(b)
        elif (exempt_edge and not kept
              and 0 < b <= len(hard) and bool(hard[b - 1])):
            # 首镜头 [0, b)：左端是视频开头而非切点，`b < min_len` 只说明
            # "片子一开始就是短镜头"，不构成噪声证据。硬切单帧判据可信，
            # 放行；渐变候选不放行（`cum` 在两端有卷积边缘偏差）。
            kept.append(b)
    edges = [0] + kept + [last]

    shots: List[Shot] = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            continue
        # 边界分数取该镜头起始处的差异值（第 0 个镜头无边界 → 0）
        score = float(diffs[s - 1]) if 0 < s <= len(diffs) else 0.0
        is_grad = bool(gradual[s - 1]) if 0 < s <= len(gradual) else False
        shots.append(Shot(index=len(shots), start=s, end=e,
                          boundary_score=score, is_gradual=is_grad))
    return shots


# =============================================================================
# 评测（受控基准：镜头边界已知）
# =============================================================================
def evaluate_shot_detection(
    predicted: Sequence[Shot],
    truth_boundaries: Sequence[int],
    *,
    tolerance: int = 2,
) -> Dict[str, Any]:
    """用已知的镜头边界算 P/R/F1。

    为什么要做这个评测：绝大多数人只会"看效果还行"，
    但我们能构造**边界已知**的合成视频，于是可以把镜头检测
    当成一个有真值的检测任务来量化 —— 阈值调参也就有了依据。

    Parameters
    ----------
    tolerance
        允许的边界偏移（帧）。渐变镜头的真实边界本身有主观性，
        偏移 1~2 帧不应算错。
    """
    pred = sorted({s.start for s in predicted if s.start > 0})
    truth = sorted(int(t) for t in truth_boundaries if t > 0)

    matched_pred, matched_truth = set(), set()
    for pi, p in enumerate(pred):
        for ti, t in enumerate(truth):
            if ti in matched_truth:
                continue
            if abs(p - t) <= tolerance:
                matched_pred.add(pi)
                matched_truth.add(ti)
                break

    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_pred": len(pred), "n_truth": len(truth),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def make_synthetic_shot_video(
    *,
    n_shots: int = 6,
    frames_per_shot: Any = 12,
    width: int = 320,
    height: int = 240,
    num_objects: int = 6,
    seed: int = 2026,
    gradual_at: Sequence[int] = (),
    blend_frames: int = 10,
    stride_per_shot: Any = 0.06,
    heterogeneous: bool = False,
) -> tuple:
    """构造**镜头边界已知**的合成视频，用于量化镜头检测与抽帧策略。

    做法：每个镜头 = 一个独立合成场景的连续相机运动（小步长 → 画面连续），
    镜头之间**硬切**（场景完全不同 → 直方图与嵌入都跳变）。
    `gradual_at` 指定的镜头交界处改用**渐变过渡**（交叉叠化），
    用来验证渐变判据是否生效 —— 这是纯逐帧差分法的经典盲区。

    Parameters
    ----------
    frames_per_shot, stride_per_shot
        可以是 int（所有镜头相同），也可以是**逐镜头的列表**。
    heterogeneous
        ★ 是否用**结构性异构**的预设（长短镜头混合 + 运动量差异大）。

    ⚠️ 为什么需要 `heterogeneous`
    ---------------------------
    如果所有镜头长度和运动量都差不多，**四种抽帧策略会打成平手** ——
    因为"按内容分配预算"和"按时间分配预算"在这种视频上恰好等价。
    实测第一版就是这样：四种策略镜头覆盖全是 100%、变化捕获 35%~52%，
    看起来"没区分度"，其实是**基准设计得没有区分力**，
    而不是策略没差别。

    真实的视频恰恰是高度异构的（长静止镜头 + 短高动态镜头混在一起），
    这才是"抽帧策略决定数据质量"的成立条件。
    所以这个开关不是为了让结论好看，而是为了让基准**具有区分力**。
    """
    from roboground.data.synthetic import make_synthetic_sequence  # noqa: PLC0415

    n = int(n_shots)
    if heterogeneous:
        # 镜像真实视频的结构：1 个很长的静止镜头（浪费预算的元凶）
        # + 几个短的高动态镜头（容易被均匀抽帧漏掉）
        lengths = [34, 5, 18, 4, 26, 7][:n]
        # ⚠️ stride 的量级必须**锚定到真实帧率下的位移**，不能随手放大：
        # 第一版用了 0.30~0.34 m/帧 —— 那相当于房间尺寸的 7.5%/帧，
        # 而真实 30fps 下步行 1 m/s 只有 ~0.8%/帧，等于把难度设成了现实的 10 倍，
        # 于是"快运动淹没真边界"这个结论被夸大了（虽然现象本身是真的）。
        # 这里取 0.02~0.12 m/帧（房间尺度 0.5%~3%/帧），覆盖"静止→快走"的真实区间。
        strides = [0.010, 0.10, 0.03, 0.12, 0.02, 0.08][:n]
        while len(lengths) < n:
            lengths.append(12)
            strides.append(0.06)
    else:
        lengths = ([int(frames_per_shot)] * n
                   if isinstance(frames_per_shot, (int, float))
                   else [int(x) for x in frames_per_shot])
        strides = ([float(stride_per_shot)] * n
                   if isinstance(stride_per_shot, (int, float))
                   else [float(x) for x in stride_per_shot])
        lengths = (lengths + [12] * n)[:n]
        strides = (strides + [0.06] * n)[:n]

    gradual_set = set(int(g) for g in gradual_at)
    frames: List[np.ndarray] = []
    labels: List[int] = []
    truth: List[int] = []

    prev_tail: Optional[np.ndarray] = None
    for s in range(n):
        # stride 取小值 → 同一镜头内是平滑的相机移动，不会自己触发边界
        seq = make_synthetic_sequence(
            seed=seed + s, num_frames=lengths[s],
            width=width, height=height, num_objects=num_objects,
            stride=strides[s],
        )
        shot_frames = [np.asarray(f.color, dtype=np.uint8) for f in seq]

        if s == 0:
            truth_local = 0
        else:
            truth_local = len(frames)
            if s in gradual_set and prev_tail is not None:
                # 交叉叠化：把过渡帧**替换**进序列。
                # 末帧恰为 B（见 `_crossfade`），所以后面从 shot_frames[1:] 接上。
                #
                # ⚠️ `blend_frames` 必须够长（默认 10 帧）才是**真正的渐变**：
                # 第一版用了 4 帧，alpha 每步跳 0.25 —— 每帧直方图差异仍高达 ~0.3，
                # 逐帧差分**照样能抓到**，根本没构成"渐变盲区"，
                # 于是测的不是渐变检测能力。
                n_blend = max(2, int(blend_frames))
                alphas = [(i + 1) / n_blend for i in range(n_blend)]
                blend = _crossfade(prev_tail, shot_frames[0], n_blend)
                # 真值边界 = 叠化中"两场景各占一半"的那一帧（alpha 最接近 0.5）
                mid = int(np.argmin([abs(a - 0.5) for a in alphas]))
                truth_local = len(frames) + mid
                frames.extend(blend)
                labels.extend([s - 1] * n_blend)
                shot_frames = shot_frames[1:]
            truth.append(truth_local)

        for f in shot_frames:
            frames.append(f)
            labels.append(s)
        prev_tail = shot_frames[-1]

    return frames, truth, labels


def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> List[np.ndarray]:
    """生成从 `a` 到 `b` 的 n 帧线性交叉叠化，**末帧恰为 `b`**。

    ⚠️ alpha 必须取到 **1.0**。第一版写成 `alpha = i/(n+1)`，
    最大只有 n/(n+1)=0.8 —— 于是叠化末帧还差 20% 才到 B，
    下一帧却是完整的 B，**凭空造出一个假硬切**（实测该处直方图差异 0.70，
    比所有真边界都大），把镜头检测的精确率打到 50%。
    这个 bug 在"只测镜头检测"时看不出来，是靠**有真值**才暴露的。
    """
    A = np.asarray(a, dtype=np.float32)
    B = np.asarray(b, dtype=np.float32)
    if A.shape != B.shape:
        import cv2  # noqa: PLC0415

        B = cv2.resize(B, (A.shape[1], A.shape[0])).astype(np.float32)
    out = []
    for i in range(1, n + 1):
        alpha = i / n                      # 1/n, 2/n, ..., 1.0（末帧 == B）
        out.append(np.clip((1 - alpha) * A + alpha * B, 0, 255).astype(np.uint8))
    return out
