"""抽帧策略：把「镜头」变成「训练用的帧」。

为什么抽帧策略是视频数据管线的胜负手
==================================
同一个视频，抽帧策略不同 → 训练数据的信息量能差几倍：

- **均匀抽帧（uniform）**：实现最简单，但**按时间而非内容**分配预算。
  一个 30 秒的静止镜头会拿走大部分帧（且几乎重复），
  而一个信息密度高但只有 2 秒的镜头几乎抽不到。
- **分镜头三等分（thirds）**：每个镜头固定取首/中/尾。
  均匀覆盖了每个镜头，但**不管镜头长短** —— 短镜头里三帧高度冗余，
  长镜头里三帧又严重欠采样。
- **关键帧（keyframe）**：取镜头内帧间变化最大处。
  信息量大，但**只覆盖变化点**，会漏掉稳定状态（而稳定状态往往才是 caption 的主体）。
- **内容自适应（adaptive）**：按镜头时长 × 内容变化率**分配预算**，
  再在镜头内按变化率**加权采样**。本质是"把抽帧预算当成资源来分配"。

本模块**四种都实现**，并用同一个下游指标去比 —— 这才是"数据决策要有依据"，
而不是凭感觉选一个。

面试可讲的一句话
==============
"抽帧不是采样问题，是**预算分配问题**：总帧数固定时，
应该按**内容变化率**而不是**时间长度**来分配。"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from roboground.data.video.shot import Shot
from roboground.utils.logging import get_logger

logger = get_logger("data.video.sampling")

#: 支持的策略名
STRATEGIES = ("uniform", "thirds", "keyframe", "adaptive")


@dataclass
class SamplingConfig:
    """抽帧配置。"""

    strategy: str = "adaptive"
    #: 目标总帧数（**预算是固定的** —— 这样四种策略才可比）
    target_frames: int = 24
    #: 每个镜头至少抽几帧（保证短镜头不被完全跳过）
    min_per_shot: int = 1
    #: adaptive 策略里"变化率"与"时长"的权重（和为 1）
    weight_motion: float = 0.6
    weight_duration: float = 0.4

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"未知抽帧策略 {self.strategy!r}，可选：{STRATEGIES}")


def _frame_change_rates(diffs: np.ndarray, start: int, end: int) -> np.ndarray:
    """镜头内逐帧变化率（用于关键帧与自适应加权）。

    `diffs[i]` 是帧 i 与 i+1 之间的差异，所以镜头 [start,end) 内部
    只有 `end-start-1` 个差异值。这里用 **0 填充**到与帧数对齐：
    首帧没有"进入它"的差异，给 0 是合理的（它不是被变化选中的）。
    """
    n = end - start
    out = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return out
    seg = np.asarray(diffs[start:end - 1], dtype=np.float64)
    out[1:1 + len(seg)] = seg
    return out


def allocate_quota(lengths: np.ndarray, budget: int,
                   weights: Optional[np.ndarray] = None,
                   min_per_shot: int = 1) -> np.ndarray:
    """把总预算分配到各镜头，返回**保证 sum == min(budget, 总帧数)** 的配额。

    ★ 这是四种策略共用的配额分配器，存在的理由是一个反复踩到的坑：
    **配额被镜头长度截断后必须再分配**，否则实际抽帧数低于预算。
    踩过三次（adaptive 24→15、24→17，keyframe 12→10）——
    每次的表现都是"这个策略看起来省帧"，其实是预算被静默扣掉了。
    而"固定预算"正是四种策略可比的前提，所以统一在这里兜住。

    Parameters
    ----------
    lengths
        各镜头长度。
    budget
        目标总帧数。
    weights
        可选的分配权重（越长/越动的镜头拿越多）。
        `None` 表示按长度等比例分配（thirds / keyframe 用）。
    min_per_shot
        每个镜头至少几帧（保证短镜头不被完全跳过）。

    Returns
    -------
    np.ndarray
        每个镜头的配额，`0 <= quota <= length`，且 `quota.sum() == min(budget, lengths.sum())`。

    Notes
    -----
    用**迭代再分配**而不是一次算完：一次分配后总有镜头因 `quota > length`
    被截断，截断释放出的份额必须回到池子里继续分。
    最多迭代 8 轮（每轮至少填满一个镜头，实际 2~3 轮就收敛）。
    """
    lens = np.asarray(lengths, dtype=np.int64)
    n = lens.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    target = int(min(budget, int(lens.sum())))
    if target <= 0:
        return np.zeros(n, dtype=np.int64)

    quota = np.minimum(np.full(n, max(0, int(min_per_shot)), dtype=np.int64), lens)
    if quota.sum() > target:                     # min_per_shot 本身就超预算
        quota = np.zeros(n, dtype=np.int64)
    w = (np.asarray(weights, dtype=np.float64) if weights is not None
         else lens.astype(np.float64))
    w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, None)

    for _ in range(8):
        remaining = target - int(quota.sum())
        if remaining <= 0:
            break
        room = np.clip(lens - quota, 0, None)
        if room.sum() <= 0:
            break
        ww = np.where(room > 0, w, 0.0)
        if ww.sum() <= 0:
            ww = room.astype(np.float64)
        raw = ww / ww.sum() * remaining
        add = np.minimum(np.floor(raw).astype(np.int64), room)
        rest = remaining - int(add.sum())
        if rest > 0:
            frac = raw - np.floor(raw)
            for i in np.argsort(-frac):
                if rest <= 0:
                    break
                if room[i] - add[i] > 0:
                    add[i] += 1
                    rest -= 1
        if add.sum() <= 0:
            break
        quota = quota + add
    return quota


def _uniform(frames_idx: np.ndarray, k: int) -> List[int]:
    """整段视频均匀取 k 帧（忽略镜头）——基线。"""
    n = len(frames_idx)
    if k >= n:
        return frames_idx.tolist()
    pos = np.linspace(0, n - 1, k)
    return sorted({int(frames_idx[int(round(p))]) for p in pos})


def _thirds(shots: Sequence[Shot], k: int, min_per_shot: int = 1) -> List[int]:
    """每个镜头取三等分位置（首/中/尾），按镜头长度比例分配预算。"""
    if not shots:
        return []
    lens = np.array([s.length for s in shots], dtype=np.int64)
    quota = allocate_quota(lens, k, weights=None, min_per_shot=min_per_shot)
    picked: List[int] = []
    for s, q in zip(shots, quota):
        if q <= 0:
            continue
        if q >= s.length:
            picked.extend(range(s.start, s.end))
            continue
        # 三等分：把镜头长度 q 等分，取每段中点（比取端点更抗过渡帧污染）
        pos = np.linspace(0, s.length - 1, q + 2)[1:-1]
        picked.extend(s.start + int(round(p)) for p in pos)
    return sorted(set(picked))


def _keyframe(shots: Sequence[Shot], diffs: np.ndarray, k: int,
              min_per_shot: int = 1) -> List[int]:
    """每个镜头取"帧间变化最大"的位置（内容关键帧）。"""
    if not shots:
        return []
    lens = np.array([s.length for s in shots], dtype=np.int64)
    quota = allocate_quota(lens, k, weights=None, min_per_shot=min_per_shot)
    picked: List[int] = []
    for s, q in zip(shots, quota):
        if q <= 0:
            continue
        if q >= s.length:
            picked.extend(range(s.start, s.end))
            continue
        cr = _frame_change_rates(diffs, s.start, s.end)
        # 取变化率最高的 q 个位置（用 argsort 保证确定性）
        idx = np.argsort(-cr, kind="stable")[:q]
        picked.extend(s.start + int(i) for i in sorted(idx))
    return sorted(set(picked))


def _adaptive(shots: Sequence[Shot], diffs: np.ndarray, k: int,
              cfg: SamplingConfig) -> List[int]:
    """内容自适应：预算按 `时长^w1 × 变化率^w2` 分配，镜头内按变化率加权。

    这是四种策略里唯一"知道自己总预算"的：
    1. 先给每个镜头分配**配额**（信息量大的镜头多拿）；
    2. 再在镜头内按变化率**加权采样**（选变化率高的位置）。

    注意配额用**乘性**组合而不是加权和：一个 0.1 秒的镜头无论变化多大
    都不该拿 10 帧（采不到那么多），而一个 30 秒全静止的镜头也不该拿 10 帧
    （拿到的几乎全是重复）。乘法天然压制这种"单项极端"。
    """
    n = len(shots)
    if n == 0:
        return []
    lengths = np.array([s.length for s in shots], dtype=np.float64)
    motion = np.array(
        [float(_frame_change_rates(diffs, s.start, s.end).mean()) for s in shots],
        dtype=np.float64,
    )
    # 归一化到 [0,1] 后做乘性打分，避免量纲差异
    ln = lengths / max(lengths.max(), 1e-9)
    mn = motion / max(motion.max(), 1e-9)
    score = np.power(np.clip(ln, 1e-6, None), cfg.weight_duration) * \
        np.power(np.clip(mn, 1e-6, None), cfg.weight_motion)
    score = np.clip(score, 1e-9, None)

    # 先给每个镜头分配**配额**（信息量大的镜头多拿），再在镜头内按变化率加权采样
    lengths_int = lengths.astype(int)
    quota = allocate_quota(lengths_int, k, weights=score,
                           min_per_shot=cfg.min_per_shot)

    picked: List[int] = []
    for s, q in zip(shots, quota):
        if q <= 0:
            continue
        if q >= s.length:
            picked.extend(range(s.start, s.end))
            continue
        cr = _frame_change_rates(diffs, s.start, s.end)
        picked.extend(s.start + i for i in _weighted_stratified(cr, int(q)))
    return sorted(set(picked))


def _weighted_stratified(weights: np.ndarray, q: int) -> List[int]:
    """按权重**分层**选出 q 个互不相同的下标。

    做法：把累积权重轴等分成 q 段，每段内取权重最大的那个位置。

    为什么不用"按累积分布取分位点"（`searchsorted(cdf, (i+0.5)/q)`）：
    那个方法在权重分布**尖锐**时会把多个分位点映射到**同一个下标**，
    去重后实际帧数少于预算 —— 而"固定预算"正是四种策略可比的前提。
    实测就踩到过：adaptive 预算 24 帧，实际只选出 15 帧。

    ⚠️ 分层法本身也有两个坑，都踩过：
    1. **权重全零会让 cdf 变平** —— 静止镜头内帧间变化率全是 0，
       于是 `searchsorted` 对所有分段都返回 0，每段都选同一个位置。
       解法是给权重加一个与量级相关的小 epsilon，保证 cdf 严格递增。
    2. **仍需 used 去重兜底** —— 极端分布下分段仍可能塌缩，所以显式记录已选，
       重复时改选段内最近的未用位置。
    """
    n = len(weights)
    q = max(1, min(int(q), n))
    if q >= n:
        return list(range(n))
    w = np.asarray(weights, dtype=np.float64).copy()
    if not np.isfinite(w).all():
        w = np.nan_to_num(w, nan=0.0)
    w = np.clip(w, 0.0, None)
    scale = float(w.max()) if w.size else 0.0
    # ★ 关键：加 epsilon 让 cdf 严格递增，否则全零权重会让分段塌缩
    w = w + (scale * 1e-3 if scale > 0 else 1e-3)
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]

    used: set = set()
    out: List[int] = []
    for k in range(q):
        lo, hi = k / q, (k + 1) / q
        i0 = int(np.searchsorted(cdf, lo, side="left"))
        i1 = int(np.searchsorted(cdf, hi, side="right"))
        i0 = min(max(i0, 0), n - 1)
        i1 = max(i1, i0 + 1)
        seg = [i for i in range(i0, min(i1, n)) if i not in used]
        if not seg:
            # 该段已全被占用 → 退而求其次：选全局最近的未用位置
            free = [i for i in range(n) if i not in used]
            if not free:
                break
            seg = [min(free, key=lambda i: abs(i - i0))]
        pick = max(seg, key=lambda i: w[i])
        used.add(pick)
        out.append(pick)
    return sorted(out)


def sample_frames(
    shots: Sequence[Shot],
    diffs: np.ndarray,
    total_frames: int,
    cfg: Optional[SamplingConfig] = None,
) -> List[int]:
    """按指定策略从镜头列表里挑出帧下标。

    Parameters
    ----------
    shots
        `detect_shots` 的输出。
    diffs
        逐帧差异数组（长度 `total_frames-1`），四种策略都要用。
    total_frames
        视频总帧数（策略需要知道全片长度）。
    cfg
        抽帧配置（含策略名与预算）。

    Returns
    -------
    List[int]
        选中的帧下标（升序、去重）。
    """
    cfg = cfg or SamplingConfig()
    k = max(1, int(cfg.target_frames))
    all_idx = np.arange(total_frames, dtype=np.int64)

    # 形状校验：`diffs` 必须是长度 total_frames-1 的一维数组。
    # 不加这层守卫的话，传错形状会在 `_frame_change_rates` 里
    # 抛出难以理解的广播错误（实测踩到过：
    # "could not broadcast input array from shape (2,96) into shape (2,)"）。
    arr = np.asarray(diffs)
    if arr.ndim != 1:
        raise ValueError(
            f"diffs 必须是一维数组（长度 total_frames-1={total_frames - 1}），"
            f"实际形状 {arr.shape}。"
            "常见原因：对各帧先做了 .mean(-1) 再 np.diff，得到的是二维。"
        )
    if total_frames > 1 and arr.shape[0] != total_frames - 1:
        raise ValueError(
            f"diffs 长度 {arr.shape[0]} 与帧数 {total_frames} 不匹配（应为 {total_frames - 1}）"
        )

    if cfg.strategy == "uniform":
        picked = _uniform(all_idx, k)
    elif cfg.strategy == "thirds":
        picked = _thirds(shots, k, cfg.min_per_shot)
    elif cfg.strategy == "keyframe":
        picked = _keyframe(shots, diffs, k, cfg.min_per_shot)
    elif cfg.strategy == "adaptive":
        picked = _adaptive(shots, diffs, k, cfg)
    else:  # pragma: no cover - 构造时已校验
        raise ValueError(cfg.strategy)

    # ★ 统一兜底：任何策略都不允许"少抽"。
    #   各策略内部已各自修正过一轮（重复下标、配额截断、取整碰撞），
    #   但历史证明这里总会漏一种（实测 thirds 因 np.linspace 取整碰撞少 1 帧）。
    #   所以在出口处统一补齐：按**帧间变化率**从高到低补，保证补进来的帧信息量最大。
    picked = _top_up(picked, k, total_frames, arr if arr.ndim == 1 else None)

    logger.debug(f"抽帧[{cfg.strategy}]：{total_frames} 帧 → 选中 {len(picked)} 帧")
    return picked


def _top_up(picked: List[int], k: int, total_frames: int,
            diffs: Optional[np.ndarray]) -> List[int]:
    """把 `picked` 补足到 k 个（若可能），优先补变化率高的帧。

    这是"固定预算"这一对比前提的最后一道保险。
    """
    out = sorted(set(int(p) for p in picked if 0 <= p < total_frames))
    if len(out) >= k or total_frames <= len(out):
        return out[:k] if len(out) > k else out
    used = set(out)
    cand = [i for i in range(total_frames) if i not in used]
    if diffs is not None and len(diffs) > 0:
        # 帧 i 的变化率取 max(diffs[i-1], diffs[i])（帧处在两个差分的交界）
        def motion(i: int) -> float:
            vals = []
            if 0 <= i - 1 < len(diffs):
                vals.append(float(diffs[i - 1]))
            if 0 <= i < len(diffs):
                vals.append(float(diffs[i]))
            return max(vals) if vals else 0.0
        cand.sort(key=lambda i: -motion(i))
    need = k - len(out)
    out.extend(cand[:need])
    return sorted(out)


# =============================================================================
# 数据侧指标：不训练也能看出策略好坏
# =============================================================================
def coverage_metrics(
    picked: Sequence[int],
    labels: Sequence[int],
    n_shots: int,
    *,
    diffs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """衡量一批抽帧的"信息效率"——**不训练**就能比出策略差异。

    四个指标，各有各的失败模式：

    | 指标 | 含义 | 低值意味着 |
    |---|---|---|
    | `shot_coverage` | 被抽到帧的镜头占比 | **漏采**：有镜头整段没进训练数据 |
    | `shot_balance` | 各镜头配额与"信息量"的匹配度（1-基尼系数） | **偏采**：预算被少数镜头吃掉 |
    | `redundancy` | 抽中帧之间的平均相似度 | **冗余**：抽了一堆几乎一样的帧 |
    | `motion_captured` | 抽中帧**按变化量加权**覆盖的总变化占比 | **漏掉变化**：只抽到了静止画面 |

    ⚠️ `motion_captured` 必须**按变化量加权**，不能数帧下标。
    第一版写的是 `covered.sum() / d.size`（覆盖了多少个下标），
    实测结果是"均匀抽帧最高（45%）"—— 但那只是因为均匀抽帧spread 得开、
    覆盖的**下标**多，跟"有没有抓到关键变化"无关。
    加权之后才真正回答"这批帧承载了全片多少信息量"。
    """
    out: Dict[str, float] = {}
    picked = sorted(set(int(p) for p in picked))
    if not picked:
        return {k: 0.0 for k in
                ("shot_coverage", "shot_balance", "redundancy", "motion_captured")}

    labs = np.asarray(labels)
    seen = {int(labs[i]) for i in picked if 0 <= i < len(labs)}
    out["shot_coverage"] = len(seen) / max(n_shots, 1)

    # 每个镜头实际拿到的帧数
    counts = np.array([sum(1 for i in picked if 0 <= i < len(labs)
                           and int(labs[i]) == s) for s in range(n_shots)],
                      dtype=np.float64)
    out["shot_balance"] = 1.0 - _gini(counts)

    if diffs is not None and len(diffs) > 1:
        d = np.asarray(diffs, dtype=np.float64)
        # 冗余：抽中帧相邻间隔内的差异越小，说明这批帧越像
        if len(picked) >= 2:
            gaps = [float(np.mean(d[picked[i]:max(picked[i] + 1, picked[i + 1])]))
                    for i in range(len(picked) - 1)]
            gaps = [g for g in gaps if np.isfinite(g)]
            mean_gap = float(np.mean(gaps)) if gaps else 0.0
            # 用全片平均差异做归一化 → 得到"相对信息密度"
            base = float(np.mean(d)) if d.size else 0.0
            out["redundancy"] = 1.0 - min(1.0, mean_gap / base) if base > 0 else 0.0
        else:
            out["redundancy"] = 1.0
        # 变化量覆盖：**按差异大小加权**，回答"承载了全片多少信息量"
        covered = np.zeros_like(d, dtype=bool)
        for i in picked:
            lo = max(0, i - 1)
            hi = min(len(d), i + 1)
            covered[lo:hi] = True
        total_motion = float(d.sum())
        out["motion_captured"] = (float(d[covered].sum()) / total_motion
                                  if total_motion > 0 else 0.0)
        # 附带：覆盖的**下标**占比（"时间轴铺开程度"），跟上面不是一回事
        out["timeline_coverage"] = float(covered.sum()) / max(d.size, 1)
    else:
        out["redundancy"] = 0.0
        out["motion_captured"] = 0.0
        out["timeline_coverage"] = 0.0
    return out


def _gini(x: np.ndarray) -> float:
    """基尼系数（0 = 完全平均，→1 = 极度集中）。"""
    v = np.sort(np.asarray(x, dtype=np.float64))
    n = v.size
    if n == 0 or v.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) / (n * v.sum())) - (n + 1) / n)
