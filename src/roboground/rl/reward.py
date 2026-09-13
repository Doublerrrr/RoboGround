"""规则引擎作为 **verifiable reward**：空间问答的可验证奖励。

这一环为什么是这个项目独有的优势
================================
RLVR（RL with Verifiable Rewards）在数学/代码领域能work，是因为有**确定的答案校验器**
（对答案、跑单测）。多模态空间推理通常**没有**这样的校验器 —— 只能训 reward model，
而 reward model 会带来 reward hacking。

但本项目有一样别人没有的东西：**几何规则引擎能从 3D 地图算出精确答案**。
于是"VLM 输出的空间答案对不对"这件事，可以**被程序精确判定**：

    VLM 输出 "杯子在 (1.74, 0.04, 0.85)"
       ↓
    规则引擎从地图算出真值 (1.70, 0.00, 0.85)
       ↓
    reward = exp(-||Δp|| / τ)          ← 可验证、不可 hack、无需训练 reward model

这就是把"主观生成 RL 的 reward 怎么设计"落到实处的答案：
**能自动验证的任务，就不要训 reward model。**

奖励设计（面试重点）
==================
分三层，各解决一个具体问题：

| 层 | 形式 | 解决什么 |
|---|---|---|
| **格式** | 0/1 门控 | 输出不是可解析的结构 → 后续无从判定，直接 0 |
| **正确性** | **塑形**（连续） | ★ 见下 |
| **长度** | 轻微惩罚 | 防止模型用"啰嗦"刷格式分 |

★ **为什么数值任务必须用塑形奖励而不是二值**
------------------------------------------------
二值奖励（"距离误差 <0.3m 给 1 否则 0"）在 RL 早期几乎恒为 0：
基座模型的坐标误差是 2.76m，离 0.3m 十万八千里，**整组样本奖励全 0
→ 组内优势全 0 → 梯度为 0 → 完全学不动**。
这就是面试里常问的"**稀疏奖励**"问题。

塑形奖励 `exp(-|Δ|/τ)` 则给出**有梯度的信号**：
误差 2.7m 得 0.004、误差 1.3m 得 0.074、误差 0.4m 得 0.45 ——
虽然都不高，但**排序正确**，组内就有相对优势，策略能被推向正确方向。

代价是塑形会引入**偏好偏差**（模型可能学会"输出看起来接近的垃圾"），
所以 τ 要和任务的可接受误差同量级，且必须配格式门控。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("rl.reward")


# =============================================================================
# Oracle：可验证的答案（由规则引擎在 3D 地图上算出）
# =============================================================================
@dataclass
class SpatialOracle:
    """一条样本的**可验证真值**。

    由几何规则引擎从 3D 语义地图算出，是 reward 的判据来源。
    注意它只存**判定所需的量**，不存自然语言答案 ——
    这样 reward 与"模型该怎么措辞"解耦，避免把语言风格也训进去。
    """

    task: str                                   # relation | distance | locate | list_on
    #: relation 任务：陈述三元组 + 该陈述是否为真
    subject: str = ""
    object_: str = ""
    relation: str = ""
    statement_true: bool = True
    #: distance 任务：中心距 / 表面间隙（米）
    distance_m: float = 0.0
    gap_m: float = 0.0
    #: locate 任务：3D 中心坐标
    coords: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    extent: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: list_on 任务：锚点与其上的物体标签
    anchor: str = ""
    on_labels: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# 奖励配置
# =============================================================================
@dataclass
class RewardConfig:
    """奖励超参。每个都有"为什么是这个值"的理由。"""

    #: 距离塑形尺度（米）。取 0.5 m：与"可接受误差 0.3 m"同量级，
    #: 太小会让有效梯度区间过窄（误差 1m 就得 0.13，信号太弱），
    #: 太大则区分度不足（误差 0.1m 和 0.5m 都接近满分）。
    tau_distance: float = 0.5
    #: 坐标塑形尺度（米）。坐标误差天然比距离误差大（实测 1.35m vs 0.42m），
    #: 所以 τ 也相应放大到 1.0 m，否则全组奖励挤在 0 附近。
    tau_coord: float = 1.0
    #: 格式门控：不通过则总奖励置 0（不给任何"乱写也有分"的空间）
    format_gate: bool = True
    #: 各分量权重
    w_format: float = 0.2
    w_correct: float = 1.0
    w_length: float = 0.05
    #: 长度惩罚：超过这个 token 数开始扣（按超出比例）
    length_soft_limit: int = 48
    #: 语言一致性：要求与问题同语言（中文问题答中文）。
    #: 设 False 可关掉 —— 消融用。
    require_language_match: bool = False


@dataclass
class RewardBreakdown:
    """奖励分解（**必须逐项记录**，否则无法诊断 reward hacking）。"""

    total: float = 0.0
    format_ok: float = 0.0
    correct: float = 0.0
    length: float = 0.0
    parsed: Optional[Dict[str, Any]] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"total": self.total, "format": self.format_ok,
                "correct": self.correct, "length": self.length,
                "reason": self.reason}


# =============================================================================
# 解析：模型输出 → 结构化答案
# =============================================================================
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def parse_answer(text: str, task: str) -> Optional[Dict[str, Any]]:
    """把模型输出解析成结构化答案；解析失败返回 None。

    **优先 JSON，失败再退正则** —— 这个顺序是刻意的：
    - JSON 是训练时要求的输出格式（可被下游程序消费）；
    - 但 RL 早期模型必然不会写 JSON，如果只认 JSON，
      格式门控会把整组样本打成 0，**组内优势全 0 → 学不动**。
      所以留一条正则兜底路径，让"说人话但说对了"也能拿到部分正确性分。
      这是"冷启动"问题的一个具体处置。
    """
    if not text or not text.strip():
        return None
    obj = _try_json(text)
    if obj is not None:
        obj["_via"] = "json"
        return obj
    return _parse_free_text(text, task)


def _try_json(text: str) -> Optional[Dict[str, Any]]:
    """从文本里抽出第一个合法 JSON 对象（容忍前后有多余文字）。"""
    # 去掉常见的 markdown 代码块围栏
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start:i + 1])
                except Exception:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _parse_free_text(text: str, task: str) -> Optional[Dict[str, Any]]:
    """正则兜底：从自然语言里抽答案。"""
    lower = text.lower()
    out: Dict[str, Any] = {"_via": "regex"}

    if task == "relation":
        # 先找否定，再找肯定 —— 顺序不能反：'not to the left' 里也含 'left'
        if re.search(r"\bno\b|不是|不在|并非|not\b", lower):
            out["answer"] = "no"
        elif re.search(r"\byes\b|是的|对|正确", lower):
            out["answer"] = "yes"
        else:
            out["answer"] = None
        return out if out["answer"] else None

    if task in ("distance", "locate"):
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]
        if not nums:
            return None
        if task == "distance":
            out["distance_m"] = nums[0]
        else:
            # 优先找括号里的三元组；找不到就用前三个数
            m = re.search(r"[\(\（]\s*(-?\d+\.?\d*)\s*[,，]\s*(-?\d+\.?\d*)\s*[,，]\s*(-?\d+\.?\d*)\s*[\)\）]", text)
            if m:
                out["coords"] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
            elif len(nums) >= 3:
                out["coords"] = nums[:3]
            else:
                return None
        return out

    if task == "list_on":
        # 数量 + 物体标签
        m = re.search(r"(\d+)\s*(?:个|objects?|items?)", lower)
        out["count"] = int(m.group(1)) if m else len(
            re.findall(r"\b(table|chair|cup|box|bottle|sofa|shelf|monitor|lamp)\b", lower))
        out["labels"] = re.findall(
            r"\b(table|chair|cup|box|bottle|sofa|shelf|monitor|lamp|trash can)\b", lower)
        return out

    return None


# =============================================================================
# 正确性打分（按任务分派）
# =============================================================================
def correctness(parsed: Dict[str, Any], oracle: SpatialOracle,
                cfg: RewardConfig) -> Tuple[float, str]:
    """返回 (分数 ∈ [0,1], 原因)。"""
    task = oracle.task
    if task == "relation":
        ans = str(parsed.get("answer", "")).lower()
        if ans not in ("yes", "no"):
            return 0.0, "relation:答案缺失"
        got = (ans == "yes")
        return (1.0 if got == oracle.statement_true else 0.0), ""

    if task == "distance":
        v = parsed.get("distance_m")
        if v is None or not np.isfinite(v):
            return 0.0, "distance:数值缺失"
        err = abs(float(v) - oracle.distance_m)
        return float(math.exp(-err / cfg.tau_distance)), ""

    if task == "locate":
        c = parsed.get("coords")
        if not c or len(c) < 3:
            return 0.0, "locate:坐标缺失"
        err = float(np.linalg.norm(np.asarray(c[:3], dtype=float)
                                   - np.asarray(oracle.coords, dtype=float)))
        return float(math.exp(-err / cfg.tau_coord)), ""

    if task == "list_on":
        got_labels = {str(x).lower() for x in (parsed.get("labels") or [])}
        want = {str(x).lower() for x in oracle.on_labels}
        if not want:
            return 0.0, "list_on:真值为空"
        if not got_labels:
            # 至少尝试用数量
            c = parsed.get("count")
            if c is not None:
                return (1.0 if int(c) == len(want) else 0.0), "仅数量匹配"
            return 0.0, "list_on:未解析出物体"
        tp = len(got_labels & want)
        prec = tp / max(len(got_labels), 1)
        rec = tp / max(len(want), 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        # 数量正确给一点额外分（防止"全列一遍"刷 F1）
        bonus = 0.2 if int(parsed.get("count", -1)) == len(want) else 0.0
        return float(min(1.0, f1 + bonus)), ""

    return 0.0, f"未知任务 {task}"


def spatial_reward(text: str, oracle: SpatialOracle,
                   cfg: Optional[RewardConfig] = None) -> RewardBreakdown:
    """计算一条样本的奖励（含分解）。

    ⚠️ **返回值必须带分解**。只看 total 无法诊断 reward hacking ——
    比如"格式分涨、正确性分不动"意味着模型学会了写 JSON 但没学会答对，
    这时该做的是提高正确性权重或降低格式权重，而不是继续训。
    """
    cfg = cfg or RewardConfig()
    bd = RewardBreakdown()
    n_tok = len(str(text).split())

    parsed = parse_answer(text, oracle.task)
    bd.parsed = parsed
    bd.format_ok = 1.0 if parsed is not None else 0.0

    if parsed is None:
        bd.reason = "格式不可解析"
        bd.total = 0.0 if cfg.format_gate else cfg.w_length * 0.0
        return bd

    bd.correct, why = correctness(parsed, oracle, cfg)
    bd.reason = why

    # 长度惩罚：只轻微惩罚超长，避免鼓励"越短越好"导致答案残缺
    over = max(0, n_tok - cfg.length_soft_limit)
    bd.length = -min(1.0, over / max(cfg.length_soft_limit, 1))

    bd.total = (cfg.w_format * bd.format_ok
                + cfg.w_correct * bd.correct
                + cfg.w_length * bd.length)
    if cfg.format_gate and bd.format_ok < 1.0:
        bd.total = 0.0
    return bd


# =============================================================================
# 奖励统计（诊断 reward hacking 用）
# =============================================================================
def reward_stats(breakdowns: Sequence[RewardBreakdown]) -> Dict[str, float]:
    """一组样本的奖励统计。

    重点看 **格式分与正确性分的比值**：
    - 格式接近 1、正确性接近 0 → 模型只学会了格式，**在刷格式分**；
    - 两者都低 → 还没学会格式，属于正常冷启动；
    - 正确性高但格式低 → 会把正确答案浪费掉，该提高格式权重。
    """
    if not breakdowns:
        return {"n": 0.0}
    tot = np.array([b.total for b in breakdowns], dtype=np.float64)
    fmt = np.array([b.format_ok for b in breakdowns], dtype=np.float64)
    cor = np.array([b.correct for b in breakdowns], dtype=np.float64)
    # ★ **必须把 JSON 路径与正则兜底路径分开统计**。
    # 踩过的坑：训练日志里 format_rate 一直是 1.00，看起来"模型很听话"，
    # 其实基座模型根本没在写 JSON —— 是正则兜底把任何含数字/含 yes-no 的
    # 自由文本都判成了"可解析"。只看 format_rate 会完全误判模型的遵从度。
    via_json = np.array(
        [1.0 if (b.parsed or {}).get("_via") == "json" else 0.0 for b in breakdowns],
        dtype=np.float64)
    return {
        "n": float(len(breakdowns)),
        "reward_mean": float(tot.mean()),
        "reward_std": float(tot.std()),
        "reward_min": float(tot.min()),
        "reward_max": float(tot.max()),
        "format_rate": float(fmt.mean()),
        #: 真正按要求的 JSON 格式输出的比例（模型遵从度的**真实**指标）
        "json_rate": float(via_json.mean()),
        "correct_mean": float(cor.mean()),
        "nonzero_rate": float((tot > 0).mean()),
        #: >0.5 且 correct_mean 低 → 疑似在刷格式分
        "format_correct_ratio": float(fmt.mean() / max(cor.mean(), 1e-6)),
    }
