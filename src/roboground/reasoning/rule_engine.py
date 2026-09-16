"""规则引擎：把自然语言问题解析成"可确定性回答的结构化意图"。

为什么需要一个"规则"后端（而不是直接上 VLM）
-------------------------------------------
1. **离线可用**：没有任何模型权重也能回答"杯子在哪/离桌子多远/桌子上有什么"，
   这让整个系统在 8GB 显存、无网络的条件下依然是个能跑的产品；
2. **可解释、可验证**：输出带有精确的米制数字与几何证据，便于单元测试
   和与 VLM 结果对比（VLM 会编数字，规则引擎不会）；
3. **作为 VLM 的兜底与工具**：真实系统里常见做法是"VLM 负责理解意图 +
   规则引擎负责精确计算"。本项目两条路都实现，并可融合。

意图类型
--------
| kind | 例子 | 输出 |
|---|---|---|
| `locate`   | "杯子在哪" / "where is the cup" | 目标 3D 位置 |
| `distance` | "杯子离桌子多远" | 米制距离（中心距 + 表面间隙） |
| `relation` | "杯子在桌子上面吗" | 空间关系 + 证据 |
| `list_on`  | "桌子上有什么" | 该区域内的物体列表 |
| `nearest`  | "离我最近的门" | 最近实例 + 距离 |
| `count`    | "有几个椅子" | 计数 |
| `describe` | "描述一下场景" | 地图摘要 |

中文分词问题
------------
中文没有空格，直接正则切词很容易出错。这里采用**词表召回**策略：
不切词，而是拿地图里已知的标签 + 双语别名表去原文里做**子串召回**，
召回结果按长度优先（避免"桌子"被"子"抢先匹配）。这样无需分词器，
且与 `mapping.query` 的词法匹配共用同一套知识。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.reasoning.spatial_relations import (
    RELATION_NAMES,
    bbox_gap,
    center_distance,
    compute_all_relations,
    compute_relation,
)
from roboground.types import ReasoningResult, SpatialRelation
from roboground.utils.logging import get_logger

logger = get_logger("reasoning.rules")


# ==========================================================================
# 关键词表
# ==========================================================================
_LOCATE_KW = ("在哪", "在哪里", "位置", "什么地方", "哪儿", "哪里",
              "where is", "where's", "where are", "locate", "find")
_DISTANCE_KW = ("多远", "距离", "有多远", "离得远", "how far", "distance")
_LIST_KW = ("有什么", "有哪些", "有什么东西", "what is on", "what's on",
            "what is in", "list", "列一下")
_NEAREST_KW = ("最近", "离我最近", "最近的", "closest", "nearest", "nearest to me")
_COUNT_KW = ("几个", "有多少", "多少个", "数量", "how many", "count")
_DESCRIBE_KW = ("描述", "介绍一下", "总结", "场景", "describe", "summarize", "what do you see")

#: 关系关键词 → 关系名（用于"X 在 Y 上面吗"这类判断）
_RELATION_TERMS: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = [
    (("上面", "上方", "上边", "之上", "on top", "above", "over"), ("above",)),
    (("下面", "下方", "下边", "之下", "under", "below", "beneath"), ("below",)),
    (("左边", "左侧", "左面", "left of", "to the left"), ("left_of",)),
    (("右边", "右侧", "右面", "right of", "to the right"), ("right_of",)),
    (("前面", "前方", "前边", "in front", "ahead"), ("in_front_of",)),
    (("后面", "后方", "后边", "behind", "back of"), ("behind",)),
    (("里面", "内部", "之中", "inside", "within", "in the"), ("inside",)),
    (("旁边", "附近", "边上", "旁边", "next to", "beside", "near"), ("near",)),
]

#: 中文疑问/助词，召回对象时先剔除，避免干扰
_STOPWORDS = (
    "的", "在", "是", "吗", "呢", "哪", "个", "有", "和", "与", "离", "到",
    "上", "下", "里", "中", "边", "面", "请", "帮我", "我", "它", "他", "她",
    "你", "现在", "位置", "多远", "距离", "多少", "几", "什么", "东西",
)


@dataclass
class Intent:
    """解析出的意图。"""

    kind: str
    raw: str
    subject: Optional[str] = None          # 主语（要找的东西）
    object_: Optional[str] = None          # 参照物（"在 X 的上面" 里的 X）
    relation: Optional[str] = None         # 询问/断言的关系
    mentions: List[str] = field(default_factory=list)
    matched_labels: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "object": self.object_,
            "relation": self.relation,
            "mentions": list(self.mentions),
            "matched_labels": list(self.matched_labels),
        }


# ==========================================================================
# 召回路数
# ==========================================================================
def _build_vocabulary(labels: Sequence[str]) -> List[str]:
    """构造召回词表。

    **重要**：词表是"全量别名表 ∪ 地图标签"，而不仅仅是地图里有的类别。
    原因是：用户可能问一个地图里根本没有的物体（"杯子在哪"，但地图里只有椅子），
    这时我们要能识别出"他在问杯子"，从而回答"没找到杯子" ——
    而不是误判成"描述场景"。
    """
    from roboground.mapping.query import ALIAS_LEXICON  # noqa: PLC0415

    vocab: set = set()
    for lab in labels:
        if lab:
            vocab.add(str(lab).lower())
    for concept, aliases in ALIAS_LEXICON.items():
        vocab.add(concept.lower())
        vocab.update(str(a).lower() for a in aliases)
    return sorted(vocab, key=len, reverse=True)


def _recall_mentions(text: str, labels: Sequence[str]) -> List[str]:
    """从文本里召回对象提及（不切词，直接子串召回，按长度优先）。"""
    lowered = text.lower()
    vocab = _build_vocabulary(labels)
    found: List[Tuple[int, str]] = []
    used_spans: List[Tuple[int, int]] = []

    for term in vocab:
        if not term:
            continue
        start = 0
        while True:
            pos = lowered.find(term, start)
            if pos < 0:
                break
            span = (pos, pos + len(term))
            # 与已占用的区间重叠则跳过（长词优先，短词不重复占位）
            if not any(span[0] < s[1] and s[0] < span[1] for s in used_spans):
                used_spans.append(span)
                found.append((span[0], text[span[0]:span[1]]))
            start = pos + len(term)

    # 按在原文中出现的顺序返回，并去重（保序）
    found.sort(key=lambda kv: kv[0])
    out: List[str] = []
    for _, term in found:
        if term not in out:
            out.append(term)
    return out


def _resolve_labels(mentions: Sequence[str], labels: Sequence[str]) -> List[str]:
    """把提及映射到地图里真实存在的标签（保留歧义）。

    ⚠️ 一个提及可能同时命中多个标签，例如 "桌子" 既是 table 也是 desk。
    这里**不做硬选择**，而是把全部达标标签都返回，顺序为：
      1. 按提及在原文中出现的顺序；
      2. 同一提及内按匹配分数降序；
      3. 分数相同时按地图给出的标签顺序（`SemanticMap.labels` 是按频次排序的，
         所以更常见的类别排在前面）。

    保留歧义的好处：上层可以只说"最可能的那一个"，也可以在需要时
    同时考虑多个候选（例如"离我最近的桌子"应该 desk/table 一起比）。
    """
    from roboground.mapping.query import LexicalMatcher  # noqa: PLC0415

    matcher = LexicalMatcher()
    known = list(dict.fromkeys(str(x) for x in labels))   # 保序去重
    if not known:
        return []

    out: List[str] = []
    for mention in mentions:
        scores = matcher.score(mention, labels=known)
        # stable 排序：分数相同时保持 known 的原始顺序作为 tie-break
        order = np.argsort(-scores, kind="stable")
        for idx in order:
            score = float(scores[int(idx)])
            if score <= 0.5:
                break
            label = known[int(idx)]
            if label not in out:
                out.append(label)
    return out


# ==========================================================================
# 意图解析
# ==========================================================================
def parse_intent(text: str, labels: Sequence[str] = ()) -> Intent:
    """解析自然语言问题为 `Intent`。"""
    raw = str(text or "").strip()
    lowered = raw.lower()
    intent = Intent(kind="describe", raw=raw)

    mentions = _recall_mentions(raw, labels)
    intent.mentions = mentions
    intent.matched_labels = _resolve_labels(mentions, labels)

    # ---- 关系词 ----
    detected_relation: Optional[str] = None
    for terms, (rel,) in _RELATION_TERMS:
        if any(t in lowered for t in terms):
            detected_relation = rel
            break
    intent.relation = detected_relation

    has = lambda kws: any(k in lowered for k in kws)  # noqa: E731

    # 注意：判定"用户问了几个物体"用的是 `mentions`（含地图里没有的），
    # 而"能不能回答"用的是 `matched_labels`（地图里真实存在的）。
    # 两者分开是必要的：否则"杯子在哪"（地图里没有杯子）会被误判成"描述场景"。
    n_mentions = len(intent.mentions)
    n_matched = len(intent.matched_labels)

    # ---- "问了一个地图里根本没有的词"必须明确说没找到（批评 #1 里发现的问题）----
    #
    # ★ 实测（scripts/43）：`airplane 在哪` / `submarine 在哪` 因为
    #   一个已知标签都没提到，会**退化成"描述场景"**，回答
    #   "场景中共有 28 个物体：clutter×10、wall×5…"。
    #   从用户视角看，这像是"回答了但答错了" —— 比直接说"没找到"更糟。
    #   所以这里：**只要带疑问词（在哪/位置/…）却一个提及都没认出来**，
    #   就按"定位"处理并把没认出来的那段文本当作主语，
    #   由 `_answer_locate` 走 `_not_found`。
    unknown_subject: Optional[str] = None
    if n_mentions == 0 and has(_LOCATE_KW):
        unknown_subject = _unknown_subject_text(raw, labels)

    # ---- 优先级：距离 > 列表 > 计数 > 最近 > 关系 > 定位 > 描述 ----
    if has(_DISTANCE_KW) and n_mentions >= 2:
        intent.kind = "distance"
    elif has(_LIST_KW) and n_mentions >= 1:
        intent.kind = "list_on"
    elif has(_COUNT_KW) and n_mentions >= 1:
        intent.kind = "count"
    elif has(_NEAREST_KW):
        intent.kind = "nearest"
    elif detected_relation is not None and n_mentions >= 1:
        intent.kind = "relation"
    elif n_mentions >= 1:
        intent.kind = "locate"
    elif unknown_subject:
        # 有疑问词、但一个已知词都没提到（"airplane 在哪"）——
        # 仍然按"定位"处理，好让回答是"没找到 airplane"
        # 而不是退化成"场景中共有 28 个物体…"
        intent.kind = "locate"
    else:
        intent.kind = "describe"

    if intent.matched_labels:
        intent.subject = intent.matched_labels[0]
        if len(intent.matched_labels) >= 2:
            intent.object_ = intent.matched_labels[1]
    elif intent.mentions:
        # 提到了但地图里没有 —— 仍然把它当作主语，好让回答能说清"没找到什么"
        intent.subject = intent.mentions[0]
        if len(intent.mentions) >= 2:
            intent.object_ = intent.mentions[1]
    elif unknown_subject:
        # 一个已知词都没提到，但有疑问词 —— 用没认出来的那段文本当主语
        intent.subject = unknown_subject

    return intent


def _unknown_subject_text(text: str, labels: Sequence[str]) -> Optional[str]:
    """从"问了个不存在的物体"的句子里抠出那个词（用于回答"没找到 X"）。

    做法很朴素：去掉停用词、疑问词和地图里已知的词，剩下的第一段
    非空文本就是用户问的东西。抠不出来就返回 None（上层用整句兜底）。
    """
    leftover = str(text or "")
    for term in sorted(list(_STOPWORDS) + list(_LOCATE_KW) + list(_DISTANCE_KW)
                       + list(_NEAREST_KW) + list(_COUNT_KW) + list(_LIST_KW)
                       + list(_DESCRIBE_KW),
                       key=len, reverse=True):
        if term:
            leftover = leftover.replace(term, " ")
    for lab in sorted((str(x) for x in labels), key=len, reverse=True):
        if lab:
            leftover = leftover.replace(lab, " ")
    leftover = leftover.strip(" \t\r\n，。？！,.?!：:；;、\"'（）()「」【】")
    return leftover.split()[0] if leftover.split() else None


# ==========================================================================
# 规则引擎
# ==========================================================================
class RuleEngine:
    """基于几何与词法匹配的确定性问答引擎。

    Examples
    --------
    >>> from roboground.mapping import SemanticMap          # doctest: +SKIP
    >>> engine = RuleEngine(semantic_map)                   # doctest: +SKIP
    >>> res = engine.answer("杯子在哪")                      # doctest: +SKIP
    >>> res.targets[0].label                                # doctest: +SKIP
    'cup'
    """

    name = "rules"

    def __init__(
        self,
        semantic_map,
        *,
        cfg: Any = None,
        thresholds: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
        robot_position: Optional[Sequence[float]] = None,
    ) -> None:
        self.map = semantic_map
        self.cfg = cfg
        if thresholds is None and cfg is not None:
            thresholds = dict(cfg.get("reasoning.relations", {}) or {})
        self.thresholds = thresholds or {}
        self.top_k = int(top_k or (cfg.get("query.top_k", 5) if cfg is not None else 5))

        if robot_position is not None:
            self.robot_position = np.asarray(robot_position, dtype=np.float64).reshape(3)
        else:
            rp = (getattr(semantic_map, "meta", {}) or {}).get("robot_position")
            self.robot_position = (
                np.asarray(rp, dtype=np.float64).reshape(3)
                if rp is not None else np.zeros(3, dtype=np.float64)
            )

    # ---------------- 入口 ----------------
    def answer(self, query: str) -> ReasoningResult:
        """回答问题，返回结构化结果。"""
        labels = self.map.labels if hasattr(self.map, "labels") else []
        intent = parse_intent(query, labels)

        handlers = {
            "locate": self._answer_locate,
            "distance": self._answer_distance,
            "relation": self._answer_relation,
            "list_on": self._answer_list_on,
            "nearest": self._answer_nearest,
            "count": self._answer_count,
            "describe": self._answer_describe,
        }
        handler = handlers.get(intent.kind, self._answer_describe)
        try:
            result = handler(intent)
        except Exception as exc:  # 永不因为一个解析失败而崩掉整条链
            logger.warn(f"规则引擎处理 {intent.kind!r} 失败：{exc}")
            result = ReasoningResult(
                query=query, answer="抱歉，我没能理解这个问题。",
                backend=self.name, confidence=0.0, debug=intent.to_dict(),
            )
        result.debug.setdefault("intent", intent.to_dict())
        result.backend = self.name
        return result

    # ---------------- 各类意图 ----------------
    def _not_found(self, what: str, query: str) -> ReasoningResult:
        """统一的"地图里没有这个物体"回答。"""
        available = "、".join(self.map.labels[:10]) or "（空）"
        return ReasoningResult(
            query=query,
            answer=f"当前地图里没有「{what}」。地图里现有的类别有：{available}。",
            backend=self.name,
            confidence=0.0,
            debug={"reason": "object_not_in_map", "queried": what},
        )

    def _lookup(self, label: str, top_k: Optional[int] = None) -> List[Any]:
        """按标签检索物体（先精确匹配，再用词法/嵌入召回的模糊匹配）。

        ⚠️ 模糊召回路径会加一道**分数下限**（0.3）：否则一个毫不相关的物体
        也可能以 0.06 的低分被返回，导致"没找到"变成"找到了别的"。
        """
        k = int(top_k or self.top_k)
        exact = [o for o in self.map.objects if str(o.label).lower() == str(label).lower()]
        if exact:
            exact.sort(key=lambda o: -o.confidence)
            return exact[:k]

        # ⚠️ 模糊兜底的分数下限**必须跟词法阈值一致**。
        #    这里曾经硬编码 0.30，而 `SemanticMap.query_min_score` 与配置里
        #    的词法阈值是 0.5 —— 于是规则引擎比地图查询**更松**，
        #    未见类别更容易被 bigram 噪声蒙中（"问什么都能返回一个物体"）。
        floor = 0.30
        if self.cfg is not None:
            floor = float(self.cfg.get("query.min_score_lexical",
                                       self.cfg.get("query.min_score", floor)))
        hits = self.map.query_text(label, top_k=k, level="object", min_score=floor)
        return [h.obj for h in hits if h.obj is not None]

    def _answer_locate(self, intent: Intent) -> ReasoningResult:
        subject = intent.subject or (intent.matched_labels[0] if intent.matched_labels else None)
        if not subject:
            return self._answer_describe(intent)

        objs = self._lookup(subject)
        if not objs:
            return self._not_found(subject, intent.raw)
        top = objs[0]
        parts = [f"找到 {len(objs)} 个「{subject}」，最可信的一个在"]
        parts.append(
            f"({top.center[0]:+.2f}, {top.center[1]:+.2f}, {top.center[2]:+.2f}) m，"
            f"尺寸约 {top.extent[0]:.2f}×{top.extent[1]:.2f}×{top.extent[2]:.2f} m"
        )
        dist_to_robot = float(np.linalg.norm(top.center - self.robot_position))
        parts.append(f"，距机器人 {dist_to_robot:.2f} m。")

        return ReasoningResult(
            query=intent.raw,
            answer="".join(parts),
            targets=objs,
            distances={"robot_to_target": dist_to_robot},
            confidence=float(top.confidence),
        )

    def _answer_distance(self, intent: Intent) -> ReasoningResult:
        if len(intent.matched_labels) < 2:
            # 区分两种情况：用户没说清 / 用户说了但地图里没有
            if len(intent.mentions) >= 2:
                missing = [m for m in intent.mentions
                           if not any(m == lab for lab in intent.matched_labels)]
                if missing:
                    return self._not_found(missing[0], intent.raw)
            return ReasoningResult(
                query=intent.raw,
                answer="请告诉我两个物体，例如「杯子离桌子多远」。",
                backend=self.name, confidence=0.0,
            )
        a_label, b_label = intent.matched_labels[0], intent.matched_labels[1]
        a_list, b_list = self._lookup(a_label, top_k=1), self._lookup(b_label, top_k=1)
        if not a_list or not b_list:
            missing = a_label if not a_list else b_label
            return self._not_found(missing, intent.raw)

        a, b = a_list[0], b_list[0]
        d_center = center_distance(a, b)
        gap = bbox_gap(a, b)
        rel = compute_relation(a, b, thresholds=self.thresholds)

        answer = (
            f"「{a_label}」的中心距「{b_label}」{d_center:.2f} m"
            f"（最近表面间隙 {gap:.2f} m）；"
            f"从方位上看，{rel.subject} 在 {rel.object} 的"
            f"{RELATION_NAMES.get(rel.relation, rel.relation)}。"
        )
        return ReasoningResult(
            query=intent.raw,
            answer=answer,
            targets=[a, b],
            relations=[rel],
            distances={
                f"{a_label}<->{b_label}_center": d_center,
                f"{a_label}<->{b_label}_gap": gap,
            },
            confidence=float(min(a.confidence, b.confidence)),
        )

    def _answer_relation(self, intent: Intent) -> ReasoningResult:
        if len(intent.matched_labels) < 2:
            return self._answer_locate(intent)

        a_label, b_label = intent.matched_labels[0], intent.matched_labels[1]
        a_list, b_list = self._lookup(a_label, top_k=1), self._lookup(b_label, top_k=1)
        if not a_list or not b_list:
            return ReasoningResult(
                query=intent.raw, answer="没找到问题里提到的物体。",
                backend=self.name, confidence=0.0,
            )

        a, b = a_list[0], b_list[0]
        rel = compute_relation(a, b, thresholds=self.thresholds)
        asked = intent.relation
        rel_name = RELATION_NAMES.get(rel.relation, rel.relation)

        if asked is None:
            answer = (
                f"{a_label} 在 {b_label} 的{rel_name}，"
                f"中心距 {rel.distance:.2f} m。"
            )
            ok = True
        else:
            ok = (rel.relation == asked) or (
                # "near" 这类近义词互认
                {rel.relation, asked} <= {"near", "overlapping"}
            )
            asked_name = RELATION_NAMES.get(asked, asked)
            answer = (
                f"{'是的' if ok else '不是'}：{a_label} 实际在 {b_label} 的{rel_name}"
                f"（你问的是{asked_name}），中心距 {rel.distance:.2f} m。"
            )

        return ReasoningResult(
            query=intent.raw,
            answer=answer,
            targets=[a, b],
            relations=[rel],
            distances={f"{a_label}<->{b_label}_center": rel.distance},
            confidence=float(min(a.confidence, b.confidence)),
            debug={"relation_matches_question": bool(ok)},
        )

    def _answer_list_on(self, intent: Intent) -> ReasoningResult:
        anchor_label = intent.subject
        anchors = self._lookup(anchor_label, top_k=1) if anchor_label else []
        if not anchors:
            return (
                self._not_found(anchor_label, intent.raw) if anchor_label
                else self._answer_describe(intent)
            )

        anchor = anchors[0]
        # "在 X 上/里" = 中心落在 X 包围盒（外扩 5cm）内的其他物体
        margin = 0.05
        lo = anchor.bbox_min - margin
        hi = anchor.bbox_max + margin
        on_it = [
            o for o in self.map.objects
            if o.obj_id != anchor.obj_id and np.all(o.center >= lo) and np.all(o.center <= hi)
        ]
        # 若严格包含没结果，退化为"距离锚点 0.6m 内"
        if not on_it:
            on_it = [o for o, d in self.map.objects_near(anchor.center, radius=0.6, exclude=anchor)]

        if not on_it:
            answer = f"「{anchor_label}」上/里没有检测到其他物体。"
        else:
            desc = "、".join(
                f"{o.label}(距 {center_distance(o, anchor):.2f}m)" for o in on_it[:6]
            )
            answer = f"「{anchor_label}」上/里有 {len(on_it)} 个物体：{desc}。"

        return ReasoningResult(
            query=intent.raw,
            answer=answer,
            targets=[anchor] + on_it,
            distances={o.label: center_distance(o, anchor) for o in on_it[:6]},
            confidence=float(anchor.confidence),
        )

    def _answer_nearest(self, intent: Intent) -> ReasoningResult:
        label = intent.subject
        cands = self._lookup(label, top_k=50) if label else list(self.map.objects)
        if not cands:
            if label:
                return self._not_found(label, intent.raw)
            return self._answer_describe(intent)

        ranked = sorted(
            cands, key=lambda o: float(np.linalg.norm(o.center - self.robot_position))
        )
        top = ranked[0]
        d = float(np.linalg.norm(top.center - self.robot_position))
        answer = (
            f"离机器人最近的「{label or '物体'}」在 "
            f"({top.center[0]:+.2f}, {top.center[1]:+.2f}, {top.center[2]:+.2f}) m，"
            f"距离 {d:.2f} m。"
        )
        return ReasoningResult(
            query=intent.raw,
            answer=answer,
            targets=ranked[: self.top_k],
            distances={f"robot_to_{label or 'object'}": d},
            confidence=float(top.confidence),
        )

    def _answer_count(self, intent: Intent) -> ReasoningResult:
        label = intent.subject
        if not label:
            return self._answer_describe(intent)
        objs = [o for o in self.map.objects if str(o.label).lower() == str(label).lower()]
        if not objs:
            objs = self._lookup(label, top_k=100)
        if not objs:
            return self._not_found(label, intent.raw)
        answer = f"地图里有 {len(objs)} 个「{label}」。"
        return ReasoningResult(
            query=intent.raw, answer=answer, targets=objs[: self.top_k],
            distances={}, confidence=1.0,
        )

    def _answer_describe(self, intent: Intent) -> ReasoningResult:
        objs = list(self.map.objects)
        if not objs:
            return ReasoningResult(
                query=intent.raw,
                answer="当前地图里还没有任何物体（可能建图时没有检测到东西，"
                       "或者 prompts 没覆盖场景里的类别）。",
                backend=self.name, confidence=0.0,
            )
        counts: Dict[str, int] = {}
        for o in objs:
            counts[o.label] = counts.get(o.label, 0) + 1
        desc = "、".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        answer = f"场景中共有 {len(objs)} 个物体：{desc}。"
        return ReasoningResult(
            query=intent.raw, answer=answer, targets=objs[: self.top_k],
            distances={}, confidence=1.0,
        )

    # ---------------- 辅助 ----------------
    def all_relations(self, *, max_distance: float = 3.0) -> List[SpatialRelation]:
        """地图内所有物体两两关系（供可视化/评测）。"""
        return compute_all_relations(
            self.map.objects, thresholds=self.thresholds, max_distance=max_distance
        )

    def __repr__(self) -> str:
        return (
            f"RuleEngine(objects={self.map.num_objects}, top_k={self.top_k}, "
            f"robot_at={np.round(self.robot_position, 2).tolist()})"
        )
