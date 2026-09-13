"""语言查询引擎：把自然语言 query 映射到 3D 位置。

两种匹配器（可单独用，也可融合）
--------------------------------
1. **`LexicalMatcher`（词法匹配，永远可用）**
   基于"标签 + 双语别名表 + 字符 n-gram 模糊匹配"。不依赖任何模型、
   不依赖网络，因此在没有 CLIP 的环境下也能跑通完整 demo。
   代价是只能匹配"检测器吐出来的类别词"及其同义表达。

2. **`EmbeddingMatcher`（嵌入匹配，开放词汇的上限）**
   用 CLIP 类文本编码器把 query 编成向量，与 3D 特征场/物体特征做余弦相似度。
   这是真正的开放词汇 —— 能查询训练时从没见过的表达（"放饮料的东西"）。
   需要 `perception.encoder` 是 `clip` 且装了 transformers。

`QueryEngine` 会自动选择：有文本编码器就走 embedding，否则退回词法；
两者都有时做**混合打分**（取长补短：嵌入管语义泛化，词法管精确别名）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("mapping.query")


# ==========================================================================
# 双语别名表（离线词法匹配的知识来源）
# ==========================================================================
# 说明：这不是"硬编码类别"，而是一张**同义表达映射表**：
# 把用户可能说的各种说法，归并到检测器可能吐出的规范词上。
# 它让离线模式具备真实的可用性，同时不影响嵌入模式的开放词汇能力。
ALIAS_LEXICON: Dict[str, List[str]] = {
    "cup": ["杯子", "水杯", "茶杯", "马克杯", "mug", "glass", "tumbler", "咖啡杯"],
    "bottle": ["瓶子", "水瓶", "饮料瓶", "保温杯", "water bottle", "flask"],
    "table": ["桌子", "餐桌", "桌面", "桌", "desk", "dining table", "countertop", "台面"],
    # 注意："桌子" 同时挂在 table 和 desk 下 —— 这是刻意的：
    # 现实中用户说"桌子"时不区分 dining table 还是 desk，而 SUN RGB-D 的
    # GT 里两者是不同类别。共享别名让"桌子"能命中两类，避免查不到。
    "desk": ["书桌", "办公桌", "写字台", "工作台", "桌子", "桌", "desk table"],
    "chair": ["椅子", "凳子", "座椅", "seat", "stool", "armchair"],
    "sofa": ["沙发", "couch", "settee"],
    "bed": ["床", "床铺", "cot"],
    "door": ["门", "房门", "门口", "doorway", "gate"],
    "window": ["窗户", "窗", "玻璃窗", "windshield"],
    "wall": ["墙", "墙壁", "墙面"],
    "floor": ["地面", "地板", "地上"],
    "ceiling": ["天花板", "顶棚"],
    "monitor": ["显示器", "屏幕", "显示屏", "电脑屏幕", "screen", "display", "tv", "电视"],
    "laptop": ["笔记本电脑", "笔记本", "电脑", "computer", "notebook pc"],
    "keyboard": ["键盘"],
    "mouse": ["鼠标"],
    "phone": ["手机", "电话", "mobile phone", "cellphone", "smartphone"],
    "book": ["书", "书本", "书籍", "notebook", "magazine"],
    "box": ["盒子", "箱子", "纸箱", "carton", "container", "收纳箱"],
    "bag": ["包", "背包", "书包", "手提包", "backpack", "handbag", "luggage"],
    "basket": ["篮子", "筐", "篓子"],
    "trash can": ["垃圾桶", "垃圾箱", "bin", "trash", "garbage can", "rubbish bin"],
    "lamp": ["灯", "台灯", "吊灯", "light", "lighting"],
    "plant": ["植物", "盆栽", "绿植", "flower", "vase", "花瓶"],
    "picture": ["画", "照片", "相框", "painting", "photo", "frame"],
    "clock": ["钟", "时钟", "挂钟", "watch", "时钟表"],
    "refrigerator": ["冰箱", "fridge", "freezer"],
    "sink": ["水槽", "洗手池", "洗碗池", "basin", "washbasin"],
    "toilet": ["马桶", "厕所", "卫生间", "restroom", "wc"],
    "microwave": ["微波炉", "oven", "烤箱"],
    "person": ["人", "行人", "人员", "人类", "human", "people", "man", "woman", "pedestrian"],
    "cat": ["猫", "猫咪", "kitten"],
    "dog": ["狗", "狗狗", "puppy"],
    "food": ["食物", "吃的东西", "meal", "dish"],
    "towel": ["毛巾", "抹布", "cloth"],
    "shelf": ["架子", "置物架", "书架", "搁板", "bookcase", "rack"],
    "cabinet": ["柜子", "橱柜", "衣柜", "storage cabinet", "cupboard", "wardrobe"],
    "pillow": ["枕头", "cushion"],
    "curtain": ["窗帘", "帘子"],
    "sign": ["标志", "指示牌", "标识", "signage", "label"],
    "cart": ["推车", "小车", "trolley", "truck"],
    "robot": ["机器人"],
    "elevator": ["电梯", "lift"],
    "stairs": ["楼梯", "台阶", "staircase"],
}

# 构建反向索引：别名（归一化后） → **规范概念集合**
#
# ⚠️ 必须是多对多（set），不能是单值映射：
# 有些别名天然属于多个概念，例如 "桌子" 既是 table（餐桌）也是 desk（书桌）；
# "desk" 本身也同时挂在 table 的别名表和自己的概念下。
# 用单值 dict 会让后写入的覆盖先写入的，导致"桌子"只能命中其中一个类别。
_ALIAS_TO_CONCEPT: Dict[str, set] = {}
for _concept, _aliases in ALIAS_LEXICON.items():
    _ALIAS_TO_CONCEPT.setdefault(_concept.lower(), set()).add(_concept)
    for _alias in _aliases:
        _ALIAS_TO_CONCEPT.setdefault(_alias.lower(), set()).add(_concept)


def normalize_text(text: str) -> str:
    """归一化：转小写、去首尾空白、压缩连续空白。"""
    return " ".join(str(text).lower().strip().split())


def _char_ngrams(text: str, n: int = 2) -> set:
    """字符 n-gram 集合（中英文通用的模糊匹配基元）。"""
    t = normalize_text(text).replace(" ", "")
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def canonical_concepts(term: str) -> set:
    """把一个词/短语映射到它可能属于的规范概念集合。

    匹配优先级（结果取并集，因为一个表达可能同时属于多个概念）：
    1. 整串直接命中别名表；
    2. 拆词后逐个命中（"my red cup" → cup）；
    3. 长度 ≥2 的别名作为子串出现（"桌子上" → 桌子）；
    4. 单字别名作为子串出现（"桌上" 含 "桌"）。
    """
    t = normalize_text(term)
    out: set = set()
    if not t:
        return out

    if t in _ALIAS_TO_CONCEPT:
        out |= _ALIAS_TO_CONCEPT[t]
    for token in t.replace(",", " ").split():
        if token in _ALIAS_TO_CONCEPT:
            out |= _ALIAS_TO_CONCEPT[token]
    for alias, concepts in _ALIAS_TO_CONCEPT.items():
        if len(alias) >= 2 and alias in t:
            out |= concepts
    for alias, concepts in _ALIAS_TO_CONCEPT.items():
        if len(alias) == 1 and alias in t:
            out |= concepts
    return out


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


# ==========================================================================
# 匹配器抽象
# ==========================================================================
class TextMatcher(ABC):
    """文本匹配器接口。

    输入 query 文本 + 一组"候选"（物体或体素），输出每个候选的匹配分数。
    分数不需要归一化到 [0,1]，但必须是"越大越相关"。
    """

    name: str = "base"
    supports_open_vocabulary: bool = False

    @abstractmethod
    def score(
        self,
        text: str,
        *,
        labels: Sequence[str],
        features: Optional[np.ndarray] = None,
        counts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """返回 (M,) 分数数组。

        Parameters
        ----------
        text
            查询文本。
        labels
            每个候选的主标签（长度 M）。
        features
            (M,D) 候选特征。词法匹配器会忽略它。
        counts
            (M,) 每个候选的观测数（可用于置信度微调）。
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class LexicalMatcher(TextMatcher):
    """词法匹配：别名表 + 子串 + 字符 n-gram 模糊匹配。

    打分规则（分层，取最高）：
    1. 规范概念完全相同         → 1.00
    2. 一方是另一方的子串       → 0.85
    3. 字符 bigram Jaccard      → 0.0 ~ 0.80（按相似度缩放）
    """

    name = "lexical"
    supports_open_vocabulary = False

    def __init__(self, lexicon: Optional[Dict[str, List[str]]] = None) -> None:
        if lexicon is not None:
            # 允许外部扩展别名表（同样保持"别名 → 概念集合"的多对多关系）
            reverse: Dict[str, set] = {}
            for concept, aliases in lexicon.items():
                reverse.setdefault(concept.lower(), set()).add(concept)
                for alias in aliases:
                    reverse.setdefault(alias.lower(), set()).add(concept)
            self._reverse = reverse
        else:
            self._reverse = _ALIAS_TO_CONCEPT

    def _concepts(self, term: str) -> set:
        t = normalize_text(term)
        if not t:
            return set()
        out: set = set()
        if t in self._reverse:
            out |= self._reverse[t]
        for token in t.replace(",", " ").split():
            if token in self._reverse:
                out |= self._reverse[token]
        for alias, concepts in self._reverse.items():
            if len(alias) >= 1 and alias in t:
                out |= concepts
        return out

    def score(
        self,
        text: str,
        *,
        labels: Sequence[str],
        features: Optional[np.ndarray] = None,
        counts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        q = normalize_text(text)
        if not q or not labels:
            return np.zeros(len(labels), dtype=np.float32)

        q_concepts = self._concepts(q)
        q_grams = _char_ngrams(q)
        scores = np.zeros(len(labels), dtype=np.float32)

        for i, raw_label in enumerate(labels):
            label = normalize_text(raw_label)
            if not label:
                continue

            # 1) 规范概念命中
            l_concepts = self._concepts(label)
            if q_concepts and l_concepts and (q_concepts & l_concepts):
                scores[i] = 1.0
                continue

            # 2) 子串包含
            if label in q or q in label:
                scores[i] = max(scores[i], 0.85)
                continue

            # 3) 字符 n-gram 模糊匹配
            sim = jaccard(q_grams, _char_ngrams(label))
            scores[i] = max(scores[i], float(min(0.8, sim * 1.6)))

        # 用观测数做轻微置信度调制（观测多的物体更可信，但影响很小，
        # 避免"观测多的大物体"永远压过"观测少的正确小物体"）
        if counts is not None and len(counts) == len(scores):
            c = np.asarray(counts, dtype=np.float32)
            if c.max() > 0:
                scores = scores * (0.9 + 0.1 * (c / max(c.max(), 1e-6)))

        return scores.astype(np.float32)


class EmbeddingMatcher(TextMatcher):
    """嵌入匹配：文本编码器 → 向量 → 与候选特征余弦相似度。

    真·开放词汇，但**原始余弦相似度不能直接当分数用**。这里做了三件事
    让它变成可用的打分器（每一条都是实测踩出来的）：

    1. **提示词集成（prompt ensembling）**
       CLIP 对提示词措辞敏感。把 query 套进多个模板
       （"a photo of a {}" / "a cropped photo of a {}" ...）取平均，
       比单模板稳定得多。

    2. **中文 → 英文概念桥接**
       `openai/clip-vit-base-patch32` 是**纯英文模型**，喂中文等于喂噪声。
       实测："杯子" 的嵌入查询会命中 monitor。
       所以先用双语别名表把中文映射到英文概念，再编码 ——
       这样中文查询才真正走嵌入路径（而不是靠词法兜底）。

    3. **空文本校准（null-text calibration）** ★ 关键
       原始余弦只在 0.1~0.3 的窄区间里波动，直接线性映射到 [0,1] 会让
       **任何 query 对任何物体都得到 ~0.55 分** —— 于是"冰箱"（地图里没有）
       也会返回一个物体。实测就是这个问题。
       解法：把 query 的相似度和一组"空文本"（"a photo of nothing" /
       "an empty background" ...）的相似度放进同一个 softmax，
       取 query 那一项的概率作为分数。这样分数天然具备**拒识能力**：
       与空文本相比没有优势的候选，分数会接近 0。
    """

    name = "embedding"
    supports_open_vocabulary = True

    #: CLIP 提示词模板（来自 CLIP 论文的 prompt engineering 结论）
    DEFAULT_TEMPLATES: Tuple[str, ...] = (
        "a photo of a {}.",
        "a photo of the {}.",
        "a cropped photo of a {}.",
        "a close-up photo of a {}.",
        "a photo of a small {}.",
        "a photo of a large {}.",
    )

    #: 空文本（用于校准）。它们代表"这里没有要查的东西"。
    DEFAULT_NULL_TEXTS: Tuple[str, ...] = (
        "a photo of nothing.",
        "an empty background.",
        "a photo of a blank wall.",
        "an out of focus image.",
        "a photo of the floor.",
        "a random texture.",
    )

    def __init__(
        self,
        text_encoder: Any,
        *,
        normalize: bool = True,
        templates: Optional[Tuple[str, ...]] = None,
        null_texts: Optional[Tuple[str, ...]] = None,
        temperature: float = 0.01,
        calibrate: bool = True,
        translate: bool = True,
    ) -> None:
        if text_encoder is None:
            raise ValueError("EmbeddingMatcher 需要一个 text_encoder")
        self.text_encoder = text_encoder
        self.normalize = bool(normalize)
        self.templates = tuple(templates) if templates is not None else self.DEFAULT_TEMPLATES
        self.null_texts = tuple(null_texts) if null_texts is not None else self.DEFAULT_NULL_TEXTS
        self.temperature = float(temperature)
        self.calibrate = bool(calibrate)
        self.translate = bool(translate)

        # 缓存：空文本特征（只需算一次）
        self._null_feats: Optional[np.ndarray] = None
        # 缓存：query → 英文概念 的翻译结果
        self._translate_cache: Dict[str, str] = {}
        # 诊断信息：让调用方能看清"这次查询到底走了哪条路"
        self.last_debug: Dict[str, Any] = {}

    # ---------------- 中文桥接 ----------------
    def to_english(self, text: str) -> str:
        """把查询文本归一成英文概念（能映射就映射，不能就原样返回）。

        必须**保守**地桥接，否则会引入假阳性。实测踩过的坑：
        用"子串包含"做桥接时，`"something to drink from the table"`
        会被映射成 `"table"`（因为句子里含 "table"），
        把一个自由表达硬掰成了具体概念。

        所以只在两种情况下桥接：
        1. **文本含中文**（英文 CLIP 根本读不懂中文，必须桥接）；
        2. **文本很短（≤2 个词）**（此时它就是"一个概念"，归一化无副作用）。

        自由表达的长句原样透传，交给 CLIP 自己去做语义泛化 ——
        这正是开放词汇应当具备的能力。
        """
        if not self.translate:
            return str(text)

        raw = str(text).strip()
        key = normalize_text(raw)
        if key in self._translate_cache:
            return self._translate_cache[key]

        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in raw)
        is_short = len(key.split()) <= 2

        out = raw
        if has_cjk or is_short:
            concepts = canonical_concepts(key)
            if concepts:
                # 多概念时取"最短"的（避免歧义展开成一句话）
                out = sorted(concepts, key=lambda c: (len(c), c))[0]

        self._translate_cache[key] = out
        return out

    # ---------------- 编码 ----------------
    def encode_query(self, text: str, *, ensemble: bool = True) -> np.ndarray:
        """把 query 编成 (D,) 单位向量。

        `ensemble=True` 且文本是"概念"（短词）时，套多个模板取平均。
        已经是完整句子（含空格且长度较大）时不做模板化 —— 否则会变成
        "a photo of a something to drink from." 这种别扭的句子。
        """
        english = self.to_english(text)
        self.last_debug["raw_query"] = str(text)
        self.last_debug["english_query"] = english

        use_ensemble = bool(ensemble) and len(english.split()) <= 3
        if use_ensemble:
            texts = [tpl.format(english) for tpl in self.templates]
        else:
            texts = [english if english.endswith(".") else f"{english}." ]

        vecs = np.asarray(self.text_encoder.encode_text(texts), dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        mean = vecs.mean(axis=0)

        if self.normalize:
            n = float(np.linalg.norm(mean))
            if n > 1e-8:
                mean = mean / n
        self.last_debug["num_templates"] = len(texts)
        return mean

    def _null_features(self, dim: int) -> Optional[np.ndarray]:
        """空文本特征矩阵 (K,D)，惰性计算并缓存。"""
        if not self.calibrate:
            return None
        if self._null_feats is not None and self._null_feats.shape[1] == dim:
            return self._null_feats
        try:
            feats = np.asarray(self.text_encoder.encode_text(list(self.null_texts)),
                               dtype=np.float32)
            if self.normalize:
                feats = feats / np.clip(
                    np.linalg.norm(feats, axis=1, keepdims=True), 1e-8, None
                )
            self._null_feats = feats
            return feats
        except Exception as exc:
            logger.warn(f"计算空文本特征失败（{exc}），退化为未校准的余弦分数")
            self.calibrate = False
            return None

    # ---------------- 打分 ----------------
    def score(
        self,
        text: str,
        *,
        labels: Sequence[str],
        features: Optional[np.ndarray] = None,
        counts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if features is None or len(features) == 0:
            return np.zeros(len(labels), dtype=np.float32)
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)

        q = self.encode_query(text)
        if q.shape[0] != feats.shape[1]:
            logger.warn(
                f"文本特征维度({q.shape[0]}) 与地图特征维度({feats.shape[1]}) 不一致，"
                "嵌入匹配被跳过（检查建图与查询是否用了同一个 encoder）"
            )
            return np.zeros(feats.shape[0], dtype=np.float32)

        if self.normalize:
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.clip(norms, 1e-8, None)

        # ---- 路径 A：后端自带原生校准（如 SigLIP 的 sigmoid 概率）----
        # 有了模型原生校准就不需要空文本 trick —— 它的分数本身就是
        # "这一个图文对匹配的概率"，天然可设阈值、天然可拒识。
        if getattr(self.text_encoder, "is_calibrated", False) and hasattr(self.text_encoder, "pair_scores"):
            try:
                probs = self.text_encoder.pair_scores(feats, q.reshape(1, -1))
                probs = np.asarray(probs, dtype=np.float32).reshape(-1)
                if probs.shape[0] == feats.shape[0]:
                    self.last_debug["calibration"] = "native(siglip)"
                    self.last_debug["top_score"] = float(probs.max()) if probs.size else 0.0
                    self.last_debug["mean_score"] = float(probs.mean()) if probs.size else 0.0
                    return probs
            except Exception as exc:
                logger.debug(f"原生校准失败，退回空文本校准：{exc}")

        sim_q = feats @ q                                  # (M,) ∈ [-1,1]
        self.last_debug["calibration"] = "null_text"
        self.last_debug["top_sim"] = float(np.max(sim_q)) if sim_q.size else 0.0
        self.last_debug["mean_sim"] = float(np.mean(sim_q)) if sim_q.size else 0.0

        nulls = self._null_features(feats.shape[1])
        if nulls is None:
            # 未校准：把余弦线性映射到 [0,1]（保留兼容行为）
            return ((sim_q + 1.0) * 0.5).astype(np.float32)

        sim_null = feats @ nulls.T                         # (M,K)
        # 与"最像的那个空文本"比较 —— 用 max 而不是 mean，避免被多个空文本平均稀释
        sim_null_max = sim_null.max(axis=1)                # (M,)

        # softmax over {query, best_null} → query 的概率
        logits = np.stack([sim_q, sim_null_max], axis=1) / max(self.temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)        # 数值稳定
        exp = np.exp(logits)
        prob = exp[:, 0] / np.clip(exp.sum(axis=1), 1e-12, None)

        self.last_debug["margin"] = float(np.mean(sim_q - sim_null_max))
        return prob.astype(np.float32)


# ==========================================================================
# 查询引擎
# ==========================================================================
class QueryEngine:
    """把匹配器、层级选择、空间重排串起来的高层查询接口。

    Examples
    --------
    >>> engine = QueryEngine(semantic_map)          # 自动选择匹配器
    >>> results = engine.query("杯子在哪", top_k=3)
    """

    def __init__(
        self,
        semantic_map,
        *,
        text_encoder: Any = None,
        lexical: bool = True,
        embedding_weight: float = 0.6,
        lexical_weight: float = 0.4,
        min_score_embedding: float = 0.5,
    ) -> None:
        self.map = semantic_map
        encoder = text_encoder if text_encoder is not None else getattr(semantic_map, "text_encoder", None)

        self.lexical_matcher: Optional[LexicalMatcher] = LexicalMatcher() if lexical else None
        self.embedding_matcher: Optional[EmbeddingMatcher] = None
        if encoder is not None and getattr(encoder, "supports_text", False):
            try:
                self.embedding_matcher = EmbeddingMatcher(encoder)
            except Exception as exc:  # pragma: no cover
                logger.warn(f"构建 EmbeddingMatcher 失败，退回词法匹配：{exc}")

        self.embedding_weight = float(embedding_weight)
        self.lexical_weight = float(lexical_weight)

        # ---- 嵌入阈值：**按编码器自适应**，且**按本引擎的模式**自适应 ----
        # 不同后端的分数尺度差两个数量级：
        #   CLIP 经空文本校准后是"概率"，命中在 0.9+；
        #   SigLIP 的 sigmoid 概率天然保守，命中只有 0.01~0.05。
        # 用同一个阈值必然坏掉一边（实测：0.5 会让 SigLIP 全部返回空）。
        #
        # 还要区分**角色**：本引擎没有词法兜底时（`lexical=False`），
        # 嵌入阈值就是唯一的接受判定，必须用独立模式下标定的那个值；
        # 有词法兜底时它只是精度过滤器，可以用更严的值。
        # 实测两者最优值差 40 倍（0.02 vs 0.0005），详见
        # `Encoder.suggested_standalone_threshold` 与 `scripts/16_eval_openvocab_query.py`。
        standalone = self.lexical_matcher is None and self.embedding_matcher is not None
        #: 自监督阈值标定的诊断信息（正负样本数、分离度）
        self.calibration_debug: Dict[str, Any] = {}
        #: 嵌入阈值来源："auto"（自监督标定）/ "encoder"（编码器声明）/ "explicit"（用户指定）
        self.embedding_threshold_source: str = "encoder"
        # ⚠️ 注意：`getattr` 的默认值只在**属性不存在**时生效；
        # 地图上的属性是显式设为 None 的（表示"自适应"），所以要判 None 而不是靠 getattr。
        explicit = getattr(semantic_map, "query_min_score_embedding", None)
        if explicit is not None:
            self.min_score_embedding = float(explicit)
            self.embedding_threshold_source = "explicit"
        else:
            # 先试**自监督就地标定**（优于任何硬编码常量，见方法文档）；
            # 标定不了（物体太少/没有共同标签）才退回编码器声明值。
            calib = (self._calibrate_standalone_threshold()
                     if standalone else None)
            if calib is not None:
                self.min_score_embedding = float(calib)
                self.embedding_threshold_source = "auto"
            else:
                self.min_score_embedding = self._suggest_threshold(
                    encoder, min_score_embedding, standalone=standalone)
                self.embedding_threshold_source = "encoder"
        #: 该阈值是否取自"独立模式"（区分角色用，供诊断）
        self.embedding_threshold_is_standalone = (explicit is None and standalone)
        #: 词法路径的接受阈值（由 query() 的 min_score 参数在运行时覆盖）
        #:
        #: ⚠️ 这个值**不能调低**。词法打分是分层的：概念命中 1.00 / 子串 0.85 /
        #: 字符 bigram Jaccard 最高 0.80。第三层是噪声源 —— 任意两个词只要共享
        #: 少量 bigram 就能拿到 0.1~0.3 分（例：`bicycle` 与 `bottle` 共享 `le`，
        #: Jaccard 0.09 → 0.145 分）。阈值设 0.05 时这些噪声全部被接受，
        #: 于是查询 "冰箱"/"飞机" 这类**地图里没有的类别**也会返回一个错误物体。
        #:
        #: 实测（scripts/16_eval_openvocab_query.py，6 场景 / 151 条未见表达 / 72 条未见类别）：
        #:   阈值 0.05 → 拒识率  72.2%，综合分 0.603
        #:   阈值 0.50 → 拒识率 100.0%，综合分 0.828
        #: 命名类命中率两者都是 100%（不损失真命中），只有 1/40 条描述性查询受影响。
        #: 0.5 恰好卡在"第三层噪声（<0.5）"与"真实相似（sim≥0.31）"之间。
        self.min_score_lexical: float = 0.5
        #: 最近一次查询的诊断信息（路由哪条路、各路分数）
        self.last_debug: Dict[str, Any] = {}

    @staticmethod
    def _suggest_threshold(encoder: Any, fallback: float,
                           standalone: bool = False) -> float:
        """从编码器读取建议阈值；读不到就用传入的默认值。

        `standalone=True` 时优先读 `suggested_standalone_threshold`
        （纯嵌入模式专用）—— 少数老后端没实现这个属性，
        此时 `getattr` 会回落到 `suggested_pair_threshold`。
        """
        if encoder is None:
            return float(fallback)
        if standalone:
            thr = getattr(encoder, "suggested_standalone_threshold", None)
            if isinstance(thr, (int, float)) and thr > 0:
                return float(thr)
        thr = getattr(encoder, "suggested_pair_threshold", None)
        if isinstance(thr, (int, float)) and thr > 0:
            return float(thr)
        return float(fallback)

    # ---------------- 自监督阈值标定 ----------------
    def _calibrate_standalone_threshold(self) -> Optional[float]:
        """用**地图自身的 (特征, 标签) 对**就地标定嵌入阈值（无需任何 GT 标注）。

        为什么要做这件事
        ----------------
        嵌入阈值不能硬编码，原因有两层（都是实测踩出来的）：

        1. **同一编码器在不同特征管线上尺度不同** —— SigLIP 的 0.02 是在
           *检测区域特征* 上标的，而查询比对的是*多视角融合后的地图物体特征*，
           尺度小约 20 倍；
        2. **同一阈值在不同模式下角色不同** —— hybrid 里它是"精度过滤器"，
           纯嵌入里它是唯一的"接受阈值"，两者最优值差 40 倍。

        靠离线扫描能救急，但**换个数据集/换个编码器就得重扫**。
        本方法把它变成自标定：地图建好后，每个物体都带着检测器给的标签，
        这本身就是一份自监督标定集 —— 物体 i 对**自己**的标签应当得高分、
        对**别的**标签应当得低分。据此选阈值即可。

        ⚠️ **必须走查询时完全相同的那条打分路径**（同一个 `EmbeddingMatcher.score`）。
        否则标出来的阈值不在查询分数的尺度上 —— 这正是本项目踩过的坑：
        用 `pair_scores` 直接标、却用 matcher 校准后的分数查询，两者不是一回事。

        选点准则：在候选阈值上**最大化平衡准确率** `TPR + TNR`。
        用平衡准确率而不是准确率，是因为正负样本极不平衡
        （N 个物体对 1 个自身标签，负样本是正样本的 U-1 倍）。
        并列最优时取**最大的**阈值 —— 同样是满分，宁可严一点。

        Returns
        -------
        float or None
            标定出的阈值；地图物体太少 / 只有一个类别 / 打分异常时返回 None
            （调用方回退到编码器声明值）。
        """
        if self.embedding_matcher is None:
            return None
        objs = getattr(self.map, "objects", None)
        if not objs or len(objs) < 2:
            return None
        feats = np.stack([o.feature for o in objs], 0) if objs[0].feature.size else None
        if feats is None:
            return None

        labels = [o.label for o in objs]
        uniq = sorted(set(labels))
        if len(uniq) < 2:
            # 只有一个类别时无法构造负样本 —— 标不出来
            return None
        counts = np.array([o.num_voxels for o in objs], dtype=np.float32)

        positives: List[float] = []
        negatives: List[float] = []
        for lab in uniq:
            try:
                s = np.asarray(
                    self.embedding_matcher.score(lab, labels=labels,
                                                 features=feats, counts=counts),
                    dtype=np.float64)
            except Exception:  # pragma: no cover - 编码器异常时不阻塞构造
                return None
            if s.shape != (len(labels),):
                return None
            want = np.asarray([l == lab for l in labels], dtype=bool)
            if not want.any():
                continue
            positives.extend(s[want].tolist())
            negatives.extend(s[~want].tolist())

        if len(positives) < 2 or len(negatives) < 2:
            return None

        pos = np.asarray(positives, dtype=np.float64)
        neg = np.asarray(negatives, dtype=np.float64)
        best_t: Optional[float] = None
        best_ba = -1.0
        # 候选阈值只需在观测到的分数上取（中间值不会改变任何一次判定）
        for t in np.unique(np.concatenate([pos, neg])):
            ba = float((pos >= t).mean() + (neg < t).mean())
            # 用 >= 比较 → 并列最优时保留**更大**的阈值（更保守）
            if t > 0 and ba >= best_ba:
                best_ba, best_t = ba, float(t)

        if best_t is None or not np.isfinite(best_t) or best_t <= 0:
            return None
        #: 诊断信息：分离度 2.0 表示正负样本完全可分
        self.calibration_debug = {
            "n_pos": int(pos.size), "n_neg": int(neg.size),
            "pos_median": float(np.median(pos)), "neg_median": float(np.median(neg)),
            "balanced_accuracy": best_ba if best_ba >= 0 else None,
        }
        return best_t

    # ---------------- 内部 ----------------
    @property
    def mode(self) -> str:
        if self.embedding_matcher is not None and self.lexical_matcher is not None:
            return "hybrid"
        if self.embedding_matcher is not None:
            return "embedding"
        return "lexical"

    def _fuse(self, lexical: Optional[np.ndarray], embedding: Optional[np.ndarray]) -> np.ndarray:
        """融合两路分数：加权和，但保留各自的最大值作为下界。

        为什么不用简单加权平均？因为两路尺度不可比 —— 词法命中就是 1.0，
        嵌入相似度通常 0.7~0.9。用 max 兜底可以避免"词法强命中被嵌入稀释"。
        """
        if embedding is None:
            return lexical if lexical is not None else np.zeros(0, dtype=np.float32)
        if lexical is None:
            return embedding
        wl, we = self.lexical_weight, self.embedding_weight
        total = wl + we
        fused = (lexical * wl + embedding * we) / max(total, 1e-8)
        return np.maximum(fused, np.maximum(lexical * 0.95, embedding * 0.95)).astype(np.float32)

    def _score_candidates(self, text: str, labels, features, counts):
        """返回 (融合分数, 词法分数, 嵌入分数)。三者都可能为 None。"""
        lex = None
        emb = None
        if self.lexical_matcher is not None:
            lex = self.lexical_matcher.score(text, labels=labels, features=features, counts=counts)
        if self.embedding_matcher is not None:
            emb = self.embedding_matcher.score(text, labels=labels, features=features, counts=counts)
        return self._fuse(lex, emb), lex, emb

    def _accept_mask(self, lex, emb) -> np.ndarray:
        """决定哪些候选可以进入结果。

        **为什么不能用一个阈值？**
        词法分数是"命中就是 1.0、不命中就是 0"的硬信号；
        嵌入分数是"对空文本校准后的 softmax 概率"，天然落在中间区间。
        用同一个阈值会出现两种错误：
        - 阈值低（0.05）→ 嵌入路径把"冰箱"这种地图里没有的查询也返回一个 0.29 分的错误答案；
        - 阈值高（0.5）→ 词法的合法命中（比如子串匹配给的 0.85）反而被误杀。

        所以这里用**双阈值**：任一路径达标即接受。
        """
        if lex is None and emb is None:
            return np.zeros(0, dtype=bool)

        mask = None
        if lex is not None:
            m = lex >= self.min_score_lexical
            mask = m if mask is None else (mask | m)
        if emb is not None:
            m = emb >= self.min_score_embedding
            mask = m if mask is None else (mask | m)
        return mask

    def _rank_score(self, fused, lex, emb) -> np.ndarray:
        """排序用分数：取两路中"达标那一侧"的较高值，避免不达标的一路拖低排名。"""
        score = fused
        if lex is not None and emb is not None:
            lex_ok = lex >= self.min_score_lexical
            emb_ok = emb >= self.min_score_embedding
            # 只有一路达标 → 用那一路的分数；都达标 → 取较大值
            score = np.where(
                lex_ok & ~emb_ok, lex,
                np.where(emb_ok & ~lex_ok, emb, np.maximum(lex, emb)),
            )
        elif lex is not None:
            score = lex
        elif emb is not None:
            score = emb
        return score

    # ---------------- 公开查询 ----------------
    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        level: str = "auto",
        min_score: Optional[float] = None,
        min_score_embedding: Optional[float] = None,
        spatial_rerank: bool = True,
    ) -> List:
        """执行查询。

        Parameters
        ----------
        level : {"auto", "object", "voxel"}
            `auto` 优先物体层；物体层为空（还没聚类）时退回体素层。
        min_score
            词法路径的接受阈值（None 时用引擎默认 0.05）。
        min_score_embedding
            嵌入路径的接受阈值（None 时用 0.5）。
            两条路径阈值分开是必须的，原因见 `_accept_mask`。
        spatial_rerank
            是否用"观测充分度"对同分候选做微调（避免单个噪声体素排前面）。
        """
        from roboground.mapping.semantic_map import QueryResult  # 延迟导入

        if not text or not str(text).strip():
            return []

        if min_score is not None:
            self.min_score_lexical = float(min_score)
        if min_score_embedding is not None:
            self.min_score_embedding = float(min_score_embedding)

        results: List[QueryResult] = []
        use_object = level in ("auto", "object") and self.map.num_objects > 0
        use_voxel = level == "voxel" or (level == "auto" and not use_object)

        if self.embedding_matcher is not None:
            self.embedding_matcher.last_debug = {}
        matched_by = self.mode

        # ---------- 物体层 ----------
        if use_object:
            objs = self.map.objects
            labels = [o.label for o in objs]
            feats = np.stack([o.feature for o in objs], axis=0) if objs and objs[0].feature.size else None
            counts = np.array([o.num_voxels for o in objs], dtype=np.float32)

            fused, lex, emb = self._score_candidates(text, labels, feats, counts)
            accept = self._accept_mask(lex, emb)
            score = self._rank_score(fused, lex, emb)

            if spatial_rerank:
                conf = np.array([o.confidence for o in objs], dtype=np.float32)
                score = score * (0.92 + 0.08 * conf)

            n_accept = int(accept.sum()) if accept.size else 0
            order = np.argsort(-score)
            for rank, idx in enumerate(order):
                if accept.size and not accept[idx]:
                    continue
                if score[idx] <= 0:
                    continue
                results.append(QueryResult(
                    score=float(score[idx]),
                    level="object",
                    obj=objs[int(idx)],
                    center=objs[int(idx)].center,
                    label=objs[int(idx)].label,
                    matched_by=matched_by,
                    extra={
                        "rank": rank,
                        "lexical_score": float(lex[idx]) if lex is not None else None,
                        "embedding_score": float(emb[idx]) if emb is not None else None,
                    },
                ))
                if len(results) >= top_k:
                    break

            self.last_debug = {
                "level": "object",
                "mode": matched_by,
                "num_candidates": len(objs),
                "num_accepted": n_accept,
                "min_score_lexical": self.min_score_lexical,
                "min_score_embedding": self.min_score_embedding,
                **(dict(self.embedding_matcher.last_debug) if self.embedding_matcher else {}),
            }

        # ---------- 体素层 ----------
        if use_voxel and (not results or level == "voxel"):
            grid = self.map.voxel_grid
            if grid.num_voxels > 0:
                top_votes = grid.top_labels(top=1)
                labels = [v[0][0] if v else "" for v in top_votes]
                feats = grid.features
                counts = grid.counts.astype(np.float32)

                fused, lex, emb = self._score_candidates(text, labels, feats, counts)
                accept = self._accept_mask(lex, emb)
                score = self._rank_score(fused, lex, emb)
                order = np.argsort(-score)

                for idx in order:
                    if accept.size and not accept[idx]:
                        continue
                    # 避免与物体层结果重复
                    already = False
                    for r in results:
                        if r.obj is not None and idx in set(r.obj.voxel_ids.tolist()):
                            already = True
                            break
                    if already:
                        continue
                    center = grid.centers[int(idx)]
                    results.append(QueryResult(
                        score=float(score[idx]),
                        level="voxel",
                        center=center,
                        label=labels[int(idx)] or None,
                        voxel_ids=np.array([int(idx)], dtype=np.int64),
                        matched_by=matched_by,
                    ))
                    if len(results) >= top_k:
                        break

        return results[:top_k]
