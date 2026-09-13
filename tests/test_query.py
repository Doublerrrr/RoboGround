"""语言查询测试：词法匹配、别名表、查询过滤、地图查询。"""

from __future__ import annotations

import numpy as np
import pytest

from roboground.mapping.query import (
    ALIAS_LEXICON,
    EmbeddingMatcher,
    LexicalMatcher,
    QueryEngine,
    canonical_concepts,
    jaccard,
    normalize_text,
)
from roboground.mapping.semantic_map import SemanticMap, SemanticObject
from roboground.geometry.voxel import VoxelGrid


# ==========================================================================
# 文本工具
# ==========================================================================
def test_normalize_text():
    assert normalize_text("  Hello   World ") == "hello world"


def test_jaccard():
    assert jaccard({1, 2}, {1, 2}) == 1.0
    assert jaccard({1}, {2}) == 0.0
    assert jaccard(set(), {1}) == 0.0


@pytest.mark.parametrize("term,expected_concept", [
    ("杯子", "cup"), ("水杯", "cup"), ("cup", "cup"),
    ("桌子", "table"), ("椅子", "chair"), ("门", "door"),
    ("垃圾桶", "trash can"), ("冰箱", "refrigerator"),
])
def test_canonical_concepts(term, expected_concept):
    assert expected_concept in canonical_concepts(term)


def test_lexicon_is_bilingual_and_covers_robot_scenes():
    """别名表必须同时含中英文（否则中文查询完全不可用）。"""
    for concept in ("cup", "table", "chair", "door", "person", "trash can"):
        aliases = ALIAS_LEXICON[concept]
        assert any(any("\u4e00" <= ch <= "\u9fff" for ch in a) for a in aliases), \
            f"{concept} 缺少中文别名"


# ==========================================================================
# 词法匹配器
# ==========================================================================
def test_lexical_exact_concept_match():
    m = LexicalMatcher()
    scores = m.score("杯子在哪", labels=["cup", "table", "chair"])
    assert scores[0] > 0.9
    assert scores[1] < 0.5 and scores[2] < 0.5


def test_lexical_english_query():
    m = LexicalMatcher()
    scores = m.score("where is the table", labels=["cup", "table"])
    assert scores[1] > scores[0]


def test_lexical_substring_fallback():
    m = LexicalMatcher()
    scores = m.score("monitor", labels=["monitor_stand"])
    assert scores[0] > 0.5


def test_lexical_no_match_returns_zero():
    m = LexicalMatcher()
    scores = m.score("宇宙飞船", labels=["cup", "table"])
    assert float(scores.max()) < 0.2


def test_lexical_empty_inputs():
    m = LexicalMatcher()
    assert m.score("", labels=["cup"]).shape == (1,)
    assert m.score("cup", labels=[]).shape == (0,)


def test_lexical_returns_one_score_per_label():
    m = LexicalMatcher()
    labels = ["cup", "table", "chair", "door"]
    assert m.score("杯子", labels=labels).shape == (len(labels),)


# ==========================================================================
# 嵌入匹配器（用假的编码器，避免依赖模型）
# ==========================================================================
class _FakeTextEncoder:
    """假文本编码器：把已知概念映射到固定向量。

    需要处理两种输入形式：
    1. **提示词模板**（"a photo of a cup."）—— 提示词集成会套模板，这里要能还原出概念；
    2. **自由文本**（"something to drink from"）—— 直接按原串匹配。

    空文本用 `[1,1,1]/√3`（"所有方向的平均"）建模，这是对 CLIP 里
    "空文本与任意图像都有中等相似度"这一事实的合理近似 ——
    用反相关向量建模会让校准测试失去意义（见 calibration 测试的注释）。
    """

    supports_text = True
    feature_dim = 3
    # 模仿 SigLIP 的"阈值随角色不同"：hybrid 用严的，纯嵌入用松的。
    # 两个值刻意差一个数量级，好让"用错了角色"在测试里能被看出来。
    suggested_pair_threshold = 0.02
    suggested_standalone_threshold = 0.002

    CONCEPTS = {
        "cup": np.array([1.0, 0.0, 0.0]),
        "table": np.array([0.0, 1.0, 0.0]),
        "chair": np.array([0.0, 0.0, 1.0]),
        # 近义表达 → 与 cup 方向接近但不相同（用于验证嵌入的语义泛化）
        "something to drink from": np.array([0.9, 0.1, 0.0]),
        # 空文本 → 指向"所有方向的平均"，与任何单一概念都有中等相似度
        "null": np.ones(3) / np.sqrt(3.0),
    }

    @classmethod
    def _vec(cls, text: str) -> np.ndarray:
        t = str(text).lower().strip().rstrip(".")
        if t in cls.CONCEPTS:
            return cls.CONCEPTS[t]
        for key, vec in cls.CONCEPTS.items():
            if key != "null" and key in t:
                return vec
        return cls.CONCEPTS["null"]

    def encode_text(self, texts):
        out = [self._vec(t) for t in texts]
        arr = np.stack(out, axis=0).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-8, None)


def test_embedding_matcher_ranks_matching_concept_first():
    m = EmbeddingMatcher(_FakeTextEncoder())
    feats = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scores = m.score("cup", labels=["cup", "table", "chair"], features=feats)

    assert scores.shape == (3,)
    assert scores[0] > scores[1]      # 与 "cup" 特征更接近的排前
    assert scores[0] > scores[2]
    assert 0.0 <= float(scores.min()) and float(scores.max()) <= 1.0   # 概率语义


def test_embedding_matcher_calibration_rejects_unrelated():
    """★ 关键回归：地图里没有的东西必须拿不到高分。

    没有空文本校准时，余弦被线性映射到 [0,1]，任何 query 对任何物体
    都能得到 ~0.5 分 —— 实测导致"冰箱"返回了 chair。校准之后，
    与空文本相比没有优势的候选必须掉到阈值以下。
    """
    m = EmbeddingMatcher(_FakeTextEncoder())
    # 地图里只有 chair，查询 "cup"
    feats = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    scores = m.score("cup", labels=["chair"], features=feats)
    assert float(scores[0]) < 0.5, f"不相关候选不该超过接受阈值，实际 {scores[0]:.3f}"


def test_embedding_matcher_chinese_bridging():
    """中文 query 应被桥接成英文概念（英文 CLIP 才能理解）。"""
    m = EmbeddingMatcher(_FakeTextEncoder())
    assert m.to_english("杯子") == "cup"
    assert m.to_english("椅子") == "chair"
    # 已是英文概念则原样保留
    assert m.to_english("cup") == "cup"
    # 自由表达不做翻译（交给 CLIP 语义泛化）
    assert m.to_english("something to drink from") == "something to drink from"


def test_embedding_matcher_prompt_ensembling_used_for_short_concept():
    m = EmbeddingMatcher(_FakeTextEncoder())
    m.encode_query("cup")
    assert m.last_debug["num_templates"] == len(m.templates)
    # 长句不该被套模板（否则变成 "a photo of a something to drink from."）
    m.encode_query("something to drink from the table")
    assert m.last_debug["num_templates"] == 1


def test_embedding_matcher_dim_mismatch_is_safe():
    m = EmbeddingMatcher(_FakeTextEncoder())
    feats = np.zeros((2, 5), dtype=np.float32)     # 维度不匹配
    scores = m.score("cup", labels=["a", "b"], features=feats)
    assert scores.shape == (2,)
    assert float(scores.max()) == 0.0     # 安全返回 0，而不是崩


def test_embedding_matcher_requires_encoder():
    with pytest.raises(ValueError):
        EmbeddingMatcher(None)


# ==========================================================================
# 地图查询
# ==========================================================================
def _make_map(labels_and_centers):
    """构造一张最小的语义地图（每个物体一个体素）。"""
    grid = VoxelGrid(voxel_size=0.1, feature_dim=3)
    for i, (label, center) in enumerate(labels_and_centers):
        grid.add(np.asarray([center], dtype=np.float64),
                 np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                 labels=[label], frame_id=f"f{i}")
    smap = SemanticMap(grid)
    # 刻意**不**覆盖 query_min_score：让测试跑在**生产默认阈值**上。
    # 曾经这里硬编码 0.05，等于把测试和生产行为解耦 —— 默认值坏掉也测不出来。
    return smap


def test_map_query_returns_empty_for_empty_map():
    smap = _make_map([])
    assert smap.query_text("杯子在哪") == []


def test_map_voxel_level_query_when_no_objects():
    """没有物体层时应自动退回体素层（而不是返回空）。"""
    smap = _make_map([("cup", [0, 0, 0]), ("table", [1, 0, 0])])
    assert smap.num_objects == 0
    results = smap.query_text("杯子", top_k=2, level="auto")
    assert len(results) > 0
    assert results[0].level == "voxel"
    assert np.allclose(results[0].position, [0.05, 0.05, 0.05])


def test_map_query_min_score_filters_irrelevant():
    smap = _make_map([("cup", [0, 0, 0])])
    assert smap.query_text("宇宙飞船") == []          # 不相关 → 被分数下限过滤
    assert len(smap.query_text("杯子")) > 0


def test_query_engine_reports_mode():
    smap = _make_map([("cup", [0, 0, 0])])
    engine = QueryEngine(smap)
    assert engine.mode == "lexical"
    engine2 = QueryEngine(smap, text_encoder=_FakeTextEncoder())
    assert engine2.mode == "hybrid"


def test_query_engine_empty_text():
    smap = _make_map([("cup", [0, 0, 0])])
    engine = QueryEngine(smap)
    assert engine.query("") == []
    assert engine.query("   ") == []


# ==========================================================================
# 回归锁：词法阈值的"拒识"行为
# ==========================================================================
def test_lexical_threshold_rejects_ngram_noise():
    """查询地图里**没有的类别**必须返回空，不能被字符 n-gram 噪声蒙到。

    回归背景：词法第三层打分是字符 bigram Jaccard（上限 0.80），
    任意两词共享少量 bigram 就有分（`bicycle` 与 `bottle` 共享 `le`，
    Jaccard 0.09 → 0.145 分）。默认阈值曾是 0.05，于是这些噪声全被接受 ——
    实测 72 条"未见类别"查询里有 27.8% 返回了错误物体。
    阈值抬到 0.5 后拒识率 100%，命名类命中率不降。
    """
    smap = _make_map([("bottle", [0, 0, 0]), ("box", [1, 0, 0]), ("lamp", [0, 1, 0])])
    for absent in ("bicycle", "helicopter", "airplane", "冰箱", "宇宙飞船"):
        assert smap.query_text(absent) == [], f"{absent!r} 不应命中任何物体"


def test_lexical_threshold_keeps_legitimate_hits():
    """抬阈值不能误杀真命中：概念命中（1.0）与子串命中（0.85）都要留下。

    这两个分数是词法匹配的**真实信号**，0.5 的阈值必须高于噪声、低于它们。
    """
    smap = _make_map([("cup", [0, 0, 0])])
    assert len(smap.query_text("杯子")) > 0          # 别名表概念命中 → 1.0
    assert len(smap.query_text("cup")) > 0           # 完全相同 → 1.0
    assert len(smap.query_text("cups")) > 0          # 子串包含 → 0.85


def test_lexical_threshold_is_consistent_between_map_and_engine():
    """地图与引擎的词法阈值默认值必须一致，否则调用路径不同行为就不同。"""
    smap = _make_map([("cup", [0, 0, 0])])
    assert smap.query_min_score == QueryEngine(smap).min_score_lexical


def test_embedding_threshold_depends_on_mode():
    """嵌入阈值必须**随角色切换**：纯嵌入用"接受阈值"，hybrid 用"精度过滤器"。

    回归背景：曾经两种情况共用一个值（SigLIP 的 0.02）。但 0.02 是在
    **检测区域特征**上标定的，而查询比对的是**多视角融合的地图物体特征**，
    尺度小约 20 倍；更关键的是角色不同 —— 纯嵌入时它是唯一的接受判定。
    实测两种角色的最优阈值差 40 倍，用错会让纯嵌入的描述类命中率从
    12.5% 掉到 2.5%。
    """
    smap = _make_map([("cup", [0, 0, 0])])
    enc = _FakeTextEncoder()

    solo = QueryEngine(smap, text_encoder=enc, lexical=False)
    hybrid = QueryEngine(smap, text_encoder=enc, lexical=True)

    assert solo.mode == "embedding" and hybrid.mode == "hybrid"
    # 纯嵌入是唯一的接受判定 → 阈值必须更宽松
    assert solo.min_score_embedding < hybrid.min_score_embedding
    assert solo.embedding_threshold_is_standalone is True
    assert hybrid.embedding_threshold_is_standalone is False


def test_explicit_map_threshold_overrides_mode_adaptation():
    """显式在地图上设了阈值时，**不允许**被自适应逻辑覆盖（用户说了算）。"""
    smap = _make_map([("cup", [0, 0, 0])])
    smap.query_min_score_embedding = 0.123
    for lexical in (True, False):
        eng = QueryEngine(smap, text_encoder=_FakeTextEncoder(), lexical=lexical)
        assert eng.min_score_embedding == pytest.approx(0.123)
        assert eng.embedding_threshold_is_standalone is False
        assert eng.embedding_threshold_source == "explicit"


# ==========================================================================
# 自监督阈值标定（用地图自身的 (特征, 标签) 对就地定阈值）
# ==========================================================================
def _make_map_with_features(entries):
    """构造带**物体层**和**不同特征**的语义地图：entries = [(label, center, feature), ...]。

    普通 `_make_map` 给所有物体同一个特征、且不建物体层，无法构造出
    可分的正负样本，所以自监督标定相关的测试需要这个版本。
    """
    grid = VoxelGrid(voxel_size=0.1, feature_dim=3)
    objects = []
    for i, (label, center, feat) in enumerate(entries):
        center = np.asarray(center, dtype=np.float64)
        feat = np.asarray(feat, dtype=np.float32)
        grid.add(center[None, :], feat[None, :], labels=[label], frame_id=f"f{i}")
        objects.append(SemanticObject(
            obj_id=i, label=label, center=center,
            bbox_min=center - 0.05, bbox_max=center + 0.05,
            feature=feat, num_voxels=1, num_points=1, confidence=1.0,
        ))
    return SemanticMap(grid, objects=objects)


_SEPARABLE = [
    ("cup", [0, 0, 0], [1.0, 0.0, 0.0]),
    ("table", [1, 0, 0], [0.0, 1.0, 0.0]),
    ("chair", [0, 1, 0], [0.0, 0.0, 1.0]),
]


def test_standalone_threshold_is_self_calibrated():
    """纯嵌入模式下，阈值应来自**自监督标定**，而不是硬编码常量。

    标定集就是地图自己：3 个物体对各自的标签得高分、对另外两个标签得低分，
    因此正样本 3 个、负样本 6 个。
    """
    smap = _make_map_with_features(_SEPARABLE)
    eng = QueryEngine(smap, text_encoder=_FakeTextEncoder(), lexical=False)

    assert eng.mode == "embedding"
    assert eng.embedding_threshold_source == "auto"
    assert eng.min_score_embedding > 0
    assert eng.calibration_debug["n_pos"] == 3
    assert eng.calibration_debug["n_neg"] == 6
    # 分离度上界是 2.0（TPR+TNR 满分）
    assert 0.0 < eng.calibration_debug["balanced_accuracy"] <= 2.0


def test_self_calibration_separates_own_label_and_rejects_absent():
    """标定出的阈值必须**同时**做到两件事：认得自己、并且拒得掉没有的类别。

    这是标定是否有意义的判据 —— 只做到一件（比如全接受或全拒绝）都是废的。
    """
    smap = _make_map_with_features(_SEPARABLE)
    eng = QueryEngine(smap, text_encoder=_FakeTextEncoder(), lexical=False)

    for label in ("cup", "table", "chair"):
        hits = eng.query(label, top_k=1)
        assert hits, f"{label} 应命中"
        assert hits[0].label == label

    # 地图里没有的类别必须返回空（假编码器对未知词返回"空文本"方向，
    # 与任一物体都没有优势 → 应被标定出的阈值挡住）
    assert eng.query("refrigerator", top_k=1) == []


def test_calibration_falls_back_when_only_one_label():
    """只有一个类别时构造不出负样本 → 必须**回退**到编码器声明值，而不是崩掉。"""
    smap = _make_map([("cup", [0, 0, 0])])
    eng = QueryEngine(smap, text_encoder=_FakeTextEncoder(), lexical=False)

    assert eng.embedding_threshold_source == "encoder"
    assert eng.min_score_embedding == pytest.approx(
        _FakeTextEncoder.suggested_standalone_threshold)
    assert eng.calibration_debug == {}


def test_hybrid_does_not_use_standalone_calibration():
    """hybrid 模式下嵌入阈值是"精度过滤器"，**不应**用独立模式的自标定值。

    两种角色最优值差 40 倍；把独立模式的宽松阈值用到 hybrid 上会引入假接受。
    """
    smap = _make_map_with_features(_SEPARABLE)
    eng = QueryEngine(smap, text_encoder=_FakeTextEncoder(), lexical=True)

    assert eng.mode == "hybrid"
    assert eng.embedding_threshold_source == "encoder"
    assert eng.min_score_embedding == pytest.approx(
        _FakeTextEncoder.suggested_pair_threshold)


def test_calibration_requires_at_least_two_objects():
    """物体少于 2 个时无法标定 —— 必须安全回退而不是抛异常。"""
    smap = _make_map_with_features([("cup", [0, 0, 0], [1.0, 0.0, 0.0])])
    eng = QueryEngine(smap, text_encoder=_FakeTextEncoder(), lexical=False)
    assert eng.embedding_threshold_source == "encoder"


def test_lexical_threshold_is_not_auto_calibrated():
    """★ **否定实验的回归锁**：词法阈值**不要**接自监督标定，保持人工常量 0.5。

    看起来把嵌入侧那套标定推广到词法侧很自然（同一套 (标签, 标签) 打分），
    实测**会变差**（综合分 0.828 → 0.742，`scripts/16` 第五节）：

    - 词法标定集是「标签查自己」，别名表必然命中 1.0、对别的≈0，
      于是正负样本**完全可分**（平衡准确率 = 2.0），优化器把阈值推到
      正样本分布的最边缘（≈0.90~0.95）；
    - 但真实查询（「杯子」「seat」）常只落在**子串那层（0.85）**，全被误杀。

    对照：嵌入标定集的正负分布本就重叠（BA≈1.4），阈值落在有意义的间隙里，
    所以泛化得好（综合分 0.232 → 0.291）。

    推广的判据：**自监督标定只在「标定集难度 ≈ 部署难度」时才有效；
    BA 接近 2.0 就是「标定集太简单」的直接证据，不是成功信号。**
    """
    smap = _make_map([("cup", [0, 0, 0]), ("table", [1, 0, 0])])
    eng = QueryEngine(smap)

    # 阈值必须停在人工常量上，而不是被标定推到 ~0.9
    assert eng.min_score_lexical == pytest.approx(0.5)
    # 行为层面：子串命中（0.85）必须仍然有效 —— 这正是标定推到 0.9 会误杀的
    assert len(smap.query_text("cups")) > 0
    assert len(smap.query_text("桌子")) > 0
