"""数据集级多模态语料工程测试。

回归锁清单（每条都对应一个**踩过或极易踩**的具体问题）
------------------------------------------------------
- `test_minhash_is_stable_across_processes` —— 用内置 `hash()` 会让签名
  跨进程不可复现（PYTHONHASHSEED 随机盐），断点续跑结果不一致。
- `test_dedup_cross_video_template_separate_from_in_video` ——
  "同一视频内重复"和"跨视频模板"是两类不同的问题，混在一起统计会误判。
- `test_match_boundaries_is_one_to_one` —— 多对一会让 P/R/F1 虚高
  （一个真值旁边挤 5 个预测 = 5 个 TP），必须一对一。
- `test_alignment_absent_encoder_is_explicit` —— 没编码器时必须显式返回
  "没做"，而不是空 dict 让人以为通过了。
- `test_mixture_temperature_flattens_long_tail` —— 温度采样的**作用**要有断言，
  不只是"跑通不报错"。
- `test_mixture_respects_max_reuse` —— 小类被过度重复是过拟合的直接来源。
- `test_funnel_excludes_failed_videos_from_means` —— 失败视频按 0 帧计入
  会拉低均值、把注意力引向错误方向。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from roboground.data.corpus import (
    CaptionCleanConfig, CaptionRecord, CorpusRunConfig, MixtureConfig,
    PackConfig, TrainingSample, VideoRecord, build_source, caption_stats,
    clean_caption, clean_captions, dedup_captions, duration_bucket,
    evaluate_goldset, iter_samples, load_goldset, match_boundaries,
    pack_shards, plan_mixture, score_alignment, token_budget_report,
)
from roboground.data.corpus.caption import (
    minhash_signature, normalize_text,
)
from roboground.data.corpus.goldset import GoldSetConfig, make_strip, save_goldset
from roboground.data.corpus.runner import _build_funnel, _failure_histogram


# ==========================================================================
# 文本归一化与清洗
# ==========================================================================
def test_normalize_text_handles_entities_and_fullwidth():
    """HTML 实体、全角字符、重复标点都要被归一化掉。

    不归一化的话，"看起来一样"的两条文本会在精确去重里被判为不同，
    于是模板污染逃过 L1 —— 这是最隐蔽的一类漏网。
    """
    assert normalize_text("  Hello   World ") == "Hello World"
    assert normalize_text("a &amp; b") == "a & b"
    assert normalize_text("<b>cat</b>") == "cat"
    assert normalize_text("cat!!!") == "cat!"
    assert normalize_text("ＡＢＣ") == "ABC"          # 全角 → 半角
    assert normalize_text("cat\u200b") == "cat"       # 零宽空格


@pytest.mark.parametrize("text,reason", [
    ("", "too_short"),
    ("a cat", "too_short"),
    ("x " * 80, "too_long"),
    ("http://a.com/x", "url"),
    ("...!!!", "too_short"),
    ("日本語 の テキスト", "non_ascii"),
    ("very very very very very very", "degenerate_repeat"),
])
def test_clean_caption_drop_reasons(text, reason):
    """每条闸门都要能被触发，且原因可区分（数据卡的淘汰归因靠它）。"""
    _, why = clean_caption(text)
    assert why == reason, f"{text!r} 期望 {reason}，实际 {why!r}"


def test_clean_caption_keeps_good_text():
    cleaned, why = clean_caption("a band is performing on the stage")
    assert why == ""
    assert cleaned == "a band is performing on the stage"


def test_clean_captions_records_stats_in_place():
    recs = [
        CaptionRecord("a dog runs in the park", "v1"),
        CaptionRecord("cat", "v1"),
        CaptionRecord("http://x.com", "v2"),
    ]
    stats = clean_captions(recs)
    assert stats["n_in"] == 3 and stats["n_kept"] == 1
    assert stats["drop_reasons"]["too_short"] == 1
    assert recs[0].tokens == 6 and recs[0].kept
    assert not recs[1].kept


# ==========================================================================
# MinHash / 去重
# ==========================================================================
def test_minhash_signature_deterministic():
    a = minhash_signature([1, 2, 3], n_perm=16, seed=0)
    b = minhash_signature([1, 2, 3], n_perm=16, seed=0)
    c = minhash_signature([1, 2, 3], n_perm=16, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)      # 不同 seed 应给出不同签名


def test_minhash_is_stable_across_processes():
    """跨进程可复现 —— 断点续跑的前提。

    如果实现里用了内置 `hash()`，Python 对 str 的哈希带随机盐，
    两个进程算出的签名不同，于是"同一条数据"在不同 run 里被判为不同，
    去重结果不可复现。这条测试用两个独立进程比对签名来锁住这个性质。
    """
    code = (
        "import sys; sys.path.insert(0,'src');"
        "from roboground.data.corpus.caption import minhash_signature;"
        "print(list(minhash_signature([7,11,13], n_perm=8, seed=0)))"
    )
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=str(Path(__file__).resolve().parents[1]))
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], "MinHash 签名跨进程不一致（用了不稳定的 hash？）"


def test_dedup_exact_and_wordset():
    recs = [
        CaptionRecord("a dog runs fast", "v1"),
        CaptionRecord("a dog runs fast", "v1"),        # 精确重复
        CaptionRecord("fast a dog runs", "v1"),        # 词集相同
        CaptionRecord("a cat sleeps", "v1"),
    ]
    clean_captions(recs)
    stats = dedup_captions(recs)
    assert stats["dropped_exact"] == 1
    assert stats["dropped_wordset"] == 1
    assert stats["n_kept"] == 2


def test_dedup_cross_video_template_separate_from_in_video():
    """跨视频高频模板要单独统计，不能和"同视频内重复"混为一谈。

    两类问题的修法完全不同：同视频内重复 → 该视频字幕冗余；
    跨视频模板 → 整个语料被句式污染，要靠配比或换数据源解决。
    """
    recs = []
    for v in range(4):
        recs.append(CaptionRecord("a man is talking about something", f"v{v}"))
    clean_captions(recs)
    stats = dedup_captions(recs, scope="per_video")
    # per_video 作用域下不算重复（各视频只有一条），但跨视频模板应被抓出
    assert stats["dropped_exact"] == 0
    assert stats["dropped_cross_video"] == 4
    assert stats["n_cross_video_templates"] == 1
    assert stats["top_cross_video"][0][1] == 4


def test_dedup_near_duplicate_detected():
    """近重复（改几个词）必须被 MinHash+LSH 抓到。"""
    base = "a young man is playing the guitar on the stage in a small bar"
    recs = [CaptionRecord(base, "v1"),
            CaptionRecord(base + " tonight", "v2")]
    clean_captions(recs)
    stats = dedup_captions(recs)
    assert stats["n_kept"] == 1
    assert stats["dropped_near"] == 1


# ==========================================================================
# 统计
# ==========================================================================
def test_caption_stats_counts_tokens_and_vocab():
    recs = [CaptionRecord("a dog runs", "v1"), CaptionRecord("a cat runs", "v1")]
    clean_captions(recs)
    s = caption_stats(recs)
    assert s["n_kept"] == 2
    assert s["total_tokens"] == 6
    assert s["vocab_size"] == 4            # a, dog, runs, cat
    assert s["captions_per_video_mean"] == 2.0


def test_alignment_absent_encoder_is_explicit():
    """没编码器时必须**显式说明没做**，不能返回空 dict 让人误以为通过了。"""
    out = score_alignment([(np.zeros((4, 4, 3), np.uint8), "a cat")], None)
    assert out["performed"] is False
    assert "未提供图文编码器" in out["reason"]
    assert "未被覆盖" in out["reason"]


def test_alignment_with_stub_encoder():
    def fake(frames, texts):
        return np.linspace(0.1, 0.9, len(texts))

    out = score_alignment([(None, "a"), (None, "b"), (None, "c")],
                          fake, threshold=0.5)
    assert out["performed"] is True
    assert out["n_pairs"] == 3
    assert out["below_threshold"] == 1


# ==========================================================================
# 配比
# ==========================================================================
def _samples(counts: dict[str, int]) -> list[TrainingSample]:
    out = []
    for g, n in counts.items():
        for i in range(n):
            out.append(TrainingSample(
                video_id=f"{g}_{i}", source="t", caption=f"{g} sample {i}",
                tokens=10, duration=20.0, frames_kept=12, group=g))
    return out


def test_mixture_temperature_flattens_long_tail():
    """α<1 必须真的拉平长尾 —— 这是温度采样存在的唯一理由。"""
    data = _samples({"big": 900, "small": 100})
    _, natural = plan_mixture(data, MixtureConfig(temperature=1.0,
                                                  max_ratio_per_key=1.0))
    _, flat = plan_mixture(data, MixtureConfig(temperature=0.5,
                                               max_ratio_per_key=1.0))
    nat_gap = natural["natural_ratio"]["big"] - natural["natural_ratio"]["small"]
    flat_gap = flat["target_ratio"]["big"] - flat["target_ratio"]["small"]
    assert flat_gap < nat_gap, f"温度采样没有拉平长尾：{nat_gap} → {flat_gap}"


def test_mixture_respects_max_ratio_cap():
    data = _samples({"big": 900, "small": 100})
    _, stats = plan_mixture(data, MixtureConfig(temperature=1.0,
                                                max_ratio_per_key=0.6))
    assert max(stats["target_ratio"].values()) <= 0.6 + 1e-9


def test_mixture_respects_max_reuse():
    """小类被重复采样的倍数必须受 `max_reuse_factor` 硬约束（过拟合防线）。"""
    data = _samples({"big": 1000, "tiny": 2})
    _, stats = plan_mixture(data, MixtureConfig(
        temperature=0.0, max_ratio_per_key=0.5, max_reuse_factor=2.0,
        total_budget=200, seed=0))
    for g, f in stats["reuse_factor"].items():
        assert f <= 2.0 + 1e-6, f"{g} 重复 {f}× 超过上限"


def test_mixture_token_budget_truncates():
    data = _samples({"a": 500})
    sel, stats = plan_mixture(data, MixtureConfig(token_budget=1000))
    assert stats["total_tokens"] <= 1000
    assert len(sel) < 500


def test_token_budget_report():
    rep = token_budget_report(_samples({"a": 4}))
    assert rep["n_samples"] == 4
    assert rep["total_tokens"] == 40
    assert rep["mean_frames_kept"] == 12.0


def test_duration_bucket_edges():
    assert duration_bucket(5) == "<15s"
    assert duration_bucket(20) == "15-30s"
    assert duration_bucket(45) == "30-60s"
    assert duration_bucket(90) == "60-120s"
    assert duration_bucket(300) == ">120s"


# ==========================================================================
# 打包
# ==========================================================================
def test_pack_shards_writes_manifest(tmp_path):
    data = _samples({"a": 5})
    man = pack_shards(data, PackConfig(shard_size=2, out_dir=tmp_path / "out"))
    assert man["n_shards"] == 3
    assert man["n_samples"] == 5
    assert (tmp_path / "out" / "manifest.json").exists()
    first = (tmp_path / "out" / "shard-00000.jsonl").read_text(encoding="utf-8")
    assert len(first.strip().splitlines()) == 2
    assert json.loads(first.splitlines()[0])["group"] == "a"


# ==========================================================================
# 金种子 / 边界匹配
# ==========================================================================
def test_match_boundaries_is_one_to_one():
    """一对一匹配 —— 防止"多对一"把指标刷高。

    一个真值边界旁边挤 5 个预测，只能算 1 个 TP + 4 个 FP。
    如果实现成"附近有真值就算命中"，会得到 5 TP —— 这正是 DETR 用
    匈牙利匹配替掉 NMS 的同一类问题。
    """
    tp, fp, fn = match_boundaries([100, 101, 102, 103, 104], [100], tolerance=5)
    assert (tp, fp, fn) == (1, 4, 0)


def test_match_boundaries_tolerance():
    assert match_boundaries([103], [100], tolerance=2) == (0, 1, 1)
    assert match_boundaries([102], [100], tolerance=2) == (1, 0, 0)


def test_evaluate_goldset_reports_sample_size():
    """指标里必须带样本量 —— 否则 12 条视频的结论会被伪装成普适结论。"""
    pred = {"v1": [10, 50], "v2": [30]}
    gold = {"v1": [10, 50], "v2": [31]}
    m = evaluate_goldset(pred, gold, tolerance=2)
    assert m["n_videos_verified"] == 2
    assert m["tp"] == 3 and m["fp"] == 0 and m["fn"] == 0
    assert m["f1"] == 1.0
    assert "不应外推" in m["note"]


def test_goldset_roundtrip(tmp_path):
    p = tmp_path / "g.json"
    save_goldset({"v1": [1, 2]}, p)
    assert load_goldset(p) == {"v1": [1, 2]}


def test_make_strip_renders_with_marks():
    frames = [np.full((30, 40, 3), i * 8 % 255, np.uint8) for i in range(10)]
    img = make_strip(frames, list(range(10)), cfg=GoldSetConfig(cols=4),
                     title="t", mark=[3])
    assert img.ndim == 3 and img.shape[0] > 0 and img.shape[1] > 0


# ==========================================================================
# 数据源（元数据路径，不需要视频文件）
# ==========================================================================
def _write_msrvtt_anno(path: Path, n_videos: int = 3, n_caps: int = 2) -> Path:
    videos, sentences, sid = [], [], 0
    for k in range(n_videos):
        vid = f"video{k:04d}"
        videos.append({"video_id": vid, "id": k, "category": k % 20,
                       "url": "http://x", "start time": 0.0, "end time": 12.0,
                       "split": "test"})
        for j in range(n_caps):
            sentences.append({"caption": f"clip {k} caption {j}",
                              "video_id": vid, "sen_id": sid})
            sid += 1
    path.write_text(json.dumps({"info": {}, "videos": videos,
                                "sentences": sentences}), encoding="utf-8")
    return path


def test_msrvtt_source_metadata_only(tmp_path):
    """只有元数据（没有视频文件）时也要能建流 —— 这是"先配比后下载"的前提。"""
    anno = _write_msrvtt_anno(tmp_path / "a.json")
    src = build_source("msrvtt", cfg=__import__(
        "roboground.data.corpus", fromlist=["MSRVTTConfig"]
    ).MSRVTTConfig(anno=anno, video_dir=None))
    recs = list(src)
    assert len(recs) == 3
    assert recs[0].n_captions == 2
    assert recs[0].path is None
    assert recs[0].duration == 12.0
    assert src.describe()["n_captions"] == 6


def test_msrvtt_captions_use_full_span_not_source_clip(tmp_path):
    """⚠️ MSR-VTT 的 `start time/end time` 是**源片段**区间，不是字幕时间。

    把它当字幕时间会导致下游误用时序定位。这里锁住"字幕 start 恒为 0"。
    """
    anno = _write_msrvtt_anno(tmp_path / "a.json")
    from roboground.data.corpus import MSRVTTConfig
    src = build_source("msrvtt", cfg=MSRVTTConfig(anno=anno))
    rec = next(iter(src))
    assert all(c.start == 0.0 for c in rec.captions)
    assert all(c.end == rec.duration for c in rec.captions)
    assert rec.meta["source_clip"] == [0.0, 12.0]


# ==========================================================================
# 漏斗汇总
# ==========================================================================
def test_funnel_excludes_failed_videos_from_means():
    """失败视频不能按 0 帧计入均值 —— 否则像"质量闸门太严"，方向就带偏了。"""
    ok = VideoRecord("v1", "t", stats={"n_frames": 100, "n_shots": 5,
                                       "n_picked": 24, "n_after_dedup": 20,
                                       "n_after_quality": 18, "bytes_rgb": 1000})
    ok.meta["duration"] = 10.0
    bad = VideoRecord("v2", "t", error="missing_file")
    f = _build_funnel([ok, bad])
    assert f["n_videos"] == 2 and f["n_ok"] == 1 and f["n_failed"] == 1
    assert f["frames_kept_per_video_mean"] == 18.0     # 不是 9.0
    assert f["overall_keep_rate"] == pytest.approx(0.18)


def test_failure_histogram():
    recs = [VideoRecord("a", "t", error="missing_file"),
            VideoRecord("b", "t", error="missing_file"),
            VideoRecord("c", "t", error="decode_failed:ValueError"),
            VideoRecord("d", "t")]
    h = _failure_histogram(recs)
    assert h == {"missing_file": 2, "decode_failed:ValueError": 1}


# ==========================================================================
# 端到端（合成数据，离线）
# ==========================================================================
@pytest.mark.slow
def test_corpus_end_to_end_synthetic(tmp_path, quiet):
    """合成语料跑通：漏斗单调、保留率在 [0,1]、失败为 0。"""
    from roboground.data.corpus import SyntheticCorpusConfig, run_corpus

    src = build_source("synthetic", cfg=SyntheticCorpusConfig(
        n_videos=2, n_shots=4, frames_per_shot=8, width=64, height=48,
        cache_dir=tmp_path / "vid"))
    res = run_corpus(src, CorpusRunConfig(workers=1, target_frames=8,
                                          checkpoint=tmp_path / "ck.jsonl"))
    f = res.funnel
    assert f["n_ok"] == 2 and f["n_failed"] == 0
    assert 0.0 <= f["overall_keep_rate"] <= 1.0
    assert f["n_frames_decoded"] >= f["n_picked"] >= f["n_after_dedup"] >= f["n_after_quality"]
    assert res.throughput["frames_per_sec"] > 0
    # 断点文件应写入
    assert (tmp_path / "ck.jsonl").exists()
