"""数据集级多模态语料工程（大规模多模态数据处理）。

模块分工
--------
| 模块 | 职责 | 对应面试问题 |
|---|---|---|
| `schema` | 数据契约（`VideoRecord` / `CaptionRecord`） | — |
| `sources` | 异构数据源适配（MSR-VTT / ActivityNet / 合成） | 数据采集 |
| `caption` | 字幕清洗、三级去重、统计、跨模态对齐 | **Video/Image Caption 基建难点** |
| `runner` | 规模跑批 + 漏斗 + 吞吐 + 断点续跑 | **视频怎么切帧 / 数据清洗** |
| `packing` | 温度配比 + token 预算 + 分片打包 | 数据配比 |
| `goldset` | 真实视频上的镜头检测金种子核验 | **镜头检测 + 三等分采样** |
| `datacard` | 数据卡 | 数据可审查性 |

与 `data/video/` 的关系
-----------------------
`data/video/` 是**单视频**管线（镜头→抽帧→去重→质量），本包是**数据集**层，
复用前者而不是重写。分层的判据很简单：
**"这段逻辑需不需要看到全量数据？"** —— 不需要的放 `data/video/`，
需要的（跨视频去重、配比、数据卡）放这里。
"""

from __future__ import annotations

from roboground.data.corpus.caption import (
    CaptionCleanConfig, DedupConfig, caption_stats, clean_caption, clean_captions,
    dedup_captions, lsh_bands, minhash_signature, normalize_text, score_alignment,
)
from roboground.data.corpus.datacard import build_datacard
from roboground.data.corpus.goldset import (
    GoldSetConfig, evaluate_goldset, load_goldset, make_strip,
    match_boundaries, render_verification_material, save_goldset,
)
from roboground.data.corpus.packing import (
    MixtureConfig, PackConfig, TrainingSample, duration_bucket, iter_samples,
    pack_shards, plan_mixture, token_budget_report,
)
from roboground.data.corpus.runner import (
    CorpusRunConfig, CorpusRunResult, run_corpus,
)
from roboground.data.corpus.schema import CaptionRecord, VideoRecord
from roboground.data.corpus.sources import (
    ActivityNetCaptionSource, ActivityNetConfig, CorpusSource,
    MSRVTTVideoSource, MSRVTTConfig, SyntheticCorpusConfig, SyntheticVideoSource,
    build_source,
)

__all__ = [
    # schema
    "VideoRecord", "CaptionRecord",
    # sources
    "CorpusSource", "MSRVTTVideoSource", "MSRVTTConfig",
    "ActivityNetCaptionSource", "ActivityNetConfig",
    "SyntheticVideoSource", "SyntheticCorpusConfig", "build_source",
    # caption
    "CaptionCleanConfig", "DedupConfig", "normalize_text", "clean_caption",
    "clean_captions", "dedup_captions", "minhash_signature", "lsh_bands",
    "caption_stats", "score_alignment",
    # runner
    "CorpusRunConfig", "CorpusRunResult", "run_corpus",
    # packing
    "TrainingSample", "MixtureConfig", "PackConfig", "duration_bucket",
    "iter_samples", "plan_mixture", "token_budget_report", "pack_shards",
    # datacard
    "build_datacard",
    # goldset
    "GoldSetConfig", "make_strip", "render_verification_material",
    "save_goldset", "load_goldset", "match_boundaries", "evaluate_goldset",
]
