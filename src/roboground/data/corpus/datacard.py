"""数据卡（Data Card）：把跑批结果写成一份"可被审查"的文档。

为什么数据卡不是"顺手生成的 README"
==================================
数据卡的价值在于**把不可见的数据决策变成可审查的断言**。一份合格的数据卡
必须能回答下面这些问题，而且**答案要带数字**：

1. 这份语料有多大？（不是"10k 视频"，而是时长分布、帧数、token 数）
2. 每一级闸门砍掉了多少？**砍掉的是哪一类**？
3. 有没有失败被吞掉？（失败率 + 归因直方图）
4. 字幕长什么样？有没有模板污染？
5. 配比之后，各类的**重复倍数**是多少？（重复才是过拟合的来源）
6. **这份数据不能用来做什么**？（局限必须写，否则下游会误用）

第 6 条最关键，也最常被省略。本项目在 `AGENTS.md` 里反复强调
"诚实清单"，数据卡同样遵循这个原则：**没做的事要显式写"没做"**，
而不是留空让人以为做过了。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from roboground.data.corpus.runner import CorpusRunResult


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


_CN_NUM = "一二三四五六七八九十"


class _SectionCounter:
    """小节编号器。

    ⚠️ 为什么不用硬编码的"一、二、三"：有几个小节是**条件出现**的
    （跨模态对齐、配比）。硬编码会在跳过它们时留下断号（"五、七、八"），
    一眼就能看出这份数据卡是拼出来的、不可信。
    """

    def __init__(self) -> None:
        self.i = 0

    def __call__(self, title: str) -> str:
        self.i += 1
        num = _CN_NUM[self.i - 1] if self.i <= len(_CN_NUM) else str(self.i)
        return f"## {num}、{title}"


def _table(rows: list[tuple[str, Any]], header: tuple[str, str]) -> str:
    out = [f"| {header[0]} | {header[1]} |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def build_datacard(
    result: CorpusRunResult,
    *,
    source_desc: Dict[str, Any],
    run_cfg: Dict[str, Any],
    mixture: Optional[Dict[str, Any]] = None,
    alignment: Optional[Dict[str, Any]] = None,
    title: str = "多模态语料数据卡",
) -> str:
    """生成 Markdown 数据卡。"""
    f, t = result.funnel, result.throughput
    cc, cd, cs = result.caption_clean, result.caption_dedup, result.caption_stats

    sec = _SectionCounter()
    L: list[str] = []
    L.append(f"# {title}")
    L.append("")
    L.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　"
             f"｜　来源：`{source_desc.get('source', '?')}`")
    L.append("")
    L.append(sec("规模总览"))
    L.append("")
    L.append(_table([
        ("视频条数", f"{f['n_videos']:,}（成功 {f['n_ok']:,} / 失败 {f['n_failed']:,}）"),
        ("总时长", f"{f['total_duration_s'] / 60:.1f} 分钟"
                   f"（中位 {f['median_duration_s']:.1f}s / 条）"),
        ("解码帧数", f"{f['n_frames_decoded']:,}"),
        ("保留帧数", f"{f['n_after_quality']:,}（整体保留率 {_pct(f['overall_keep_rate'])}）"),
        ("镜头数", f"{f['n_shots']:,}（均值 {f['shots_per_video_mean']:.1f}/视频）"),
        ("字幕条数", f"{cc['n_in']:,} → 清洗后 {cc['n_kept']:,} → 去重后 {cd['n_kept']:,}"),
        ("字幕 token 总量", f"{cs['total_tokens']:,}"),
    ], ("指标", "值")))
    L.append("")

    L.append(sec("逐阶段漏斗（每一级砍掉了多少）"))
    L.append("")
    L.append(_table([
        ("① 解码", f"{f['n_frames_decoded']:,} 帧"),
        ("② 抽帧", f"{f['n_picked']:,} 帧（抽取率 {_pct(f['pick_rate'])}）"),
        ("③ 去重", f"{f['n_after_dedup']:,} 帧（保留 {_pct(f['dedup_keep_rate'])}）"),
        ("④ 质量闸门", f"{f['n_after_quality']:,} 帧（保留 {_pct(f['quality_keep_rate'])}）"),
        ("整体", f"{_pct(f['overall_keep_rate'])} 的解码帧最终进入训练"),
    ], ("阶段", "结果")))
    L.append("")
    if f["quality_drop_reasons"]:
        L.append("**质量闸门淘汰归因**：" + "、".join(
            f"`{k}` {v:,}" for k, v in f["quality_drop_reasons"].items()))
    else:
        L.append("**质量闸门淘汰归因**：无淘汰（阈值可能偏松，值得复核）")
    L.append("")

    L.append(sec("吞吐（决定迭代速度）"))
    L.append("")
    L.append(_table([
        ("视频处理", f"{t['videos_per_min']:.1f} 视频/分钟"
                     f"（{t['seconds_per_video']:.2f} s/条，{t['workers']} 线程）"),
        ("帧吞吐", f"{t['frames_per_sec']:.0f} 帧/秒"),
        ("数据吞吐", f"{t['mb_per_sec']:.2f} MB/s（按解码 RGB 量估算）"),
        ("阶段耗时", f"视频管线 {result.stage_seconds.get('video_pipeline', 0)}s、"
                     f"字幕清洗+去重 {result.stage_seconds.get('caption_clean_dedup', 0)}s"),
        ("总墙钟", f"{t['wall_seconds']:.1f}s"),
    ], ("维度", "实测")))
    L.append("")

    L.append(sec("失败归因（必须为 0 容忍，否则指标被悄悄美化）"))
    L.append("")
    if result.failures:
        L.append(_table(list(result.failures.items()), ("失败类型", "条数")))
    else:
        L.append("无失败记录。")
    L.append("")

    L.append(sec("字幕质量"))
    L.append("")
    L.append("**清洗淘汰归因**：" + ("、".join(
        f"`{k}` {v:,}" for k, v in cc["drop_reasons"].items()) or "无淘汰"))
    L.append("")
    L.append("**去重分项**：" + "、".join([
        f"精确重复 {cd['dropped_exact']:,}",
        f"词集重复 {cd['dropped_wordset']:,}",
        f"近重复 {cd['dropped_near']:,}",
        f"跨视频模板 {cd['dropped_cross_video']:,}",
    ]))
    L.append("")
    L.append(_table([
        ("唯一文本数", f"{cd['n_unique_texts']:,}"),
        ("跨视频高频模板数", f"{cd['n_cross_video_templates']:,}"
                            f"（出现在 ≥{cd.get('cross_video_min_videos', 3)} 个视频上）"),
        ("平均长度", f"{cs['mean_tokens']:.1f} tokens"
                     f"（P10 {cs['p10_tokens']:.0f} / 中位 {cs['median_tokens']:.0f}"
                     f" / P90 {cs['p90_tokens']:.0f}）"),
        ("词表大小", f"{cs['vocab_size']:,}"),
        ("Type-Token Ratio", f"{cs['type_token_ratio']:.4f}"),
        ("每视频字幕数", f"均值 {cs['captions_per_video_mean']:.1f}"
                         f"（{cs['captions_per_video_min']}~{cs['captions_per_video_max']}）"),
    ], ("指标", "值")))
    L.append("")
    if cd.get("top_cross_video"):
        L.append("**跨视频高频字幕 Top-5**（模板污染候选，值得人工看一眼）：")
        L.append("")
        for text, n in cd["top_cross_video"][:5]:
            L.append(f"- `{text}` — 出现在 **{n}** 个不同视频")
        L.append("")

    if alignment is not None:
        L.append(sec("跨模态对齐（图文是否匹配）"))
        L.append("")
        if alignment.get("performed"):
            L.append(_table([
                ("样本数", alignment["n_pairs"]),
                ("相似度均值", f"{alignment['mean']:.4f}"),
                ("中位数", f"{alignment['median']:.4f}"),
                ("P10 / P90", f"{alignment['p10']:.4f} / {alignment['p90']:.4f}"),
                ("低于阈值", f"{alignment['below_threshold']}（阈值 {alignment['threshold']}）"),
            ], ("指标", "值")))
        else:
            L.append(f"⚠️ **未执行**：{alignment.get('reason', '')}")
        L.append("")

    if mixture is not None:
        L.append(sec("配比"))
        L.append("")
        L.append(f"分组维度 `{mixture['group_key']}`，温度 α={mixture['temperature']}；"
                 f"可用 {mixture['n_available']:,} 条 → 选中 {mixture['n_selected']:,} 条。")
        L.append("")
        rows = []
        for g in list(mixture["natural_ratio"])[:12]:
            rows.append((
                g,
                f"{mixture['natural_counts'].get(g, 0):,}",
                _pct(mixture["natural_ratio"].get(g, 0)),
                _pct(mixture["target_ratio"].get(g, 0)),
                _pct(mixture["actual_ratio"].get(g, 0)),
                f"{mixture['reuse_factor'].get(g, 0):.2f}×",
            ))
        L.append("| 分组 | 自然条数 | 自然占比 | 目标占比 | 实际占比 | 重复倍数 |")
        L.append("|---|---|---|---|---|---|")
        for r in rows:
            L.append("| " + " | ".join(str(x) for x in r) + " |")
        L.append("")
        L.append("> **重复倍数 > 1 表示该组被有放回重复采样**。"
                 "小类被重复是拉平长尾的必然代价，但倍数过高（>3×）会过拟合，"
                 "所以 `max_reuse_factor` 是硬约束。")
        L.append("")

    L.append(sec("已知局限（诚实清单）"))
    L.append("")
    L.append(_limitations(source_desc, alignment, cd))
    L.append("")

    L.append(sec("配置快照（可复现）"))
    L.append("")
    L.append("```json")
    import json
    L.append(json.dumps({"source": source_desc, "run": run_cfg},
                        ensure_ascii=False, indent=2))
    L.append("```")
    L.append("")
    return "\n".join(L)


def _limitations(source_desc: Dict[str, Any],
                 alignment: Optional[Dict[str, Any]],
                 dedup: Dict[str, Any]) -> str:
    """局限清单。**没做的事显式写"没做"**。"""
    items: list[str] = []
    src = source_desc.get("source")
    if src == "msrvtt":
        items.append(
            "**字幕是整段描述，没有片内时间定位** —— MSR-VTT 的 20 条字幕共享"
            "同一个时间区间，所以本语料**不能**用于时序定位（temporal grounding）训练。"
            "要训时序任务必须换 ActivityNet Captions 这类带 `timestamps` 的数据。")
        items.append(
            "**类别分布不均衡**（最大类与最小类相差约 7 倍），"
            "所以配比是必需的而不是可选的。")
    if alignment is None or not alignment.get("performed"):
        items.append(
            "**跨模态对齐打分未执行** —— 缺少图文编码器时『字幕与画面无关』"
            "这一类噪声**完全没有被覆盖**。这是本语料目前最大的未知项。")
    if dedup.get("dropped_near", 0) == 0:
        items.append(
            "**近重复去重没有实际生效**（命中 0 条）—— 要么数据确实干净，"
            "要么 MinHash/LSH 参数（`n_perm` / `band_size`）召回不足，需要复核。")
    items.append(
        "**抽帧结果未做人工核验抽样**（除镜头检测金种子外）—— "
        "『抽到的帧是否覆盖了字幕描述的内容』没有量化。")
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(items))


__all__ = ["build_datacard"]
