"""数据集级多模态语料的数据契约。

为什么要在 `data/video/` 之上再加一层
=====================================
`data/video/` 解决的是**单条视频怎么处理**（镜头 → 抽帧 → 去重 → 质量），
它的输入是"一段帧序列"、输出是"一段保留帧"。

但真实的数据工程问题不在单条视频，而在**数据集规模**上：

1. **来源异构** —— MSR-VTT 是 10~30 秒的 web 短片 + 20 条字幕，
   ActivityNet 是 2 分钟的 YouTube 长视频 + 稠密时序标注，
   自采数据可能是几十秒的机器人第一视角。它们的字段、字幕形态、
   时间语义都不一样，**必须先归一化到同一个契约**，下游才能统一处理。
2. **字幕是二等公民** —— 单视频管线只关心帧，但真正决定多模态数据质量的是
   **帧和文本对不对得上**。所以字幕必须和视频一样进入质量闸门。
3. **要能只处理一部分** —— 10k 条视频不可能每次全跑一遍，
   必须支持"按 split / 按 shard / 按抽样"流式读取，且**断点续跑**。

所以本模块的定位是：**把 N 条异构视频 + M 条字幕，收敛成一份可统计、可配比、
可打包的训练语料**，且每一步都有量化依据。

约定
----
- 视频一律用 `VideoRecord` 表示，字幕一律用 `CaptionRecord` 表示；
- 两者都是**可变**的 dataclass（处理过程中会回填 `stats` / `cleaned` 等字段），
  因为数据工程的中间态就是"逐步补全一条记录"，用不可变对象反而到处要 copy。
- `meta` 保留**原始字段**，不丢信息 —— 事后追查"这条数据为什么被丢"必须能回溯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: 支持的语料来源标识
SOURCE_NAMES = ("msrvtt", "activitynet", "synthetic")


# =============================================================================
# 字幕
# =============================================================================
@dataclass
class CaptionRecord:
    """一条视频-文本对（或视频片段-文本对）。

    Attributes
    ----------
    text
        原始字幕文本。
    video_id
        所属视频。
    source
        来源标识（`msrvtt` / `activitynet` / `synthetic`）。
    start, end
        该字幕对应的时间区间（秒）。整段视频共用一条字幕时为 `(0.0, duration)`。
        ActivityNet 的稠密标注靠这两个字段区分片段，MSR-VTT 恒为整段。
    cleaned
        清洗后的文本；为空表示"还没清洗或已被丢弃"。
    tokens
        清洗后文本的 token 数（按空白切分，供 token 预算核算）。
    drop_reason
        被丢弃的原因（空字符串表示保留）。**保留原因而不是直接删掉记录** ——
        这样才能统计"哪一类问题杀掉了最多数据"。
    """

    text: str
    video_id: str
    source: str = "msrvtt"
    start: float = 0.0
    end: float = 0.0
    cleaned: str = ""
    tokens: int = 0
    drop_reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def kept(self) -> bool:
        return not self.drop_reason

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        tag = "keep" if self.kept else f"drop:{self.drop_reason}"
        return f"<Caption {self.video_id} {tag} {self.text[:32]!r}>"


# =============================================================================
# 视频
# =============================================================================
@dataclass
class VideoRecord:
    """一条视频及其全部字幕、元信息与处理结果。

    Attributes
    ----------
    video_id
        数据集内的唯一标识。
    source
        来源标识。
    split
        `train` / `val` / `test` / `all`。
    path
        视频文件路径。为 None 表示只有元数据（尚未下载）——
        这个状态是**故意允许**的：可以先在元数据上跑配比统计，再补视频。
    captions
        该视频的全部字幕。
    meta
        原始元信息（时长、fps、类别、URL 等），不做任何删减。
    stats
        管线回填的处理统计（镜头数、保留帧数、吞吐等）。
    error
        处理失败的原因（空表示成功）。**失败必须被记录而不是静默跳过** ——
        否则"处理了 10000 条"里藏着的解码失败会污染所有下游指标。
    """

    video_id: str
    source: str
    split: str = "all"
    path: Optional[Path] = None
    captions: List[CaptionRecord] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    # ---------------- 便捷视图 ----------------
    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def duration(self) -> float:
        """视频时长（秒）。元数据缺失时回退到字幕区间的最大值。"""
        d = self.meta.get("duration")
        if isinstance(d, (int, float)) and d > 0:
            return float(d)
        return max((c.end for c in self.captions), default=0.0)

    @property
    def n_captions(self) -> int:
        return len(self.captions)

    @property
    def kept_captions(self) -> List[CaptionRecord]:
        return [c for c in self.captions if c.kept]

    @property
    def n_frames_kept(self) -> int:
        return int(self.stats.get("n_after_quality", 0))

    def to_dict(self, *, include_captions: bool = True) -> Dict[str, Any]:
        """序列化（写 checkpoint / 数据卡用）。"""
        d: Dict[str, Any] = {
            "video_id": self.video_id,
            "source": self.source,
            "split": self.split,
            "path": str(self.path) if self.path else None,
            "duration": round(self.duration, 3),
            "error": self.error,
            "stats": self.stats,
        }
        if include_captions:
            d["captions"] = [
                {"text": c.text, "cleaned": c.cleaned, "tokens": c.tokens,
                 "start": c.start, "end": c.end, "drop_reason": c.drop_reason}
                for c in self.captions
            ]
        return d

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        tag = "ok" if self.ok else f"ERR:{self.error}"
        return (f"<Video {self.video_id} {self.source}/{self.split} "
                f"{self.n_captions}cap {tag}>")


__all__ = ["CaptionRecord", "VideoRecord", "SOURCE_NAMES"]
