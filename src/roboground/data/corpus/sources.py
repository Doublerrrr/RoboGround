"""异构数据源 → 统一 `VideoRecord` 流的适配层。

为什么需要"适配层"而不是"直接读 json"
=====================================
不同数据集的**字段语义完全不同**，如果让下游管线去 if/else 分支处理，
代码会迅速腐化。这里把差异**一次性吸收掉**，对外只暴露一个迭代器：

    for rec in source:          # 不管来源是 MSR-VTT / ActivityNet / 合成
        ...                     # 都是 VideoRecord，字段含义一致

三个来源的关键差异（这正是要吸收的东西）：

| 来源 | 视频形态 | 字幕形态 | 时间语义 |
|---|---|---|---|
| MSR-VTT | 10~30s web 短片 | 每视频 20 条，**整段共用** | `start/end` 恒为整段，无时序定位含义 |
| ActivityNet | ~2min YouTube 长片 | 每视频 3~5 条，**稠密时序** | `start/end` 是真实的片段区间 |
| 合成 | 内存帧序列 | 模板生成 | 无 |

⚠️ **MSR-VTT 的 `start time` / `end time` 是"这条视频从原 YouTube 视频里截取的区间"，
不是"字幕在片段内的时间位置"。** 这是个容易搞错的地方：有人会把它们当成
字幕的时间标注去做时序定位，那是错的 —— 同一视频的 20 条字幕共享同一个区间。
本模块显式把它归到 `meta["source_clip"]`，而 `CaptionRecord.start/end` 一律
填整段 `(0, duration)`，避免下游误用。

关于"视频还没下载"这个状态
==========================
`VideoRecord.path` 允许为 None。这不是缺陷而是**刻意的**：
数据工程里"先拿元数据做统计和配比、再决定下哪些视频"是常态
（10k 条视频全下要几十 GB，但配比统计只需要字幕和类别）。
"""

from __future__ import annotations

import json
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from roboground.data.corpus.schema import CaptionRecord, VideoRecord
from roboground.utils.logging import get_logger

logger = get_logger("data.corpus.sources")


# =============================================================================
# 抽象基类
# =============================================================================
class CorpusSource(ABC):
    """数据源抽象：一个可迭代的 `VideoRecord` 流 + 自描述信息。"""

    #: 来源标识（写进 `VideoRecord.source`）
    name: str = "unknown"

    @abstractmethod
    def __iter__(self) -> Iterator[VideoRecord]:
        """产出 `VideoRecord`。实现应保证**流式**，不把全量视频载入内存。"""

    @abstractmethod
    def __len__(self) -> int:
        """记录条数（视频数，不是字幕数）。"""

    def describe(self) -> Dict[str, Any]:
        """自描述（写进数据卡）。子类应覆盖以补充来源特有的字段。"""
        return {"source": self.name, "n_videos": len(self)}

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        return f"<{type(self).__name__} name={self.name} n={len(self)}>"


# =============================================================================
# MSR-VTT
# =============================================================================
#: MSR-VTT 的 20 个类别。⚠️ 名称映射来自公开资料，**未在本项目内独立核验**，
#: 所以只用于数据卡的"可读性"，配比一律用**数字 category id**，
#: 避免一个没核验过的字符串映射悄悄影响数据决策。
MSRVTT_CATEGORIES: Dict[int, str] = {
    0: "music", 1: "people", 2: "gaming", 3: "sports", 4: "news",
    5: "tv", 6: "cooking", 7: "advertising", 8: "movie", 9: "animation",
    10: "vehicles", 11: "beauty", 12: "dance", 13: "animals", 14: "education",
    15: "howto", 16: "travel", 17: "science", 18: "documentary", 19: "comedy",
}


@dataclass
class MSRVTTConfig:
    """MSR-VTT 读取配置。"""

    #: 元数据 json（可以是 .json，也可以是含 json 的 .zip）
    anno: Path
    #: 视频目录（解压后的 mp4 所在目录）。None 表示只读元数据。
    video_dir: Optional[Path] = None
    split: str = "test"
    #: 只取前 N 条（快速冒烟用）；None 表示全量
    limit: Optional[int] = None
    #: 视频文件名的候选模板。MSR-VTT 官方是 `video{id}.mp4`，
    #: 但不同镜像打包方式不同，所以给多个候选、逐个探测。
    name_templates: Sequence[str] = ("video{id}.mp4", "{video_id}.mp4", "{id}.mp4")


class MSRVTTVideoSource(CorpusSource):
    """MSR-VTT 测试集：2990 条真实 YouTube 短片，每条 20 条人工字幕。

    选它做主力语料的原因：它是**视频-文本**领域被引用最多的基准之一，
    字段干净（每视频固定 20 条字幕，可用来测"字幕去重"和"字幕质量分布"），
    且视频是真实网络视频、**带真实剪辑** —— 这正是评估镜头检测所需要的。
    """

    name = "msrvtt"

    def __init__(self, cfg: MSRVTTConfig) -> None:
        self.cfg = cfg
        self._videos, self._sentences = _load_msrvtt_anno(cfg.anno)
        if cfg.limit is not None:
            self._videos = self._videos[: cfg.limit]
        self._missing = 0
        self._roots: Optional[List[Path]] = None

    # ---------------- 路径解析 ----------------
    def _candidate_roots(self) -> List[Path]:
        """视频所在的候选目录（**缓存**，否则每条视频都要扫一遍目录）。

        ⚠️ 真实数据集打包时几乎总有一层"壳目录"：MSR-VTT 官方 zip 里是
        `TestVideo/`，ActivityNet 的 tar 里是 `ActivityNet_Videos/`。
        只查 `video_dir` 本身会让 2990 条视频**全部**被记为 missing，
        而现象只是"找不到视频"—— 很难归因到"少探了一层目录"。
        """
        if self._roots is None:
            root = Path(self.cfg.video_dir)  # type: ignore[arg-type]
            roots = [root]
            try:
                roots += sorted(p for p in root.glob("*") if p.is_dir())
            except OSError:
                pass
            self._roots = roots
        return self._roots

    def _resolve_path(self, video_id: str, numeric_id: Any) -> Optional[Path]:
        if self.cfg.video_dir is None:
            return None
        roots = self._candidate_roots()
        for tpl in self.cfg.name_templates:
            name = tpl.format(id=numeric_id, video_id=video_id)
            for root in roots:
                p = root / name
                if p.exists():
                    return p
        return None

    # ---------------- 迭代 ----------------
    def __iter__(self) -> Iterator[VideoRecord]:
        by_vid: Dict[str, List[Dict[str, Any]]] = {}
        for s in self._sentences:
            by_vid.setdefault(s["video_id"], []).append(s)

        self._missing = 0
        for v in self._videos:
            vid = v["video_id"]
            dur = float(v.get("end time", 0.0)) - float(v.get("start time", 0.0))
            path = self._resolve_path(vid, v.get("id"))
            if self.cfg.video_dir is not None and path is None:
                self._missing += 1

            caps = [
                CaptionRecord(
                    text=s["caption"],
                    video_id=vid,
                    source=self.name,
                    # ⚠️ 见模块 docstring：MSR-VTT 的字幕**没有**片内时间位置，
                    # 整段共用，所以这里填整段而不是用源数据的 start/end。
                    start=0.0,
                    end=max(dur, 0.0),
                    meta={"sen_id": s.get("sen_id")},
                )
                for s in by_vid.get(vid, [])
            ]
            yield VideoRecord(
                video_id=vid,
                source=self.name,
                split=self.cfg.split,
                path=path,
                captions=caps,
                meta={
                    "duration": dur,
                    "category": v.get("category"),
                    "category_name": MSRVTT_CATEGORIES.get(v.get("category"), "?"),
                    "url": v.get("url"),
                    # 源片段在原 YouTube 视频里的位置（不是字幕时间）
                    "source_clip": [v.get("start time"), v.get("end time")],
                    "n_captions_expected": 20,
                },
            )

    def __len__(self) -> int:
        return len(self._videos)

    def describe(self) -> Dict[str, Any]:
        cats: Dict[str, int] = {}
        for v in self._videos:
            key = str(v.get("category"))
            cats[key] = cats.get(key, 0) + 1
        return {
            "source": self.name,
            "split": self.cfg.split,
            "n_videos": len(self._videos),
            "n_captions": len(self._sentences),
            "captions_per_video": round(len(self._sentences) / max(len(self._videos), 1), 2),
            "category_hist": dict(sorted(cats.items(), key=lambda kv: int(kv[0]))),
            "video_dir": str(self.cfg.video_dir) if self.cfg.video_dir else None,
            "missing_videos": self._missing,
            "note": "字幕为整段描述，无片内时间定位；源片段区间见 meta.source_clip",
        }


def _load_msrvtt_anno(anno: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """读 MSR-VTT 标注，支持 .json 与含 json 的 .zip 两种形态。

    ⚠️ 为什么两种都要支持：HF 上的镜像把 json 打包成 zip，
    而本地解压后又变成裸 json。写死一种，换数据源就炸。
    """
    anno = Path(anno)
    if not anno.exists():
        raise FileNotFoundError(f"找不到 MSR-VTT 标注文件：{anno}")
    if anno.suffix.lower() == ".zip":
        with zipfile.ZipFile(anno) as z:
            names = [n for n in z.namelist() if n.endswith(".json")]
            if not names:
                raise ValueError(f"{anno} 里没有 json")
            data = json.loads(z.read(names[0]).decode("utf-8"))
    else:
        data = json.loads(anno.read_text(encoding="utf-8"))
    return data.get("videos", []), data.get("sentences", [])


# =============================================================================
# ActivityNet Captions
# =============================================================================
@dataclass
class ActivityNetConfig:
    """ActivityNet Captions 读取配置。

    json 结构::

        {"database": {"<video_id>": {"duration": float,
                                     "timestamps": [[s, e], ...],
                                     "sentences":  ["...", ...]}}}
    """

    anno: Path
    video_dir: Optional[Path] = None
    split: str = "train"
    limit: Optional[int] = None


class ActivityNetCaptionSource(CorpusSource):
    """ActivityNet Captions：真实长视频 + **稠密时序字幕**。

    和 MSR-VTT 的关键差别是字幕带**片内时间区间**，所以它能支撑
    "按时间段切帧 → 和该段字幕配对"这种真正的**时序**多模态数据构造，
    而不只是"整段视频 ↔ 整句描述"。
    """

    name = "activitynet"

    def __init__(self, cfg: ActivityNetConfig) -> None:
        self.cfg = cfg
        self._db = _load_activitynet_anno(cfg.anno)
        self._ids = list(self._db.keys())
        if cfg.limit is not None:
            self._ids = self._ids[: cfg.limit]

    def __iter__(self) -> Iterator[VideoRecord]:
        for vid in self._ids:
            entry = self._db[vid]
            dur = float(entry.get("duration", 0.0) or 0.0)
            ts = entry.get("timestamps", []) or []
            sents = entry.get("sentences", []) or []
            caps = []
            for i, text in enumerate(sents):
                a, b = (ts[i] if i < len(ts) else [0.0, dur])
                caps.append(CaptionRecord(
                    text=str(text), video_id=vid, source=self.name,
                    start=float(a), end=float(b),
                ))
            p = None
            if self.cfg.video_dir is not None:
                for cand in (f"{vid}.mp4", f"{vid}.mkv", f"{vid}.webm"):
                    if (Path(self.cfg.video_dir) / cand).exists():
                        p = Path(self.cfg.video_dir) / cand
                        break
            yield VideoRecord(
                video_id=vid, source=self.name, split=self.cfg.split,
                path=p, captions=caps,
                meta={"duration": dur, "n_segments": len(ts),
                      "temporal": True},
            )

    def __len__(self) -> int:
        return len(self._ids)

    def describe(self) -> Dict[str, Any]:
        n_cap = sum(len(e.get("sentences", []) or []) for e in self._db.values())
        durs = [float(e.get("duration", 0) or 0) for e in self._db.values()]
        return {
            "source": self.name,
            "split": self.cfg.split,
            "n_videos": len(self._ids),
            "n_captions": n_cap,
            "mean_duration_s": round(sum(durs) / max(len(durs), 1), 2),
            "note": "字幕带片内时间区间（temporal），可支撑时序配对",
        }


def _load_activitynet_anno(anno: Path) -> Dict[str, Any]:
    anno = Path(anno)
    if not anno.exists():
        raise FileNotFoundError(f"找不到 ActivityNet 标注：{anno}")
    data = json.loads(anno.read_text(encoding="utf-8"))
    return data.get("database", data)


# =============================================================================
# 合成源（对照用）
# =============================================================================
@dataclass
class SyntheticCorpusConfig:
    """合成语料配置：把内存帧序列**落成真实 mp4** 再进管线。

    为什么落盘而不是直接传内存帧
    ----------------------------
    因为要保证合成数据与真实数据**走完全相同的解码路径**。
    如果合成走内存、真实走 cv2，那两者测出来的吞吐和"帧下标对齐"行为
    根本不可比 —— 而合成数据的全部价值就是"给出一个已知真值的对照"。
    """

    n_videos: int = 8
    n_shots: int = 6
    frames_per_shot: int = 12
    width: int = 320
    height: int = 240
    seed: int = 2026
    heterogeneous: bool = True
    cache_dir: Path = Path("data/cache/synthetic_videos")


class SyntheticVideoSource(CorpusSource):
    """合成视频源：镜头边界**已知**，用来做管线正确性的金标准对照。"""

    name = "synthetic"

    def __init__(self, cfg: Optional[SyntheticCorpusConfig] = None) -> None:
        self.cfg = cfg or SyntheticCorpusConfig()
        self._records: List[VideoRecord] = []
        self._build()

    def _build(self) -> None:
        from roboground.data.video.shot import make_synthetic_shot_video

        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        for k in range(self.cfg.n_videos):
            frames, truth, labels = make_synthetic_shot_video(
                n_shots=self.cfg.n_shots,
                frames_per_shot=self.cfg.frames_per_shot,
                width=self.cfg.width, height=self.cfg.height,
                seed=self.cfg.seed + k,
                heterogeneous=self.cfg.heterogeneous,
            )
            path = self.cfg.cache_dir / f"synthetic_{k:03d}.mp4"
            if not path.exists():
                _write_video(path, frames)
            self._records.append(VideoRecord(
                video_id=f"synthetic_{k:03d}",
                source=self.name, split="all", path=path,
                captions=[CaptionRecord(
                    text=f"a synthetic indoor scene with {len(truth) + 1} shots",
                    video_id=f"synthetic_{k:03d}", source=self.name,
                    start=0.0, end=len(frames) / 10.0,
                )],
                meta={
                    "duration": len(frames) / 10.0,
                    "n_frames": len(frames),
                    #: 真值镜头边界（帧下标）—— 只有合成数据才有
                    "truth_boundaries": list(truth),
                    "shot_labels": list(labels),
                },
            ))

    def __iter__(self) -> Iterator[VideoRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def describe(self) -> Dict[str, Any]:
        return {
            "source": self.name,
            "n_videos": len(self._records),
            "n_shots_each": self.cfg.n_shots,
            "frames_each": self.cfg.n_shots * self.cfg.frames_per_shot,
            "note": "镜头边界已知，作为管线正确性的金标准对照",
        }


def _write_video(path: Path, frames: Sequence[Any], fps: float = 10.0) -> None:
    """把 RGB 帧序列写成 mp4。编解码器不可用时**明确报错**而不是静默降级。"""
    import cv2  # noqa: PLC0415

    if not frames:
        raise ValueError("没有帧可写")
    h, w = frames[0].shape[:2]
    for fourcc_name in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name),
                                 fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
    else:  # pragma: no cover - 取决于本机 codec
        raise RuntimeError("本机 OpenCV 没有可用的 mp4 编码器（试过 mp4v/avc1）")
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


# =============================================================================
# 工厂
# =============================================================================
def build_source(kind: str, **kwargs: Any) -> CorpusSource:
    """按名字构建数据源（与项目其它 registry 风格一致）。

    Examples
    --------
    >>> src = build_source("msrvtt", cfg=MSRVTTConfig(anno=Path("...json")))
    """
    k = kind.lower()
    if k == "msrvtt":
        return MSRVTTVideoSource(kwargs.get("cfg") or MSRVTTConfig(**kwargs))
    if k in ("activitynet", "anet"):
        return ActivityNetCaptionSource(kwargs.get("cfg") or ActivityNetConfig(**kwargs))
    if k == "synthetic":
        return SyntheticVideoSource(kwargs.get("cfg") or SyntheticCorpusConfig(**kwargs))
    raise ValueError(f"未知数据源 {kind!r}，可选：msrvtt / activitynet / synthetic")


__all__ = [
    "CorpusSource", "MSRVTTVideoSource", "MSRVTTConfig", "MSRVTT_CATEGORIES",
    "ActivityNetCaptionSource", "ActivityNetConfig",
    "SyntheticVideoSource", "SyntheticCorpusConfig",
    "build_source",
]
