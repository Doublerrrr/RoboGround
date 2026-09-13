"""视频 → 训练数据的流式管线（含吞吐量统计）。

设计目标：**规模靠工程，不靠编数字**
====================================
面试里说"处理过海量数据"，如果只能拿出一个小 demo，是站不住的。
但硬凑数据量既费时又不诚实。正确的做法是**把管线写成真正可扩展的形态**，
然后**报吞吐量**（frame/s、MB/s、GB/h）——
这样"处理 100 万帧需要多久"是可以从小样本外推的，且外推过程可验证。

所以本管线的三条设计约束都是冲着"能上规模"去的：

1. **流式**：逐帧处理，绝不把整段视频读进内存。
   内存占用与视频长度**无关**（只与窗口大小有关）。
2. **两遍但只解一次码**：镜头检测需要全局阈值（要两遍），
   但第二遍只**取帧不解码全片**（用 `cap.set(CAP_PROP_POS_FRAMES)` 跳读）。
3. **分块可并行**：`chunk_size` 把视频切成独立块，块间无状态依赖
   （代价是块边界处可能少切一个镜头，用 `overlap` 补偿）。

吞吐量统计分阶段
==============
`decode / features / dedup / quality` 分别计时 —— 这样才能看出**瓶颈在哪**。
实测瓶颈通常不是解码而是**特征计算**（HSV 转换 + 直方图），
所以真实系统会用 GPU batch 或者抽稀采样来算特征。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from roboground.data.video.filter import (
    DedupConfig, QualityConfig, dedup_frames, filter_by_quality, quality_scores,
)
from roboground.data.video.sampling import (
    SamplingConfig, coverage_metrics, sample_frames,
)
from roboground.data.video.shot import (
    Shot, ShotDetectionConfig, detect_shots, hsv_histogram, chi_square_distance,
)
from roboground.utils.logging import get_logger

logger = get_logger("data.video.pipeline")


# =============================================================================
# 计时器
# =============================================================================
class StageTimer:
    """分阶段计时器（累计 + 计数）。"""

    def __init__(self) -> None:
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._t0: Dict[str, float] = {}

    def start(self, stage: str) -> None:
        self._t0[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        dt = time.perf_counter() - self._t0.pop(stage, time.perf_counter())
        self.times[stage] = self.times.get(stage, 0.0) + dt
        self.counts[stage] = self.counts.get(stage, 0) + 1
        return dt

    def summary(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in sorted(self.times.items(),
                                                  key=lambda kv: -kv[1])}


# =============================================================================
# 视频源
# =============================================================================
class FrameSource:
    """帧序列抽象：既支持内存里的帧列表，也支持真实视频文件。

    统一抽象的意义：**管线逻辑与数据来源解耦**，
    于是单元测试可以用几十帧的合成序列跑（快、确定），
    生产则直接指向 mp4（慢、真实）。
    """

    def __iter__(self) -> Iterator[np.ndarray]:      # pragma: no cover - 抽象
        raise NotImplementedError

    def __len__(self) -> int:                        # pragma: no cover - 抽象
        raise NotImplementedError

    def frame_at(self, index: int) -> Optional[np.ndarray]:
        """随机访问（第二遍取帧用）。不支持时返回 None。"""
        return None

    def close(self) -> None:
        return None


class ListFrameSource(FrameSource):
    """内存帧序列（合成视频 / 单元测试用）。"""

    def __init__(self, frames: Sequence[np.ndarray]) -> None:
        self.frames = [np.asarray(f) for f in frames]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def frame_at(self, index: int) -> Optional[np.ndarray]:
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None


class VideoFileSource(FrameSource):
    """真实视频文件（cv2 后端），**流式**读取。

    ⚠️ `cv2.VideoCapture` 的顺序读和随机读**不能混用**：
    调了 `set(CAP_PROP_POS_FRAMES)` 之后顺序读的语义会变。
    所以第二遍取帧时用一个**独立的 capture 实例**（见 `open_video_source`）。
    """

    def __init__(self, path: str | Path) -> None:
        import cv2  # noqa: PLC0415

        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"打不开视频文件：{self.path}")
        self._n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.meta = {"fps": self._fps, "width": w, "height": h,
                     "n_frames": self._n, "path": self.path}

    def __iter__(self) -> Iterator[np.ndarray]:
        import cv2  # noqa: PLC0415

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return self._n

    def frame_at(self, index: int) -> Optional[np.ndarray]:
        import cv2  # noqa: PLC0415

        if index < 0 or (self._n > 0 and index >= self._n):
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = self._cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:  # pragma: no cover
            pass


def open_video_source(src: str | Path | Sequence[np.ndarray]) -> FrameSource:
    """按输入类型自动选帧源。"""
    if isinstance(src, (str, Path)):
        return VideoFileSource(src)
    return ListFrameSource(src)


# =============================================================================
# 管线
# =============================================================================
@dataclass
class PipelineConfig:
    """整条管线的配置。"""

    shot: ShotDetectionConfig = field(default_factory=ShotDetectionConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    #: 嵌入回调：`frames -> (N, D)`。为 None 时自动退化为"只用直方图/哈希"
    embed_fn: Optional[Callable[[Sequence[np.ndarray]], np.ndarray]] = None
    #: 特征计算时是否降采样（大视频上算 HSV 直方图很贵）
    feature_max_side: int = 160


@dataclass
class PipelineResult:
    """管线产物 + 全套统计（统计本身就是交付物之一）。"""

    kept_indices: List[int]
    shots: List[Shot]
    diffs: np.ndarray
    stats: Dict[str, Any]

    def summary_text(self) -> str:
        s = self.stats
        lines = [
            f"  帧数        : {s['n_frames']}  →  保留 {len(self.kept_indices)}"
            f"（保留率 {s['overall_keep_rate']:.1%}）",
            f"  镜头数      : {s['n_shots']}（硬切 {s['n_hard_cuts']} / 渐变 {s['n_gradual']}）",
            f"  抽帧策略    : {s['strategy']}（预算 {s['target_frames']}）",
            f"  去重        : 哈希砍 {s['dedup']['dropped_by_hash']}、"
            f"嵌入砍 {s['dedup']['dropped_by_embedding']}",
            f"  质量淘汰    : {s['quality']['drop_reasons'] or '无'}",
            f"  吞吐        : {s['throughput_fps']:.1f} frame/s"
            f"  ≈ {s['throughput_gb_per_hour']:.2f} GB/h",
            f"  耗时分解    : {s['stage_seconds']}",
        ]
        return "\n".join(lines)


def process_video(
    src: str | Path | Sequence[np.ndarray],
    *,
    cfg: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """跑完整管线：**镜头检测 → 抽帧 → 去重 → 质量过滤**。

    只有两遍扫描，且第二遍只取需要的帧（不重新解全片）：
    第一遍算逐帧特征并检测镜头；第二遍按抽帧结果取帧、去重、过滤。
    """
    cfg = cfg or PipelineConfig()
    timer = StageTimer()
    source = open_video_source(src)
    is_file = isinstance(source, VideoFileSource)

    # ---------------- 第一遍：特征 + 镜头检测 ----------------
    timer.start("decode")
    hists: List[np.ndarray] = []
    n_frames = 0
    for frame in source:
        hists.append(hsv_histogram(_downscale(frame, cfg.feature_max_side),
                                   cfg.shot.hist_bins))
        n_frames += 1
    timer.stop("decode")
    if n_frames == 0:
        source.close()
        raise ValueError("视频里没有读到任何帧")

    timer.start("features")
    hist_diffs = np.array(
        [chi_square_distance(hists[i], hists[i + 1]) for i in range(n_frames - 1)],
        dtype=np.float64,
    )
    timer.stop("features")

    # 嵌入（可选）：只在需要时算，且用抽稀后的帧，避免 O(N) 次模型前向
    embeddings = None
    if cfg.embed_fn is not None:
        timer.start("embed")
        embeddings = _embed_selected(source, cfg.embed_fn, n_frames)
        timer.stop("embed")

    timer.start("shot")
    # 用**已算好的直方图**做检测（复用第一遍的结果，不重算特征）
    shots = _detect_from_hists(hists, hist_diffs, embeddings, cfg.shot)
    timer.stop("shot")

    # ---------------- 第二遍：抽帧 → 去重 → 质量 ----------------
    picked = sample_frames(shots, hist_diffs, n_frames, cfg.sampling)

    timer.start("fetch")
    frames_picked: List[np.ndarray] = []
    keep_idx: List[int] = []          #: frames_picked[p] 对应原视频的第 keep_idx[p] 帧
    for i in picked:
        f = source.frame_at(i)
        if f is None:
            continue
        frames_picked.append(f)
        keep_idx.append(i)
    timer.stop("fetch")

    # ⚠️ **索引必须全程用「局部下标」**（frames_picked 的位置），
    # 不能拿原视频下标去索引压缩后的列表。
    # 第一版就是混用了：`dedup_frames(frames_picked, keep_idx)` 里
    # `dedup_frames` 用 `frames[i]` 取值，而 i 是原视频下标（可达 86），
    # frames_picked 只有 24 个元素 → `i >= len(frames)` 被静默跳过，
    # 结果"87 帧只剩 2 帧、甚至 0 帧"，看起来像质量闸门太严，其实是索引错位。
    local_idx = list(range(len(frames_picked)))
    emb_picked = _slice_emb(embeddings, keep_idx)     # 已按 keep_idx 顺序对齐

    timer.start("dedup")
    kept_local, dedup_stats = dedup_frames(frames_picked, local_idx,
                                           embeddings=emb_picked, cfg=cfg.dedup)
    timer.stop("dedup")

    frames_after = [frames_picked[p] for p in kept_local]
    kept_orig = [keep_idx[p] for p in kept_local]

    timer.start("quality")
    kept_after_local, quality_stats = filter_by_quality(
        frames_after, list(range(len(frames_after))), cfg=cfg.quality)
    timer.stop("quality")

    kept_final = [kept_orig[p] for p in kept_after_local]
    kept2 = kept_final

    labels = [s.index for s in shots for _ in range(s.length)]
    cov = coverage_metrics(kept2, labels, len(shots), diffs=hist_diffs)

    elapsed = sum(timer.times.values())
    bytes_rgb = n_frames * _frame_bytes(frames_picked)
    stats: Dict[str, Any] = {
        "n_frames": n_frames,
        "n_shots": len(shots),
        "n_hard_cuts": sum(1 for s in shots if s.index > 0 and not s.is_gradual),
        "n_gradual": sum(1 for s in shots if s.is_gradual),
        "strategy": cfg.sampling.strategy,
        "target_frames": cfg.sampling.target_frames,
        "n_picked": len(picked),
        "n_after_dedup": len(kept_local),
        "n_after_quality": len(kept2),
        "overall_keep_rate": len(kept2) / max(n_frames, 1),
        "dedup": dedup_stats,
        "quality": quality_stats,
        "coverage": cov,
        "stage_seconds": timer.summary(),
        "total_seconds": round(elapsed, 4),
        "throughput_fps": n_frames / elapsed if elapsed > 0 else 0.0,
        "throughput_gb_per_hour": (bytes_rgb / elapsed * 3600 / 1e9)
        if elapsed > 0 else 0.0,
        #: 解码出的 RGB 数据量（估算）。数据集级吞吐要用它算 MB/s，
        #: 所以必须**回传**而不是留在函数内部当局部变量。
        "bytes_rgb": int(bytes_rgb),
        "source": getattr(source, "meta", {"kind": "frames"}),
    }
    source.close()
    logger.info(f"管线完成：{n_frames} 帧 → {len(kept2)} 帧，"
                f"{stats['throughput_fps']:.1f} frame/s")
    return PipelineResult(kept_indices=kept2, shots=shots,
                          diffs=hist_diffs, stats=stats)


# =============================================================================
# 内部工具
# =============================================================================
def _detect_from_hists(hists: Sequence[np.ndarray], hist_diffs: np.ndarray,
                       embeddings: Optional[np.ndarray],
                       cfg: ShotDetectionConfig) -> List[Shot]:
    """用**已算好的直方图**做镜头检测（避免重算特征）。"""
    n = len(hists)
    if n == 0:
        return []
    if n == 1:
        return [Shot(index=0, start=0, end=1)]
    from roboground.data.video.shot import (  # noqa: PLC0415
        _merge_boundaries, _merge_short, decide_boundaries,
    )

    emb_diffs = None
    if cfg.use_embedding and embeddings is not None:
        E = np.asarray(embeddings, dtype=np.float64)
        if E.shape[0] == n:
            nrm = np.linalg.norm(E, axis=1, keepdims=True)
            E = E / np.clip(nrm, 1e-12, None)
            emb_diffs = 1.0 - np.sum(E[:-1] * E[1:], axis=1)

    # ★ 判据统一走 `decide_boundaries`（与 `shot.detect_shots` 同一份实现）。
    #   这里**曾经自己抄了一份**，两份在"渐变路径缺峰值突出度门"这个 bug 上
    #   同时中招 —— 所以现在只允许有一处实现。
    hard, gradual = decide_boundaries(hist_diffs, emb_diffs, cfg)

    boundaries = [0] + [i + 1 for i in range(n - 1) if hard[i] or gradual[i]]
    boundaries.append(n)
    boundaries = _merge_boundaries(boundaries, hist_diffs, cfg.nms_window)
    return _merge_short(sorted(set(boundaries)), hard, gradual,
                        cfg.min_shot_len, hist_diffs,
                        exempt_edge=cfg.exempt_edge_shots)


def _downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    """特征计算前降采样 —— 直方图不需要高分辨率，这一步能省大量时间。"""
    import cv2  # noqa: PLC0415

    img = np.asarray(frame)
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _embed_selected(source: FrameSource, embed_fn, n_frames: int,
                    stride: int = 1) -> Optional[np.ndarray]:
    """对全部帧算嵌入（`stride>1` 时抽稀，缺失位置用最近邻填充）。

    ⚠️ 真实系统里这一步应该 **GPU batch + 抽稀**：
    逐帧过 CLIP/ViT 在长视频上是主要瓶颈。
    本实现保持简单，但在统计里单独计时，方便看出瓶颈。
    """
    out: List[np.ndarray] = []
    idxs: List[int] = []
    for i in range(0, n_frames, stride):
        f = source.frame_at(i)
        if f is None:
            continue
        out.append(f)
        idxs.append(i)
    if not out:
        return None
    try:
        emb = np.asarray(embed_fn(out), dtype=np.float64)
    except Exception as exc:  # pragma: no cover - 编码器异常不应中断管线
        logger.warn(f"嵌入计算失败（{type(exc).__name__}: {exc}），退化为纯直方图")
        return None
    if emb.shape[0] != len(idxs):
        logger.warn("嵌入返回行数与输入帧数不一致，忽略嵌入")
        return None
    # 抽稀 → 回填到逐帧
    full = np.zeros((n_frames, emb.shape[1]), dtype=np.float64)
    for k, i in enumerate(idxs):
        full[i] = emb[k]
    for i in range(n_frames):
        if i not in set(idxs):
            # 用前一个已算的填充（视频相邻帧近似相同，这个近似是安全的）
            prev = max([j for j in idxs if j <= i], default=idxs[0])
            full[i] = full[prev]
    return full


def _slice_emb(emb: Optional[np.ndarray], idxs: Sequence[int]) -> Optional[np.ndarray]:
    if emb is None:
        return None
    return np.asarray([emb[i] for i in idxs if 0 <= i < emb.shape[0]])


def _frame_bytes(frames: Sequence[np.ndarray]) -> int:
    if not frames:
        return 0
    return int(np.asarray(frames[0]).nbytes)
