"""视频 → 多模态训练数据的管线。

层级关系（**面试讲视频切帧就按这个顺序讲**）：:

    视频流
      └─ 镜头检测（shot.py）        ← 语义单位是"镜头"，不是"秒"
           └─ 抽帧（sampling.py）   ← 预算按内容变化率分配，不按时间长度
                └─ 去重（filter.py）    ← 两级级联：感知哈希 → 嵌入
                     └─ 质量过滤（filter.py）  ← 四道闸门
                          └─ 打标 / 训练数据

`pipeline.process_video` 把上面四步串成**流式**管线，并输出
分阶段耗时与吞吐量（frame/s、GB/h）—— 规模靠工程而不是靠宣称。
"""

from roboground.data.video.filter import (
    DedupConfig,
    QualityConfig,
    dedup_frames,
    dhash,
    filter_by_quality,
    hamming,
    quality_scores,
)
from roboground.data.video.pipeline import (
    FrameSource,
    ListFrameSource,
    PipelineConfig,
    PipelineResult,
    StageTimer,
    VideoFileSource,
    open_video_source,
    process_video,
)
from roboground.data.video.sampling import (
    STRATEGIES,
    SamplingConfig,
    coverage_metrics,
    sample_frames,
)
from roboground.data.video.shot import (
    Shot,
    ShotDetectionConfig,
    chi_square_distance,
    detect_shots,
    evaluate_shot_detection,
    hsv_histogram,
    make_synthetic_shot_video,
)

__all__ = [
    # 镜头
    "Shot", "ShotDetectionConfig", "detect_shots", "hsv_histogram",
    "chi_square_distance", "evaluate_shot_detection", "make_synthetic_shot_video",
    # 抽帧
    "SamplingConfig", "STRATEGIES", "sample_frames", "coverage_metrics",
    # 去重与质量
    "DedupConfig", "QualityConfig", "dedup_frames", "filter_by_quality",
    "dhash", "hamming", "quality_scores",
    # 管线
    "PipelineConfig", "PipelineResult", "process_video", "open_video_source",
    "FrameSource", "ListFrameSource", "VideoFileSource", "StageTimer",
]
