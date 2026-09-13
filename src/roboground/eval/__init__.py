"""评测层：指标与基准。"""

from roboground.eval.metrics import (
    aggregate,
    box3d_iou,
    detection_metrics,
    format_metrics,
    localization_errors,
    match_boxes,
    polygon_area,
    rotated_rect_corners,
    topk_accuracy,
)
from roboground.eval.benchmark import (
    benchmark_map_query,
    benchmark_perception,
    benchmark_pipeline,
    benchmark_query,
    format_report,
)

__all__ = [
    # 指标
    "box3d_iou",
    "match_boxes",
    "detection_metrics",
    "localization_errors",
    "topk_accuracy",
    "aggregate",
    "format_metrics",
    "rotated_rect_corners",
    "polygon_area",
    # 基准
    "benchmark_pipeline",
    "benchmark_perception",
    "benchmark_query",
    "benchmark_map_query",
    "format_report",
]
