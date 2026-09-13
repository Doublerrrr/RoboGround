"""感知层：开放词汇 2D 感知（检测 / 分割 / 特征编码）。

典型用法
--------
>>> from roboground import load_config
>>> from roboground.perception import build_pipeline
>>> cfg = load_config()
>>> pipe = build_pipeline(cfg)          # 默认离线 stub 后端
>>> pipe.detector.name
'stub'

真实开放词汇后端（需先装可选依赖）
----------------------------------
>>> cfg.set("perception.detector", "grounding_dino")   # doctest: +SKIP
>>> cfg.set("perception.segmenter", "sam")             # doctest: +SKIP
>>> cfg.set("perception.encoder", "clip")              # doctest: +SKIP
"""

from roboground.perception.base import (
    Detector,
    Encoder,
    PerceptionPipeline,
    Segmenter,
)
from roboground.perception.registry import (
    available_detectors,
    available_encoders,
    available_segmenters,
    build_detector,
    build_encoder,
    build_pipeline,
    build_segmenter,
    register_detector,
    register_encoder,
    register_segmenter,
)

__all__ = [
    # 接口
    "Detector",
    "Segmenter",
    "Encoder",
    "PerceptionPipeline",
    # 注册表
    "register_detector",
    "register_segmenter",
    "register_encoder",
    "build_detector",
    "build_segmenter",
    "build_encoder",
    "build_pipeline",
    "available_detectors",
    "available_segmenters",
    "available_encoders",
]

# ---------------------------------------------------------------------------
# 触发后端注册
# ---------------------------------------------------------------------------
# 这三个 import 必须放在文件末尾：子模块里的 `@register_*` 装饰器在导入时执行，
# 而它们又依赖本模块顶部的 `base` / `registry` 已经就绪。
# 放在这里的好处是：`import roboground.perception` 之后注册表就是完整的，
# `available_detectors()` 立刻能看到所有后端（而不是要等第一次 build 才填上）。
from roboground.perception import detectors, encoders, segmenters  # noqa: E402,F401
