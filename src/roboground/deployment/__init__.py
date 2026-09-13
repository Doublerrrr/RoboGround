"""部署层：量化、ONNX 导出、异步流水线、ROS2 节点。"""

from roboground.deployment.quantization import (
    cast_inputs_to_model,
    compare_quantization,
    convert_to_fp16,
    dynamic_quantize,
    measure_latency,
    model_dtype,
    model_size_mb,
    quantize_model,
)
from roboground.deployment.export_onnx import (
    export_to_onnx,
    get_onnx_info,
    onnx_available,
    verify_onnx,
)
from roboground.deployment.pipeline import (
    AsyncPipeline,
    LatencyTracker,
    TierStats,
)

__all__ = [
    # 量化
    "quantize_model",
    "dynamic_quantize",
    "convert_to_fp16",
    "measure_latency",
    "model_size_mb",
    "model_dtype",
    "cast_inputs_to_model",
    "compare_quantization",
    # 导出
    "export_to_onnx",
    "verify_onnx",
    "onnx_available",
    "get_onnx_info",
    # 流水线
    "AsyncPipeline",
    "LatencyTracker",
    "TierStats",
]
