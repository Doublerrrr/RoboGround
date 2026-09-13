"""部署层测试：量化、延迟测量、异步流水线、ROS2 bridge（无需 ROS 运行时）。"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from roboground.deployment.pipeline import AsyncPipeline, LatencyTracker
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
from roboground.deployment.ros2.bridge import (
    ROS2_AVAILABLE,
    TOPIC_TYPES,
    answer_to_dict,
    dict_to_frame,
    frame_to_dict,
    from_json,
    map_to_dict,
    to_json,
)
from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame, ReasoningResult


# ==========================================================================
# 量化
# ==========================================================================
class _TinyModel(torch.nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, 4),
        )

    def forward(self, x):
        return self.net(x)


def test_model_size_mb_positive():
    assert model_size_mb(_TinyModel()) > 0


def test_convert_to_fp16_halves_size():
    model = _TinyModel(64)
    before = model_size_mb(model)
    half = convert_to_fp16(model, inplace=False)
    after = model_size_mb(half)
    assert after < before * 0.6          # fp16 应显著变小
    # 原模型不应被修改（inplace=False）
    assert next(model.parameters()).dtype == torch.float32


def test_convert_to_fp16_inplace_modifies():
    model = _TinyModel(16)
    convert_to_fp16(model, inplace=True)
    assert next(model.parameters()).dtype == torch.float16


def test_dynamic_quantize_runs():
    model = _TinyModel(32)
    q = dynamic_quantize(model)
    out = q(torch.randn(2, 32))
    assert out.shape == (2, 4)


def test_quantize_model_dispatch():
    model = _TinyModel(8)
    assert quantize_model(model, "none") is model
    assert next(quantize_model(model, "fp16").parameters()).dtype == torch.float16
    # 未知模式应安全返回原模型
    assert quantize_model(model, "nonsense") is model


def test_measure_latency_returns_distribution():
    model = _TinyModel(16)
    x = torch.randn(4, 16)
    stats = measure_latency(lambda: model(x), warmup=1, iters=5)
    assert stats["iters"] == 5
    assert stats["mean_ms"] >= 0
    assert stats["p95_ms"] >= stats["min_ms"]


def test_compare_quantization_table():
    # 用一个足够大的模型，让 fp16 的体积差异可测量（小模型会被四舍五入淹没）
    rows = compare_quantization(
        build_fn=lambda: _TinyModel(512),
        input_factory=lambda: torch.randn(2, 512),
        modes=("none", "fp16"),
        warmup=1, iters=3,
    )
    assert len(rows) == 2
    assert all(r["ok"] for r in rows), rows
    by_mode = {r["mode"]: r for r in rows}
    assert by_mode["fp16"]["size_mb_after"] < by_mode["none"]["size_mb_after"] * 0.7
    # 输出形状必须一致（量化不能改变输出结构）
    assert by_mode["none"]["output_shape"] == by_mode["fp16"]["output_shape"] == [2, 4]


def test_model_dtype_and_input_casting():
    model = _TinyModel(8)
    assert model_dtype(model) == torch.float32

    half = convert_to_fp16(model)
    assert model_dtype(half) == torch.float16

    # FP32 输入被自动转成 FP16 —— 这正是避免 "same dtype" 报错的关键
    inputs = cast_inputs_to_model(half, torch.randn(2, 8))
    assert inputs[0].dtype == torch.float16
    assert half(*inputs).shape == (2, 4)


def test_dynamic_quantized_model_dtype_is_float32():
    """动态量化后权重是 qint8，但输入仍应是 fp32。"""
    q = dynamic_quantize(_TinyModel(8))
    assert model_dtype(q) == torch.float32


# ==========================================================================
# 延迟统计
# ==========================================================================
def test_latency_tracker_stats():
    t = LatencyTracker(window=10)
    for v in (10.0, 20.0, 30.0):
        t.add(v)
    s = t.stats()
    assert s["count"] == 3
    assert abs(s["mean_ms"] - 20.0) < 1e-9
    assert abs(s["p50_ms"] - 20.0) < 1e-9


def test_latency_tracker_window_limits_samples():
    t = LatencyTracker(window=3)
    for v in range(10):
        t.add(float(v))
    s = t.stats()
    assert s["count"] == 10                     # 总计数不封顶
    assert s["max_ms"] == 9.0
    assert s["min_ms"] == 7.0                   # 窗口只保留最后 3 个


def test_latency_tracker_empty():
    assert LatencyTracker().stats()["count"] == 0


# ==========================================================================
# 异步流水线
# ==========================================================================
def test_async_pipeline_processes_frames():
    processed = []

    def fast(frame):
        processed.append(frame)
        return {"n": len(processed)}

    pipe = AsyncPipeline(fast, fast_hz=0.0, queue_size=8)
    pipe.set_snapshot("initial")
    pipe.start()
    try:
        for i in range(5):
            pipe.submit(i)
        # 等快档消费完（fast_hz=0 表示不节流）
        deadline = time.time() + 3.0
        while len(processed) < 5 and time.time() < deadline:
            time.sleep(0.02)
    finally:
        pipe.stop()

    assert len(processed) == 5
    assert pipe.fast_processed == 5
    assert pipe.get_snapshot() is not None


def test_async_pipeline_slow_tier_runs():
    calls = []
    pipe = AsyncPipeline(
        fast_fn=lambda f: f,
        slow_fn=lambda: calls.append(1) or "answer",
        fast_hz=0.0, slow_hz=50.0,
    )
    pipe.start()
    try:
        time.sleep(0.4)
    finally:
        pipe.stop()
    assert len(calls) > 0
    assert pipe.last_answer == "answer"


def test_async_pipeline_drops_when_full():
    """队列满且不允许丢最旧时，submit 应返回 False 并计数。"""
    gate = {"go": False}

    def fast(frame):
        while not gate["go"]:
            time.sleep(0.005)
        return frame

    pipe = AsyncPipeline(fast, fast_hz=0.0, queue_size=1, drop_oldest=False)
    pipe.start()
    try:
        time.sleep(0.05)                 # 让快档卡在第一个任务上
        results = [pipe.submit(i) for i in range(5)]
    finally:
        gate["go"] = True
        pipe.stop()

    assert any(r is False for r in results)
    assert pipe.fast_dropped > 0


def test_async_pipeline_survives_fast_fn_exception():
    def bad(frame):
        raise RuntimeError("boom")

    pipe = AsyncPipeline(bad, fast_hz=0.0)
    pipe.set_snapshot(None)
    pipe.start()
    try:
        pipe.submit(1)
        time.sleep(0.2)
    finally:
        pipe.stop()
    assert pipe.errors                            # 错误被记录而不是崩掉
    assert "boom" in pipe.errors[0]


def test_async_pipeline_context_manager():
    with AsyncPipeline(fast_fn=lambda f: f, fast_hz=0.0) as pipe:
        assert pipe.stats()["running"] is True
    assert pipe.stats()["running"] is False


def test_async_pipeline_stats_shape():
    pipe = AsyncPipeline(fast_fn=lambda f: f, fast_hz=0.0)
    s = pipe.stats()
    assert "fast" in s and "slow" in s
    assert "p50_ms" in s["fast"]
    assert isinstance(pipe.describe(), str)


# ==========================================================================
# ROS2 bridge（纯数据层，不需要 ROS）
# ==========================================================================
def _frame() -> RGBDFrame:
    K = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, width=100, height=80)
    return RGBDFrame(
        color=np.full((80, 100, 3), 128, np.uint8),
        depth_m=np.full((80, 100), 2.0, np.float32),
        intrinsics=K, pose=CameraPose.identity(), frame_id="f0", timestamp=1.5,
        meta={"boxes_3d": np.array([[0.0, 2.0, 0.0, 0.5, 0.5, 0.5, 0.0]], np.float32),
              "labels": ["cup"]},
    )


def test_topic_types_are_declared():
    assert "/camera/color/image_raw" in TOPIC_TYPES
    assert "/roboground/answer" in TOPIC_TYPES


def test_frame_to_dict_metadata_only():
    d = frame_to_dict(_frame())
    assert d["frame_id"] == "f0"
    assert d["intrinsics"]["fx"] == 100.0
    assert d["labels"] == ["cup"]
    assert "color_bytes" not in d                 # 默认不塞图像


def test_frame_dict_roundtrip_with_images():
    d = frame_to_dict(_frame(), include_images=True)
    restored = dict_to_frame(d)
    assert restored is not None
    assert restored.color.shape == (80, 100, 3)
    assert np.allclose(restored.depth_m, 2.0)
    assert restored.frame_id == "f0"
    assert restored.meta["labels"] == ["cup"]


def test_dict_to_frame_without_images_returns_none():
    assert dict_to_frame({"frame_id": "x"}) is None


def test_map_to_dict(cfg, quiet, prompts=None):
    from roboground.data.synthetic import make_synthetic_sequence
    from roboground.mapping import MapBuilder

    frames = make_synthetic_sequence(seed=7, num_frames=1, width=160, height=120, num_objects=3)
    cfg.set("perception.prompts", ["table", "chair", "cup", "box", "bottle",
                                   "sofa", "shelf", "monitor", "trash can", "lamp"])
    smap = MapBuilder(cfg).build_from_frames(frames)

    payload = map_to_dict(smap)
    assert payload["type"] == "SemanticMap"
    assert payload["num_objects"] == smap.num_objects
    assert isinstance(payload["objects"], list)
    assert "summary" in payload


def test_answer_to_dict():
    res = ReasoningResult(query="杯子在哪", answer="在桌子上", confidence=0.9)
    d = answer_to_dict(res)
    assert d["type"] == "ReasoningResult"
    assert d["answer"] == "在桌子上"


def test_json_helpers_handle_numpy():
    payload = {"a": np.int64(3), "b": np.float32(1.5), "c": np.zeros(3), "d": np.bool_(True)}
    text = to_json(payload)
    back = from_json(text)
    assert back["a"] == 3
    assert abs(back["b"] - 1.5) < 1e-6
    assert back["c"] == [0.0, 0.0, 0.0]
    assert back["d"] is True


def test_from_json_bad_input_is_safe():
    out = from_json("{not json")
    assert "error" in out


def test_ros2_availability_is_boolean():
    assert isinstance(ROS2_AVAILABLE, bool)


@pytest.mark.ros
def test_ros2_nodes_importable_even_without_rclpy():
    """无论有没有 ROS，节点模块都必须可 import（防御式导入）。"""
    from roboground.deployment.ros2 import nodes  # noqa: F401

    assert hasattr(nodes, "PerceptionNode")
    assert hasattr(nodes, "QueryNode")

    if not ROS2_AVAILABLE:
        with pytest.raises(ImportError, match="ROS2"):
            nodes.PerceptionNode(None)
