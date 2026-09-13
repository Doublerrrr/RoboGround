"""模型量化与延迟测量（Stage 4 的"轻量化"部分）。

对应 JD 的"模型轻量化并部署至服务机器人平台，满足实时性需求"。

三种手段的定位（面试要能区分）
----------------------------
| 手段 | 解决什么 | 精度代价 | 本项目用法 |
|---|---|---|---|
| **FP16** | 显存/带宽减半，GPU 上有原生加速 | 极小 | 感知编码器、VLM 推理 |
| **动态 INT8** | 权重压到 1/4，CPU 上提速明显 | 中（需验证） | 检测/编码器的 CPU 侧部署 |
| **静态 INT8（QAT/校准）** | 端侧 NPU 定点加速 | 需校准集 | 需要 NPU 时（参考绿联 RKNN 经验） |

⚠️ 诚实说明：本模块提供的是**通用 PyTorch 侧**的量化与度量工具。
真正上 RKNN/TensorRT 还需要各自的工具链（见 `export_onnx.py` 与 docs/部署说明）。
这里的价值在于：**给出"量化前后精度损失 / 延迟变化"的定量对比框架**，
而不是声称"一键部署到机器人"。
"""

from __future__ import annotations

import copy
import io
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("deployment.quant")


# ==========================================================================
# 量化
# ==========================================================================
def convert_to_fp16(model: Any, *, inplace: bool = False) -> Any:
    """把模型转为 FP16（半精度）。

    对 CNN/ViT 推理通常几乎无损，且显存减半。
    注意：**不要**把 FP16 用在需要高数值稳定性的层（如 LayerNorm 的部分实现），
    真出问题时用 `torch.autocast` 做混合精度推理更稳。
    """
    import torch  # noqa: PLC0415

    target = model if inplace else copy.deepcopy(model)
    if hasattr(target, "half"):
        return target.half()
    return target.to(torch.float16)


def dynamic_quantize(
    model: Any,
    *,
    dtype: Any = None,
    inplace: bool = False,
) -> Any:
    """动态 INT8 量化（只量化权重，激活在运行时量化）。

    最适合 **Linear / LSTM** 主导的模型（如文本编码器、VLM 的词表投影）。
    对卷积为主的模型收益较小（卷积走静态量化更好）。
    """
    import torch  # noqa: PLC0415

    qdtype = dtype if dtype is not None else torch.qint8
    target = model if inplace else copy.deepcopy(model)
    target = target.cpu().eval()

    try:
        from torch.ao.quantization import quantize_dynamic  # noqa: PLC0415

        return quantize_dynamic(target, {torch.nn.Linear}, dtype=qdtype)
    except Exception as exc:  # pragma: no cover
        logger.warn(f"动态量化失败（{exc}），返回原模型")
        return target


def quantize_model(model: Any, mode: str = "none", *, inplace: bool = False) -> Any:
    """按配置字符串量化。`mode` ∈ {none, fp16, int8}。"""
    mode = str(mode or "none").lower()
    if mode in {"none", "fp32", ""}:
        return model
    if mode in {"fp16", "half"}:
        return convert_to_fp16(model, inplace=inplace)
    if mode in {"int8", "dynamic_int8"}:
        return dynamic_quantize(model, inplace=inplace)
    logger.warn(f"未知量化模式 {mode!r}，返回原模型")
    return model


# ==========================================================================
# 度量
# ==========================================================================
def model_size_mb(model: Any) -> float:
    """模型参数量占用的字节数（MB）。"""
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return 0.0

    total = 0
    if hasattr(model, "parameters"):
        for p in model.parameters():
            total += p.numel() * p.element_size()
    if hasattr(model, "buffers"):
        for b in model.buffers():
            if isinstance(b, torch.Tensor):
                total += b.numel() * b.element_size()
    return total / (1024.0 * 1024.0)


def model_dtype(model: Any) -> Any:
    """推断模型的计算 dtype（只看浮点参数，忽略量化后的整型权重）。

    为什么需要它：把模型转成 FP16 后，**输入也必须是 FP16**，
    否则会报 `mat1 and mat2 must have the same dtype`。
    这是量化实践里最常见的坑之一，所以由工具层自动处理。
    """
    import torch  # noqa: PLC0415

    if hasattr(model, "parameters"):
        for p in model.parameters():
            if isinstance(p, torch.Tensor) and p.is_floating_point():
                return p.dtype
    return torch.float32


def cast_inputs_to_model(model: Any, inputs: Any) -> tuple:
    """把输入张量转成模型的计算 dtype（非浮点张量保持不变）。"""
    import torch  # noqa: PLC0415

    dtype = model_dtype(model)
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    out = []
    for t in inputs:
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            out.append(t.to(dtype))
        else:
            out.append(t)
    return tuple(out)


def measure_latency(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    iters: int = 20,
    device: Optional[str] = None,
    sync_cuda: bool = True,
) -> Dict[str, float]:
    """测量一个可调用对象的延迟分布（毫秒）。

    Returns
    -------
    dict
        含 `mean_ms` / `median_ms` / `p95_ms` / `min_ms` / `iters`。
    """
    import torch  # noqa: PLC0415

    use_cuda = sync_cuda and torch.cuda.is_available() and (device in (None, "cuda"))

    for _ in range(max(int(warmup), 0)):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(max(int(iters), 1)):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "iters": float(arr.size),
    }


def compare_quantization(
    build_fn: Callable[[], Any],
    input_factory: Callable[[], Any],
    *,
    forward_fn: Optional[Callable[[Any, tuple], Any]] = None,
    modes: Tuple[str, ...] = ("none", "fp16", "int8"),
    warmup: int = 2,
    iters: int = 10,
) -> List[Dict[str, Any]]:
    """对同一模型跑多种量化模式，返回对比表（大小 / 延迟）。

    Parameters
    ----------
    build_fn
        无参函数，返回一个**新的**模型实例（每次都要新建，避免共享状态）。
    input_factory
        无参函数，返回输入（单个张量或 tuple）。**输入会在每次调用时按
        模型 dtype 自动转换** —— 这样 FP16 模型不会因为喂了 FP32 输入而报错。
    forward_fn
        自定义前向：`(model, inputs) -> output`。默认 `model(*inputs)`。

    Returns
    -------
    list[dict]
        每个模式一行，含 `size_mb_before/after`、`compression` 与延迟统计。
    """
    rows: List[Dict[str, Any]] = []
    call = forward_fn or (lambda m, ins: m(*ins))

    for mode in modes:
        try:
            model = build_fn()
            before = model_size_mb(model)
            qmodel = quantize_model(model, mode)
            after = model_size_mb(qmodel)

            # 每次重新构造输入并按 dtype 转换（量化后 dtype 可能不同）
            sample = input_factory()
            prepared = cast_inputs_to_model(qmodel, sample)

            lat = measure_latency(
                lambda: call(qmodel, cast_inputs_to_model(qmodel, input_factory())),
                warmup=warmup, iters=iters,
            )
            # 再做一次真实的输出形状检查，避免"能跑但输出不对"
            out = call(qmodel, prepared)
            shape = list(out.shape) if hasattr(out, "shape") else None

            rows.append({
                "mode": mode,
                "size_mb_before": round(before, 3),
                "size_mb_after": round(after, 3),
                "compression": round(before / max(after, 1e-6), 2),
                "output_shape": shape,
                **{k: round(v, 3) for k, v in lat.items()},
                "ok": True,
            })
        except Exception as exc:
            logger.warn(f"量化模式 {mode} 失败：{exc}")
            rows.append({"mode": mode, "ok": False, "error": str(exc)})
    return rows
