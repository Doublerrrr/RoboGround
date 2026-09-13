"""ONNX 导出与校验（Stage 4 的"格式互操作"部分）。

为什么导出 ONNX
--------------
服务机器人端侧几乎不会直接跑 PyTorch：
- NVIDIA 边缘板（Jetson）→ TensorRT；
- 瑞芯微 NPU（RK3588 等）→ RKNN；
- 通用 ARM → NCNN / MNN / ONNXRuntime。

它们的共同入口基本都是 **ONNX**。所以"PyTorch → ONNX → 平台引擎"是
端侧部署的标准链路（这也是我在绿联把 TSM 导出 ONNX 再做 RKNN 编译的路径）。

⚠️ 导出不等于跑通
----------------
- 有些算子 ONNX 表达不了（要拆）；
- 有些平台不支持某些算子（如 RKNN 不支持 3D 卷积 —— 这正是我当年改选
  TSM 的原因）；
- 预处理（resize 插值/归一化/通道顺序）必须和训练时逐像素一致。

所以本模块的核心不只是 `torch.onnx.export`，更是**导出后的数值对齐校验**
（`verify_onnx`）—— 没有校验的导出是不可信的。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from roboground.utils.io import ensure_dir
from roboground.utils.logging import get_logger

logger = get_logger("deployment.onnx")


def onnx_available() -> bool:
    """ONNX 依赖是否就绪。"""
    try:
        import onnx  # noqa: F401, PLC0415
        import onnxruntime  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def export_to_onnx(
    model: Any,
    dummy_input: Any,
    path: str,
    *,
    input_names: Sequence[str] = ("input",),
    output_names: Sequence[str] = ("output",),
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset: int = 17,
    simplify: bool = False,
) -> str:
    """把 PyTorch 模型导出为 ONNX。

    Parameters
    ----------
    dummy_input
        一个 Tensor 或 tuple（多输入时用 tuple）。**shape 必须与部署时一致**。
    dynamic_axes
        动态维度声明，例如 `{"input": {0: "batch", 2: "height", 3: "width"}}`。
        机器人场景建议至少让 batch 维动态。

    Returns
    -------
    str
        导出文件路径。
    """
    import torch  # noqa: PLC0415

    out = Path(path)
    ensure_dir(out.parent)

    model = model.eval()
    inputs = dummy_input if isinstance(dummy_input, (tuple, list)) else (dummy_input,)

    torch.onnx.export(
        model,
        tuple(inputs),
        str(out),
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=int(opset),
        do_constant_folding=True,
    )
    logger.ok(f"ONNX 已导出：{out}（{out.stat().st_size / 1e6:.2f} MB）")

    if simplify and onnx_available():
        try:
            import onnx  # noqa: PLC0415
            from onnxsim import simplify as onnx_simplify  # noqa: PLC0415

            model_onnx = onnx.load(str(out))
            simplified, ok = onnx_simplify(model_onnx)
            if ok:
                onnx.save(simplified, str(out))
                logger.info("ONNX 图已简化")
        except Exception as exc:
            logger.warn(f"ONNX 简化失败（{exc}），保留原始图")

    return str(out)


def verify_onnx(
    onnx_path: str,
    model: Any,
    dummy_input: Any,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    providers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """对比 PyTorch 与 ONNX Runtime 的输出，验证导出一致性。

    没有这一步的导出是"不可信"的 —— 排序错误、算子近似、常量折叠 bug
    都可能让精度悄悄下降几十个点。

    Returns
    -------
    dict
        含 `max_abs_diff` / `passed` / 每个输出的误差。
    """
    if not onnx_available():
        raise ImportError(
            "需要 onnx 与 onnxruntime：pip install -e \".[export]\""
        )
    import onnxruntime as ort  # noqa: PLC0415
    import torch  # noqa: PLC0415

    inputs = dummy_input if isinstance(dummy_input, (tuple, list)) else (dummy_input,)
    feed = {
        f"input_{i}" if i else "input": t.detach().cpu().numpy()
        for i, t in enumerate(inputs)
    }

    # 用 ONNX 自己的输入名，避免命名不一致导致的假失败
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=list(providers) if providers else None,
    )
    input_names = [i.name for i in session.get_inputs()]
    if len(input_names) == len(inputs):
        feed = {name: t.detach().cpu().numpy() for name, t in zip(input_names, inputs)}

    onnx_out = session.run(None, feed)

    with torch.no_grad():
        torch_out = model(*inputs)
    if isinstance(torch_out, (tuple, list)):
        torch_out = list(torch_out)
    else:
        torch_out = [torch_out]

    results = []
    passed = True
    for i, (ref, got) in enumerate(zip(torch_out, onnx_out)):
        ref_np = ref.detach().cpu().numpy()
        got_np = np.asarray(got)
        if ref_np.shape != got_np.shape:
            results.append({
                "index": i, "shape_match": False,
                "ref_shape": list(ref_np.shape), "onnx_shape": list(got_np.shape),
            })
            passed = False
            continue
        diff = np.abs(ref_np - got_np)
        max_diff = float(diff.max())
        denom = np.maximum(np.abs(ref_np), 1e-6)
        max_rel = float((diff / denom).max())
        ok = bool(np.allclose(ref_np, got_np, atol=atol, rtol=rtol))
        passed &= ok
        results.append({
            "index": i, "shape_match": True,
            "max_abs_diff": max_diff, "max_rel_diff": max_rel, "passed": ok,
        })

    report = {
        "onnx_path": str(onnx_path),
        "max_abs_diff": max((r.get("max_abs_diff", float("inf")) for r in results), default=float("inf")),
        "passed": bool(passed),
        "atol": atol, "rtol": rtol,
        "outputs": results,
    }
    if passed:
        logger.ok(f"ONNX 校验通过（最大绝对误差 {report['max_abs_diff']:.2e}）")
    else:
        logger.warn(f"ONNX 校验未通过：{report}")
    return report


def get_onnx_info(onnx_path: str) -> Dict[str, Any]:
    """读取 ONNX 图的基本信息（opset、输入输出、算子统计）。"""
    if not onnx_available():
        raise ImportError("需要 onnx：pip install -e \".[export]\"")
    import onnx  # noqa: PLC0415

    model = onnx.load(str(onnx_path))
    ops: Dict[str, int] = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    def _shape(vi) -> Any:
        try:
            return [d.dim_value if d.dim_value else d.dim_param for d in vi.type.tensor_type.shape.dim]
        except Exception:
            return None

    return {
        "path": str(onnx_path),
        "size_mb": round(Path(onnx_path).stat().st_size / 1e6, 3),
        "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "inputs": [{"name": i.name, "shape": _shape(i)} for i in model.graph.input],
        "outputs": [{"name": o.name, "shape": _shape(o)} for o in model.graph.output],
        "num_nodes": len(model.graph.node),
        "top_ops": sorted(ops.items(), key=lambda kv: -kv[1])[:15],
    }
