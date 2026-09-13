#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""07 · 模型导出与量化对照（Stage 4 部署链路演示）。

三步走（端侧部署的标准路径）：
```
PyTorch  →  ONNX（格式互操作）  →  量化（FP16/INT8）  →  平台引擎（TensorRT/RKNN，本脚本不涉及）
                       ↑
              导出后必须做数值对齐校验（verify_onnx）
```

本脚本用一个**形状接近轻量视觉骨干**的占位模型演示完整流程，
因为真实的 Grounding DINO / SAM / CLIP 权重动辄几百 MB~几 GB，
不适合在演示脚本里下载。

用法::

    python scripts/07_export_onnx.py
    python scripts/07_export_onnx.py --out runs/export --opset 17
    python scripts/07_export_onnx.py --skip-onnx      # 只做量化对照
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.utils.logging import get_logger, set_verbosity   # noqa: E402


def build_toy_backbone(feature_dim: int = 72):
    """构造一个形状接近轻量骨干的占位模型（输入 1×3×H×W → 输出 1×feature_dim）。"""
    import torch

    class ToyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(128, feature_dim),
            )

        def forward(self, x):
            return self.stem(x)

    return ToyBackbone()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/export")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--feature-dim", type=int, default=72)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)
    log = get_logger("export")

    try:
        import torch
    except ImportError:
        log.error("需要 torch。请使用 lxr 环境（见 setup_env.md）。")
        return 2

    from roboground.deployment.quantization import (
        cast_inputs_to_model,
        compare_quantization,
        model_size_mb,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 66)
    log.info("Stage 4 部署链路演示（占位骨干网络）")
    log.info("=" * 66)
    log.info(f"输入尺寸：1×3×{args.height}×{args.width}，输出维度：{args.feature_dim}")

    model = build_toy_backbone(args.feature_dim).eval()
    log.info(f"原始模型体积：{model_size_mb(model):.3f} MB")

    # ---------------- 1) ONNX 导出 + 校验 ----------------
    if not args.skip_onnx:
        from roboground.deployment.export_onnx import (
            export_to_onnx,
            get_onnx_info,
            onnx_available,
            verify_onnx,
        )

        if not onnx_available():
            log.warn('未安装 onnx/onnxruntime，跳过导出。安装：pip install -e ".[export]"')
        else:
            dummy = torch.randn(1, 3, args.height, args.width)
            onnx_path = out_dir / "toy_backbone.onnx"
            log.info(f"[1/3] 导出 ONNX：{onnx_path}")
            export_to_onnx(
                model, dummy, str(onnx_path),
                input_names=["images"], output_names=["features"],
                dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
                opset=args.opset,
            )

            log.info("[2/3] 数值对齐校验（PyTorch vs ONNX Runtime）")
            report = verify_onnx(str(onnx_path), model, dummy, atol=1e-4, rtol=1e-3)
            log.kv("校验结果", {
                "通过": report["passed"],
                "最大绝对误差": f"{report['max_abs_diff']:.3e}",
            })
            if not report["passed"]:
                log.warn("ONNX 校验未通过 —— 不要直接部署，先排查算子/精度问题")

            info = get_onnx_info(str(onnx_path))
            log.kv("ONNX 图信息", {
                "opset": info["opset"],
                "输入": info["inputs"],
                "输出": info["outputs"],
                "节点数": info["num_nodes"],
                "top 算子": info["top_ops"][:6],
            })

    # ---------------- 2) 量化对照 ----------------
    log.info("[3/3] 量化对照（none / fp16 / int8）")
    rows = compare_quantization(
        build_fn=lambda: build_toy_backbone(args.feature_dim).eval(),
        input_factory=lambda: torch.randn(1, 3, args.height, args.width),
        modes=("none", "fp16", "int8"),
        warmup=2, iters=args.iters,
    )

    header = f"  {'模式':<8}{'体积前(MB)':>12}{'体积后(MB)':>12}{'压缩比':>8}{'mean(ms)':>10}{'p95(ms)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        if not r.get("ok"):
            print(f"  {r['mode']:<8}{'失败':>12}  {str(r.get('error', ''))[:60]}")
            continue
        print(f"  {r['mode']:<8}{r['size_mb_before']:>12.3f}{r['size_mb_after']:>12.3f}"
              f"{r['compression']:>8.2f}{r['mean_ms']:>10.3f}{r['p95_ms']:>10.3f}")

    log.info("")
    log.info("注意：动态 INT8 对 Linear 主导的模型收益大，对纯卷积骨干收益有限；")
    log.info("     要和精度损失一起看（本脚本只演示方法，真实模型需做逐层敏感度分析）。")
    log.info("     上 NPU（如 RKNN）时还要看算子支持情况 —— 例如 RKNN 不支持 3D 卷积，")
    log.info("     这正是本项目在端侧选型时要注意的硬约束（见 docs/实现笔记.md）。")
    log.ok(f"产物目录：{out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
