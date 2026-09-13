#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""镜头检测诊断图：把「帧差曲线 + 阈值 + 检测边界 + 边界邻域帧」画到一张图。

为什么需要它
------------
镜头检测的数字指标（P/R/F1）需要人工标注的真值，而真值本身要靠**看图**。
这个脚本产出"一张图讲完一个视频"的诊断材料：

1. 上：帧差曲线 + 各条阈值（robust / Otsu / 中位数+3σ）+ 检测到的边界；
2. 中：全片总览联络图（查**漏检** —— 整体色调突变但没标出来）；
3. 下：每个边界前后 ±3 帧（查**误检** —— 标了但画面其实没切）。

用法::

    python scripts/21_shot_diag.py --videos video7013 video7012 --out runs/shotdiag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402


def _use_cjk_font() -> None:
    """给 matplotlib 配上中文字体。

    ⚠️ 不配的话中文标签会渲染成"豆腐块"（`Glyph 26159 missing from font`），
    而**这张图的唯一用途就是人工核验** —— 标签读不出来等于图白画。
    候选按平台常见顺序排（Windows 上 Microsoft YaHei 基本必然存在）。
    """
    from matplotlib import font_manager  # noqa: PLC0415

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "PingFang SC", "WenQuanYi Micro Hei"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False   # 负号别被吃成方块
            return


_use_cjk_font()

from roboground.data.video.pipeline import (  # noqa: E402
    PipelineConfig, open_video_source, process_video,
)
from roboground.data.video.shot import (  # noqa: E402
    ShotDetectionConfig, _otsu_threshold, _robust_threshold,
)

ROOT = Path(__file__).resolve().parents[1]


def analyse(path: Path, cfg: ShotDetectionConfig):
    """**必须复用生产管线**（`process_video`），不要自己重算特征。

    踩过两次坑：
    1. 早期版本自己用 `cv2.calcHist` 算 **RGB 8³** 直方图，
       而生产路径用的是 `hsv_histogram(32)` —— 不是同一个特征，
       诊断图上的边界和 `runs/goldset/` 里的核验图**对不上**，
       人工核验时根本没法比对；
    2. 改用 HSV 之后仍然对不上：因为生产路径会先把帧
       **降采样到 `feature_max_side=160`** 再算直方图，而我用的是全分辨率。

    结论：**诊断工具必须调用被测函数本身**，而不是"复刻"它的逻辑 ——
    复刻出来的第二份实现迟早会和第一份分叉。
    """
    res = process_video(path, cfg=PipelineConfig(shot=cfg))
    src = open_video_source(path)
    try:
        frames = list(src)
    finally:
        src.close()
    return frames, res.diffs, res.shots


def render(name: str, path: Path, out: Path, cfg: ShotDetectionConfig) -> Path:
    frames, d, shots = analyse(path, cfg)
    n = len(frames)
    bounds = [s.start for s in shots[1:]]

    t_rob = _robust_threshold(d, cfg.robust_k, cfg.robust_quantile, 0.35)
    t_otsu = _otsu_threshold(d)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    t_3s = med + 3 * 1.4826 * mad

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.5, 1.0], hspace=0.32)

    # ---- 1) 帧差曲线 ----
    ax = fig.add_subplot(gs[0])
    ax.plot(np.arange(len(d)), d, lw=0.9, color="#2563eb", label="帧差 (L1 直方图距离)")
    ax.axhline(t_rob, color="#16a34a", ls="-", lw=1.6,
               label=f"robust 阈值 = {t_rob:.4f}  ← 现行")
    ax.axhline(t_otsu, color="#dc2626", ls="--", lw=1.4,
               label=f"Otsu 阈值 = {t_otsu:.4f}  ← 修复前默认")
    ax.axhline(t_3s, color="#9ca3af", ls=":", lw=1.2,
               label=f"中位数+3σ = {t_3s:.4f}")
    for b in bounds:
        ax.axvline(b - 1, color="#16a34a", alpha=0.35, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylim(max(float(d.min()) * 0.8, 1e-6), None)
    ax.set_xlabel("帧下标")
    ax.set_ylabel("帧差（对数轴）")
    ax.set_title(f"{name}   n={n} 帧   检测到 {len(bounds)} 个边界 {bounds}",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    # ---- 2) 全片总览 ----
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    cols, rows = 12, 4
    step = max(n // (cols * rows), 1)
    idx = list(range(0, n, step))[: cols * rows]
    tw = 96
    sheet = np.zeros((rows * (tw // 4 * 3 + 16), cols * tw, 3), dtype=np.uint8)
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        th = cv2.resize(frames[i], (tw, tw // 4 * 3))
        cv2.putText(th, f"#{i}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
        if any(abs(i - b) <= step for b in bounds):
            th = cv2.copyMakeBorder(th, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255))
            th = cv2.resize(th, (tw, tw // 4 * 3))
        y, x = r * (tw // 4 * 3 + 16), c * tw
        sheet[y:y + th.shape[0], x:x + tw] = th
    ax2.imshow(sheet)
    ax2.set_title(f"总览（红框=检测边界附近；查漏检）  采样步长 {step}", fontsize=10)

    # ---- 3) 边界邻域 ----
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    if bounds:
        w = 3
        ncol = len(bounds)
        cell = 128
        strip = np.zeros((cell, ncol * (2 * w + 1) * (cell // 2) + ncol * 8, 3), dtype=np.uint8)
        x = 0
        for b in bounds:
            for off in range(-w, w + 1):
                i = min(max(b + off, 0), n - 1)
                th = cv2.resize(frames[i], (cell // 2, cell // 2))
                color = (0, 0, 255) if off == 0 else (255, 255, 0)
                cv2.putText(th, f"{i}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1)
                if off == 0:
                    th = cv2.copyMakeBorder(th, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                            value=(0, 0, 255))
                    th = cv2.resize(th, (cell // 2, cell // 2))
                strip[0:th.shape[0], x:x + cell // 2] = th
                x += cell // 2
            x += 8
        ax3.imshow(strip)
        ax3.set_title("各边界 ±3 帧（红框=边界帧；查误检：前后画面是否真变了）", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "无检测边界（单镜头）", ha="center", fontsize=12)
        ax3.set_title("各边界 ±3 帧", fontsize=10)

    out.mkdir(parents=True, exist_ok=True)
    p = out / f"diag_{name}.png"
    fig.savefig(p, dpi=95, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="镜头检测诊断图")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--video-dir",
                    default=str(ROOT / "data/raw/msrvtt/test_videos/TestVideo"))
    ap.add_argument("--out", default=str(ROOT / "runs/shotdiag"))
    args = ap.parse_args()

    cfg = ShotDetectionConfig()
    for v in args.videos:
        p = Path(args.video_dir) / (v if v.endswith(".mp4") else f"{v}.mp4")
        if not p.exists():
            print(f"找不到 {p}")
            continue
        out = render(p.stem, p, Path(args.out), cfg)
        print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
