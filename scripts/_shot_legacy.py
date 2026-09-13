#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复前的镜头检测实现（**只用于消融复现**，生产代码不要 import 它）。

为什么要单独一个文件
====================
`threshold_mode="otsu"` **不能**用来复现修复前的行为 —— 修复给
`_otsu_threshold` 加了"阈值必须高于噪声地板"的守卫，于是现在即使显式选
`otsu` 也会在真实视频上回退到 `robust`，两者结果一模一样（实测 36 vs 36）。
要测真正的修复前行为，必须**把旧的实现整个换回去**。

而"换回去"这件事 `22_shot_ablation.py` 和 `24_goldset_eval.py` 都要做 ——
**同一份旧实现只能有一处**（本项目已经因为"判据在两处各写一份"
踩过一次真实故障：修一边漏一边）。所以抽到这里共享。

两处被还原的修复（2026-09-11）
=============================
1. `_otsu_threshold`：**没有**"阈值必须高于噪声地板"的第二道守卫
   （只有"两类样本都不能太少"那一道，而它挡不住单峰分布）；
2. `decide_boundaries`：渐变路径**没有**峰值突出度门。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def old_otsu_threshold(diffs: np.ndarray, n_bins: int = 128) -> float:
    """修复前的 Otsu：只有一道守卫，且回退目标是 `_adaptive_threshold`。"""
    from roboground.data.video import shot as S

    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 8:
        return float(np.max(d)) if d.size else 0.0
    eps = 1e-6
    logs = np.log10(np.clip(d, eps, None))
    lo, hi = float(logs.min()), float(logs.max())
    if hi - lo < 1e-6:
        return float(d.max())
    hist, edges = np.histogram(logs, bins=n_bins, range=(lo, hi))
    total = hist.sum()
    if total == 0:
        return float(d.max())
    p = hist.astype(np.float64) / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    idx = int(np.argmax(sigma_b))
    # ⚠️ 只有这一道守卫 —— 单峰分布被对半劈时它挡不住
    w0, w1 = omega[idx], 1.0 - omega[idx]
    if min(w0, w1) < 0.05:
        return S._adaptive_threshold(d, 6.0, float(d.max()))
    return float(10 ** centers[idx])


def apply_legacy_patches() -> Any:
    """把修复前的实现 monkeypatch 回去，返回被改的模块（供恢复用）。

    ⚠️ 调用方**必须**负责恢复 —— 见 `restore(module)`。
    """
    from roboground.data.video import shot as S

    S._otsu_threshold = old_otsu_threshold
    return S


def restore() -> None:
    """从磁盘重新加载 `shot` 模块，撤销 monkeypatch。

    ⚠️ 不能用"把原函数赋值回去"的方式恢复 —— 那要求调用方自己保存过原函数，
    多一处状态就多一处出错的可能。直接 `importlib.reload` 更可靠。
    """
    import importlib

    from roboground.data.video import shot as S

    importlib.reload(S)
