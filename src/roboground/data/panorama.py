# -*- coding: utf-8 -*-
"""多视角 → 全景融合（等距柱状投影 / equirectangular）。

这个模块解决什么
==============
一个采集点上有 **N 个真实视角**（服务机器人原地转一圈，或 Matterport 相机
3 个俯仰 × 6 个方位 = 18 张 RGB-D）。它们**位姿已知、时间上相邻**，
位姿配准后就能融合成一张 **360° 全景**：

     N 张 RGB-D（各自的内参 + 位姿） ──► 球面投影 ──► 逐像素加权融合 ──► 1 张全景

融合出来的全景同时给出 **RGB 与深度**，因此可以直接喂给现有的
`MapBuilder`（它消费的就是「彩色 + 深度 + 内参 + 位姿」），
**不需要任何"虚拟视角重渲染"**。

为什么不用"先单帧造点云再重渲染"（本项目以前的做法）
================================================
那是**同一份观测的重采样**：新视角里原视角看不到的区域**没有点云**，
实测无效深度像素从 42.8% 涨到 77.0%（见 `docs/多视角数据核查报告.md`）。
本模块走的是相反方向：**把多个真实观测融合到同一个球面坐标系**，
每个全景像素都来自**真实采集**，覆盖率是**测出来的**而不是编出来的。

与"虚拟视角"的关键区别（写在这里以免以后被误解）
---------------------------------------------
· 虚拟视角：1 个真实视点 → 造出 N 个**假的**视点（信息量不增加）；
· 本模块：N 个真实视点 → 融合成 1 张全景（**信息量真的增加了**，
  因为 N 个视点看到的是场景的不同部分）。

核心约定
=======
· `CameraPose` 存的是 **world → camera**：`p_cam = R @ p_world + t`；
  相机光心 `C = -R.T @ t`，camera→world 的旋转是 `R.T`。
· 等距柱状图坐标：`u = (az + π) / 2π · W`，`v = (π/2 - el) / π · H`，
  `az = atan2(y, x)`、`el = asin(z / |d|)`（**世界系**，z 朝上）。
· 深度是**沿光轴的 z-depth**（与相机内参配套），不是到光心的距离。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("data.panorama")


# ==========================================================================
# 结果容器
# ==========================================================================
@dataclass
class Panorama:
    """一张融合出来的等距柱状全景（RGB + 深度 + 覆盖统计）。"""

    rgb: np.ndarray                      # (H, W, 3) uint8
    depth_m: np.ndarray                  # (H, W) float32，0 表示无效
    weight: np.ndarray                   # (H, W) float32：该像素累计权重（0 = 无覆盖）
    n_views: int = 0                     # 参与融合的视角数
    n_used: int = 0                      # 真正有像素落进来的视角数
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------- 统计 ----------------
    @property
    def coverage(self) -> float:
        """有有效深度的像素占比。"""
        return float((self.depth_m > 0).mean())

    @property
    def rgb_coverage(self) -> float:
        """有 RGB 的像素占比（可能略高于深度覆盖，因为远端点会被 max_depth 砍掉）。"""
        return float((self.weight > 0).mean())

    def describe(self) -> Dict[str, Any]:
        return {
            "分辨率": f"{self.rgb.shape[1]}x{self.rgb.shape[0]}",
            "参与视角": self.n_views,
            "有效视角": self.n_used,
            "深度覆盖": f"{self.coverage * 100:.1f}%",
            "RGB覆盖": f"{self.rgb_coverage * 100:.1f}%",
            **self.meta,
        }


# ==========================================================================
# 视角 → 球面
# ==========================================================================
def view_rays_world(frame) -> Tuple[np.ndarray, np.ndarray]:
    """把一帧的每个像素变成**世界系单位方向向量**。

    返回 `(dirs, valid_mask)`：`dirs` 形状 `(H, W, 3)`（已归一化），
    `valid_mask` 形状 `(H, W)`，标出深度有效的像素。
    """
    K = frame.intrinsics
    H, W = int(K.height), int(K.width)
    # ★ 用**像素中心** (+0.5)：标准约定是像素 (i,j) 的中心在 (j+0.5, i+0.5)。
    #   不写 +0.5 会给每个像素带来约半像素的系统性角偏差
    #   （实测：光轴像素的仰角偏 0.009 rad ≈ 0.52°，正好半个像素）。
    u = (np.arange(W, dtype=np.float64) + 0.5)[None, :].repeat(H, axis=0)
    v = (np.arange(H, dtype=np.float64) + 0.5)[:, None].repeat(W, axis=1)
    # 相机系方向（未归一化，z=1 平面）
    x = (u - float(K.cx)) / float(K.fx)
    y = (v - float(K.cy)) / float(K.fy)
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1)      # (H, W, 3)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    # camera → world：旋转部分是 R.T（pose 存的是 world→camera）
    R = np.asarray(frame.pose.R, dtype=np.float64).reshape(3, 3)
    dirs = d_cam @ R                                        # (R.T @ d).T = d @ R
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs = dirs / np.maximum(norm, 1e-12)
    valid = np.asarray(frame.depth_m, dtype=np.float64) > 0
    return dirs, valid


def directions_to_equirect(dirs: np.ndarray, width: int, height: int
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """世界系方向 → 等距柱状像素坐标。

    返回 `(u, v, el)`：`u` 已按 W 取模（自动处理 ±180° 接缝），
    `el` 是仰角（弧度），供权重使用。
    """
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    az = np.arctan2(y, x)                                   # [-π, π]
    el = np.arcsin(np.clip(z, -1.0, 1.0))                   # [-π/2, π/2]
    # 用 floor 而不是 round：保证映射到像素中心时是"包含左边界"的半开区间，
    # 这样 W 个像素刚好铺满 360°、不会有像素永远取不到（round 会漏掉边界）
    u = np.floor((az + np.pi) / (2.0 * np.pi) * width).astype(np.int64) % width
    v = np.floor((np.pi / 2.0 - el) / np.pi * height).astype(np.int64)
    v = np.clip(v, 0, height - 1)
    return u, v, el


def equirect_directions(width: int, height: int) -> np.ndarray:
    """全景图上每个像素对应的世界系方向（用于反查/验证投影）。"""
    u = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 * np.pi - np.pi
    v = (np.arange(height, dtype=np.float64) + 0.5) / height * np.pi
    az, el = u[None, :].repeat(height, axis=0), np.pi / 2.0 - v[:, None].repeat(width, axis=1)
    cos_el = np.cos(el)
    return np.stack([cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)], axis=-1)


# ==========================================================================
# 权重
# ==========================================================================
def _angle_weight(dirs: np.ndarray, forward: np.ndarray, *, power: float,
                  min_cos: float) -> np.ndarray:
    """视角中心权重：越靠近光轴权重越高（边缘畸变大、分辨率被拉伸）。

    `forward` 是该视角光轴在世界系的方向。权重取 `cos` 的 `power` 次幂；
    低于 `min_cos` 的像素直接丢弃（避免用极端边缘的拉伸像素参与融合）。
    """
    cos = np.clip(dirs @ forward, -1.0, 1.0)
    w = np.where(cos >= min_cos, cos ** power, 0.0)
    return w


# ==========================================================================
# 主接口
# ==========================================================================
def _prepare_view(frame, W: int, H: int, *, weight_power: float, min_cos: float,
                  max_depth: float):
    """把一帧整理成融合需要的扁平数组；没有可用像素时返回 None。

    返回 `(pix, wf, color_flat, depth_flat)`，都是**一维**（长度 = 该帧像素数）。
    """
    depth = np.asarray(frame.depth_m, dtype=np.float64)
    color = np.asarray(frame.color)
    if color.ndim == 2:                                      # 灰度图兜底
        color = np.repeat(color[:, :, None], 3, axis=2)
    if color.shape[:2] != depth.shape[:2]:
        return None
    ok = (depth > 0) & (depth <= float(max_depth))
    if not ok.any():
        return None

    dirs, _valid = view_rays_world(frame)
    forward = np.asarray(frame.pose.R, dtype=np.float64).reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    w = _angle_weight(dirs, forward, power=float(weight_power),
                      min_cos=float(min_cos)) * ok
    if not w.any():
        return None

    u, v, _el = directions_to_equirect(dirs, W, H)
    pix = (v.ravel() * W + u.ravel()).astype(np.int64)
    return (pix, w.ravel(),
            color.reshape(-1, 3).astype(np.float64),
            depth.ravel())


def fuse_to_equirect(
    frames: Sequence[Any],
    *,
    width: int = 2048,
    height: int = 1024,
    weight_power: float = 2.0,
    min_cos: float = 0.35,
    max_depth: float = 8.0,
    depth_outlier_m: float = 0.5,
    rgb_mode: str = "weighted_mean",
) -> Panorama:
    """把 N 个**真实**视角融合成一张等距柱状全景。

    Parameters
    ----------
    frames
        若干 `RGBDFrame`（各自带内参与位姿）。**顺序无关**。
    width, height
        全景分辨率（宽 : 高 = 2 : 1 是等距柱状的标准比例）。
    weight_power
        视角中心加权指数的幂；越大越偏向每帧的正前方像素。
    min_cos
        低于该余弦的像素丢弃（默认 0.35 ≈ 离光轴 69.5°）。
    max_depth
        超过该深度（米）的点视为不可靠，丢弃。
    depth_outlier_m
        两遍融合的离群阈值：第一遍算加权均值，第二遍丢掉偏离超过该值的样本
        （同一个全景像素可能被前景/背景同时覆盖，不剔除会把深度拉成"平均"）。
    rgb_mode
        `weighted_mean`（默认，抗噪）或 `nearest`（取权重最高的样本，最锐利）。

    Returns
    -------
    `Panorama`
    """
    if rgb_mode not in ("weighted_mean", "nearest"):
        raise ValueError(f"rgb_mode 只支持 weighted_mean / nearest，收到 {rgb_mode!r}")
    if not frames:
        raise ValueError("frames 为空，没有可融合的视角")

    H, W = int(height), int(width)
    rgb_sum = np.zeros((H, W, 3), dtype=np.float64)
    w_sum = np.zeros((H, W), dtype=np.float64)
    best_w = np.zeros((H, W), dtype=np.float64)
    best_rgb = np.zeros((H, W, 3), dtype=np.float64)
    best_d = np.zeros((H, W), dtype=np.float64)
    n_used = 0

    for fi, frame in enumerate(frames):
        prep = _prepare_view(frame, W, H, weight_power=weight_power,
                             min_cos=min_cos, max_depth=max_depth)
        if prep is None:
            logger.warn(f"第 {fi} 帧没有可用像素（权重门限 / 深度），跳过")
            continue
        pix, wf, color_flat, depth_flat = prep

        # ---- 累加（用 np.bincount 做逐像素归约，比 add.at 快得多）----
        for c in range(3):
            rgb_sum.reshape(-1, 3)[:, c] += np.bincount(
                pix, weights=(wf * color_flat[:, c]), minlength=H * W)[: H * W]
        w_sum += np.bincount(pix, weights=wf, minlength=H * W)[: H * W].reshape(H, W)

        # ---- "权重最高者胜"的候选（nearest 模式用；也保证边界像素有值）----
        np.maximum.at(best_w.reshape(-1), pix, wf)
        win = wf >= best_w.reshape(-1)[pix] - 1e-12
        if win.any():
            bp = pix[win]
            best_rgb.reshape(-1, 3)[bp] = color_flat[win]
            best_d.reshape(-1)[bp] = depth_flat[win]
        n_used += 1

    # ---- RGB ----
    has = w_sum > 0
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    if rgb_mode == "nearest":
        rgb[has] = np.clip(best_rgb[has], 0, 255).astype(np.uint8)
    else:
        rgb[has] = np.clip(rgb_sum[has] / w_sum[has][:, None], 0, 255).astype(np.uint8)

    # ---- 深度：以"权重最高的真实样本"为基准 + 共识融合 ----
    #
    # 为什么不用"先均值再剔离群"（我第一版就是这么写的，被测试抓出来了）：
    # 同一个全景像素被**前景 2 m 与背景 5 m** 同时覆盖时，加权均值是 3.5 m，
    # 而 3.5 与两个真实样本都相差 1.5 m → **两个样本都被当成离群剔掉**
    # → 什么都没剩下，只能回退到那个几何上根本不存在的 3.5 m。
    #
    # 正确做法是**先选一个真实样本当基准**（权重最高者 = 最靠近某台相机光轴、
    # 最可靠的那个观测），再只融合与它一致的样本。这样：
    #   · 绝不会在两张不相连的表面上取平均；
    #   · 仍然能靠多视角平均降噪（一致的那些样本会被融合）。
    depth_out = np.zeros((H, W), dtype=np.float32)
    if (w_sum > 0).any():
        # 基准 = 权重最高的样本（best_d 在累加阶段已按权重选出）
        ref = best_d.reshape(-1)
        d2_sum = np.zeros(H * W)
        d2_w = np.zeros(H * W)
        for frame in frames:
            prep = _prepare_view(frame, W, H, weight_power=weight_power,
                                 min_cos=min_cos, max_depth=max_depth)
            if prep is None:
                continue
            pix, wf, _color_flat, depth_flat = prep
            agree = np.abs(depth_flat - ref[pix]) <= float(depth_outlier_m)
            wf2 = np.where(agree, wf, 0.0)
            d2_sum += np.bincount(pix, weights=(wf2 * depth_flat), minlength=H * W)[: H * W]
            d2_w += np.bincount(pix, weights=wf2, minlength=H * W)[: H * W]
        good = d2_w > 0
        depth_out.reshape(-1)[good] = (d2_sum[good] / d2_w[good]).astype(np.float32)
        # 只有基准、没有别的样本与它一致时，就用基准本身（它是个真实观测）
        solo = (~good) & (best_w.reshape(-1) > 0)
        depth_out.reshape(-1)[solo] = ref[solo].astype(np.float32)

    return Panorama(rgb=rgb, depth_m=depth_out, weight=w_sum.astype(np.float32),
                    n_views=len(frames), n_used=n_used,
                    meta={"weight_power": weight_power, "min_cos": min_cos,
                          "max_depth": max_depth, "rgb_mode": rgb_mode,
                          "depth_outlier_m": depth_outlier_m})


def frame_for_panorama(pano: Panorama, *, frame_id: str = "panorama", pose=None):
    """把全景包成一个 `RGBDFrame`（内参按等距柱状的像素-角度关系构造）。

    这样下游 `MapBuilder` / 可视化**不需要任何改动**就能消费全景。

    ⚠️ 等距柱状投影**不是针孔投影**，所以这里给的内参是一种
    "等效针孔"近似（`fx = W / 2π`、`fy = H / π`、`cx = W/2`、`cy = H/2`），
    它只对**小角度范围**准确。要用全景做精确的三维反投影，
    应该用 `panorama_rays_world()` 直接取每个像素的世界方向（本模块提供），
    而不是走针孔反投影。这个区别很重要，所以写在这里。
    """
    from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame

    H, W = pano.rgb.shape[:2]
    K = CameraIntrinsics(fx=W / (2.0 * np.pi), fy=H / np.pi,
                         cx=W / 2.0, cy=H / 2.0, width=W, height=H)
    return RGBDFrame(color=pano.rgb, depth_m=pano.depth_m, intrinsics=K,
                     pose=(pose if pose is not None else CameraPose.identity()),
                     frame_id=frame_id,
                     meta={"source": "panorama_fusion", "n_views": pano.n_views,
                           "coverage": pano.coverage})


def panorama_rays_world(pano: Panorama, pose=None) -> np.ndarray:
    """全景图每个像素在世界系的方向（精确版，用于三维反投影）。"""
    H, W = pano.rgb.shape[:2]
    dirs = equirect_directions(W, H)
    if pose is not None:
        R = np.asarray(pose.R, dtype=np.float64).reshape(3, 3)
        # pose 是 world→camera；全景像素方向先在"全景相机系"里，再转世界
        dirs = dirs @ R.T
    return dirs
