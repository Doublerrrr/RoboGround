# -*- coding: utf-8 -*-
"""38 · 把整条链路的每一段都画出来（原始多视角 → 融合全景 → 物体在全景里的位置）。

产出四张图
=========
| 图 | 内容 | 回答什么问题 |
|---|---|---|
| `pipeline_1_inputs.png` | N 个**真实输入视角** + 各自光轴方位 | 输入到底长什么样、覆盖了哪些方向 |
| `pipeline_2_fusion.png` | **融合出的 360° 全景** + 覆盖/斜距统计 | 融合结果对不对、覆盖了多少 |
| `pipeline_3_objects.png` | 全景上叠加 **GT 框** 与 **地图输出物体** | 物体坐标落在全景的哪里、对不对得上 |
| `pipeline_4_topdown.png` | 俯视图：光心/各视角朝向/GT 框/地图物体 | 世界系里的几何关系 |

为什么值得画
==========
1. **它是"多视角真的是多视角"的直观证据** —— 能一眼看到 N 张不同朝向的真实照片，
   而不是把一张图重采样 N 次（本项目以前的做法，已删除）。
2. **物体坐标投回全景**是检验 3D→2D 一致性的最快手段：`MapBuilder` 输出的
   物体中心若在全景上落在对应实物上，说明反投影 + 体素融合 + 关联这一串都对。
   反之若系统性偏移，一眼就能看出来。
3. 图上同时标注**覆盖边界**（仰角范围），把"这只是水平环带、天顶天底没观测"
   这条限制画在脸上，避免看图的人误以为是全球面。

用法::

    python scripts/38_visualize_pipeline.py --room office_6 --views 12
    python scripts/38_visualize_pipeline.py --room hallway_6 --views 18 --max-views 48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config  # noqa: E402
from roboground.data.pano_scene import (  # noqa: E402
    PanoScene,
    load_scene,
    select_location,
    visible_gt,
)

DEFAULT_ROOT = Path(r"G:\2d3ds\area_1\area_1")
PROMPTS = ["chair", "table", "door", "bookcase", "sofa", "board",
           "window", "column", "beam", "clutter", "wall", "ceiling", "floor"]
#: 类别 → 颜色（挑高对比色，且与常见语义直觉一致）
CLASS_COLORS = {
    "chair": "#e6194B", "table": "#f58231", "door": "#4363d8",
    "bookcase": "#911eb4", "sofa": "#f032e6", "board": "#3cb44b",
    "window": "#42d4f4", "column": "#9A6324", "beam": "#808000",
    "clutter": "#800000", "wall": "#469990", "ceiling": "#a9a9a9",
    "floor": "#bcbd22", "stairs": "#000075",
}


def _setup_chinese_font() -> None:
    """中文字体：Windows 上 `Microsoft YaHei` 基本必然存在（与脚本 21 同一做法）。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Zen Hei"):
        try:
            matplotlib.font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            break
        except Exception:                                  # noqa: BLE001
            continue
    plt.rcParams["axes.unicode_minus"] = False              # 负号别被吃成方块


# ==========================================================================
# 世界系 → 全景像素（与 panorama.py 的等距柱状约定严格一致）
# ==========================================================================
def world_to_pano(scene: PanoScene, pts: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """世界点 → 全景连续像素坐标 `(u, v)` 与斜距 `r`。

    约定与 `panorama.equirect_directions` 一致：
    `az = atan2(dy, dx)`、`el = asin(dz / r)`、
    `u = (az + π) / 2π · W`、`v = (π/2 − el) / π · H`。
    返回的 `u` 落在 `[0, W]`、`v` 落在 `[0, H]`，可以直接配合
    `imshow(extent=[0, W, H, 0])` 使用（像素 i 横跨 [i, i+1]，中心 i+0.5）。
    """
    C = np.asarray(scene.center, dtype=np.float64).reshape(3)
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    d = p - C[None, :]
    r = np.linalg.norm(d, axis=1)
    safe = np.maximum(r, 1e-9)
    az = np.arctan2(d[:, 1], d[:, 0])
    el = np.arcsin(np.clip(d[:, 2] / safe, -1.0, 1.0))
    H, W = scene.panorama.rgb.shape[:2]
    u = (az + np.pi) / (2.0 * np.pi) * W
    v = (np.pi / 2.0 - el) / np.pi * H
    return u, v, r


def _aabb_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                    dtype=np.float64)


#: 立方体 12 条棱（用 8 个角点的下标表示）
_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
          (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def draw_aabb_equirect(ax, scene: PanoScene, lo: np.ndarray, hi: np.ndarray,
                       *, color: str, lw: float = 1.4, alpha: float = 1.0,
                       label: Optional[str] = None, samples: int = 24) -> None:
    """把一个世界系轴对齐框**画成等距柱状图上的线框**。

    做法：每条棱上采样若干点 → 逐个投到 `(u, v)` → 画折线。
    ★ 接缝处理：当相邻采样点的 `u` 跳变超过半个周期时**断开**，
    否则会在图上出现一条横穿整幅图的假线（这是等距柱状最经典的画错方式）。
    """
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    corners = _aabb_corners(lo, hi)
    H, W = scene.panorama.rgb.shape[:2]
    labeled = False
    for (i, j) in _EDGES:
        t = np.linspace(0.0, 1.0, int(samples))[:, None]
        pts = corners[i][None, :] * (1 - t) + corners[j][None, :] * t
        u, v, _ = world_to_pano(scene, pts)
        # 接缝断开
        jump = np.abs(np.diff(u)) > W / 2.0
        if jump.any():
            u = u.copy()
            u[1:][jump] = np.nan
        ax.plot(u, v, color=color, lw=lw, alpha=alpha,
                label=(label if not labeled else None))
        labeled = True


# ==========================================================================
# 图 1：原始输入视角
# ==========================================================================
def fig_inputs(scene: PanoScene, loc, out: Path, *, cols: int = 6) -> Path:
    from matplotlib import pyplot as plt

    ids = list(scene.meta.get("frame_ids") or [])
    n = len(ids)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for k, fid in enumerate(ids):
        ax = axes[k]
        fr = loc.frame(fid, resize=(320, 320))
        if fr is None:
            ax.axis("off")
            continue
        ax.imshow(fr.color)
        fwd = np.asarray(fr.pose.R).T @ np.array([0.0, 0.0, 1.0])
        az = np.degrees(np.arctan2(fwd[1], fwd[0]))
        el = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
        valid = float((np.asarray(fr.depth_m) > 0).mean() * 100)
        ax.set_title(f"#{fid}  az {az:+.0f}° / el {el:+.0f}°\n深度有效 {valid:.0f}%",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"{scene.room}：{n} 个「真实」输入视角（同一光心、不同朝向；"
        f"光心离散 {scene.panorama.meta.get('center_spread_m', 0):.1e} m）\n"
        f"这些是官方在真实采集位姿上渲染的像素，不是把一张图重采样 N 次",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ==========================================================================
# 图 2：融合出的全景
# ==========================================================================
def fig_fusion(scene: PanoScene, out: Path) -> Path:
    from matplotlib import pyplot as plt

    pano = scene.panorama
    H, W = pano.rgb.shape[:2]
    el_lo, el_hi = pano.elevation_span_deg()

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 1.5, 1.0], hspace=0.35)

    ax = fig.add_subplot(gs[0])
    ax.imshow(pano.rgb, extent=[0, W, H, 0])
    ax.set_title(
        f"{scene.room} 融合全景（等距柱状 {W}×{H}，{pano.n_used}/{pano.n_views} 个真实视角参与）",
        fontsize=10)
    ax.set_xticks(np.linspace(0, W, 9))
    ax.set_xticklabels([f"{int(a)}°" for a in np.linspace(-180, 180, 9)], fontsize=7)
    ax.set_yticks(np.linspace(0, H, 5))
    ax.set_yticklabels([f"{int(e)}°" for e in np.linspace(90, -90, 5)], fontsize=7)
    ax.set_xlabel("方位角 az", fontsize=8); ax.set_ylabel("仰角 el", fontsize=8)

    # 覆盖边界：把"只看到水平环带"直接画出来
    for y, lab in ((el_hi, f"覆盖上界 el={el_hi:+.0f}°"), (el_lo, f"覆盖下界 el={el_lo:+.0f}°")):
        yy = (np.pi / 2.0 - np.radians(y)) / np.pi * H
        ax.axhline(yy, color="w", lw=1.4, ls="--", alpha=0.9)
        ax.text(6, yy - 8, lab, color="w", fontsize=7,
                bbox=dict(fc="k", alpha=0.45, pad=1.2, lw=0))

    ax2 = fig.add_subplot(gs[1])
    rng = np.ma.masked_where(pano.depth_m <= 0, pano.depth_m)
    im = ax2.imshow(rng, extent=[0, W, H, 0], cmap="turbo", vmin=0.4,
                    vmax=float(np.percentile(pano.depth_m[pano.depth_m > 0], 99)))
    ax2.set_title("融合出的斜距（米，从光心起算；空白 = 该方向未被任何视角覆盖）",
                  fontsize=9)
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.01).ax.tick_params(labelsize=7)

    ax3 = fig.add_subplot(gs[2])
    el = np.degrees(np.pi / 2.0 - (np.arange(H) + 0.5) / H * np.pi)
    ax3.plot(el, (pano.depth_m > 0).mean(axis=1) * 100, color="#1f77b4", lw=1.5)
    ax3.axvline(el_hi, color="r", ls="--", lw=1); ax3.axvline(el_lo, color="r", ls="--", lw=1)
    ax3.set_xlim(90, -90)
    ax3.set_xlabel("仰角 el（度）", fontsize=8)
    ax3.set_ylabel("该行覆盖率 %", fontsize=8)
    ax3.set_title(
        f"逐仰角覆盖率：像素覆盖 {pano.coverage*100:.1f}% / "
        f"立体角覆盖 {pano.solid_angle_coverage*100:.1f}%；"
        f"「天顶与天底为 0」—— 采集是水平环带（实测 3,345 个真实视角的光轴都在 ±23° 内）",
        fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=7)

    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ==========================================================================
# 图 3：全景上的物体（GT 框 + 地图输出）
# ==========================================================================
def verify_projection(scene: PanoScene, map_objects: Sequence[Any]) -> Dict[str, Any]:
    """★ 用**数字**验证"物体坐标画在全景上是对的"，而不是靠肉眼看图。

    做法：把地图物体的中心投到全景像素 `(u, v)`，再读出**全景在该像素实测的斜距**
    `r_obs`，与投影算出的 `r_pred = |center − C|` 比较。

    · 若两者接近 ⇒ 3D 物体位置与该方向的真实几何一致，画上去必然落在实物上；
    · 若系统性偏大/偏小 ⇒ 反投影、体素融合或坐标系约定有错，图上看着"差不多"
      也可能整体偏了一大截。

    这条检查是本脚本真正的验收标准 —— 图好不好看是次要的。
    """
    W = scene.panorama.rgb.shape[1]
    H = scene.panorama.rgb.shape[0]
    rows: List[Dict[str, Any]] = []
    for ob in map_objects:
        c = np.asarray(ob.center, dtype=np.float64).reshape(3)
        u, v, r_pred = world_to_pano(scene, c[None, :])
        ui = int(np.clip(np.floor(u[0]), 0, W - 1))
        vi = int(np.clip(np.floor(v[0]), 0, H - 1))
        r_obs = float(scene.panorama.depth_m[vi, ui])
        covered = r_obs > 0
        # 物体中心通常落在物体**内部**，而全景看到的是物体**表面**，
        # 所以 r_obs 一般略小于 r_pred（相差不超过物体的半尺寸量级）。
        half = 0.5 * float(np.linalg.norm(
            np.asarray(ob.bbox_max, dtype=np.float64)
            - np.asarray(ob.bbox_min, dtype=np.float64)))
        rows.append({
            "obj_id": int(ob.obj_id), "label": str(ob.label),
            "u": float(u[0]), "v": float(v[0]),
            "r_pred_m": float(r_pred[0]),
            "r_obs_m": r_obs, "covered": bool(covered),
            "diff_m": (r_obs - float(r_pred[0])) if covered else None,
            "half_size_m": half,
        })
    cov = [r for r in rows if r["covered"]]
    diffs = np.array([abs(r["diff_m"]) for r in cov]) if cov else np.zeros(0)
    within_half = sum(1 for r in cov if abs(r["diff_m"]) <= max(r["half_size_m"], 0.25))
    return {
        "n_objects": len(rows),
        "n_covered": len(cov),
        "covered_frac": (len(cov) / len(rows)) if rows else 0.0,
        "abs_diff_median_m": float(np.median(diffs)) if diffs.size else None,
        "abs_diff_p90_m": float(np.percentile(diffs, 90)) if diffs.size else None,
        "within_half_size_or_25cm": f"{within_half}/{len(cov)}" if cov else "0/0",
        "per_object": rows,
    }


def fig_objects(scene: PanoScene, map_objects: Sequence[Any], out: Path,
                *, max_range: float) -> Path:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    pano = scene.panorama
    H, W = pano.rgb.shape[:2]
    C = np.asarray(scene.center, dtype=np.float64).reshape(3)
    vis = visible_gt(scene, max_range_m=max_range)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12.5))
    ext = [0, W, H, 0]

    # ---- A) GT 框 ----
    ax = axes[0]
    ax.imshow(pano.rgb, extent=ext)
    used = {}
    for o in vis:
        cls = o["label"]
        col = CLASS_COLORS.get(cls, "#ff0000")
        b = o["box"]
        lo, hi = b[:3] - b[3:6] / 2.0, b[:3] + b[3:6] / 2.0
        draw_aabb_equirect(ax, scene, lo, hi, color=col, lw=1.2, alpha=0.95)
        u, v, _ = world_to_pano(scene, b[:3][None, :])
        ax.plot(u, v, "o", color=col, ms=3.5)
        ax.annotate(f"{cls} {o['range_m']:.1f}m", (u[0], v[0]), color=col,
                    fontsize=6.5, xytext=(3, -7), textcoords="offset points")
        used[cls] = col
    ax.set_title(f"① GT 物体框投到全景上（可见 {len(vis)} 个，范围 ≤{max_range:.0f} m；"
                 f"轴对齐盒；已按遮挡剔除看不见的）", fontsize=9.5)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in sorted(used.items())],
              loc="upper right", fontsize=6.5, ncol=4, framealpha=0.75)
    ax.set_xticks([]); ax.set_yticks([])

    # ---- B) 地图输出物体 ----
    ax = axes[1]
    ax.imshow(pano.rgb, extent=ext)
    uu, vv = [], []
    for ob in map_objects:
        c = np.asarray(ob.center, dtype=np.float64).reshape(3)
        u, v, r = world_to_pano(scene, c[None, :])
        col = CLASS_COLORS.get(str(ob.label), "#ff0000")
        ax.plot(u, v, "*", color=col, ms=11, mec="k", mew=0.6)
        ax.annotate(f"{ob.label} ({r[0]:.1f}m)", (u[0], v[0]), color="k",
                    fontsize=6.5, xytext=(5, 4), textcoords="offset points",
                    bbox=dict(fc="w", alpha=0.62, pad=0.9, lw=0))
        lo = np.asarray(ob.bbox_min, dtype=np.float64)
        hi = np.asarray(ob.bbox_max, dtype=np.float64)
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo):
            draw_aabb_equirect(ax, scene, lo, hi, color=col, lw=0.9, alpha=0.55)
        uu.append(u[0]); vv.append(v[0])
    ax.set_title(f"② 「地图输出的物体」投回全景（{len(map_objects)} 个，★ = 物体中心，"
                 f"细线 = 地图里的 3D 包围盒）—— 星星应落在对应实物上", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])

    # ---- C) 同上，但底图换成斜距，便于判断"是不是真的落在那个表面上" ----
    ax = axes[2]
    rng = np.ma.masked_where(pano.depth_m <= 0, pano.depth_m)
    im = ax.imshow(rng, extent=ext, cmap="turbo", vmin=0.4,
                   vmax=float(np.percentile(pano.depth_m[pano.depth_m > 0], 99)))
    for ob in map_objects:
        c = np.asarray(ob.center, dtype=np.float64).reshape(3)
        u, v, _ = world_to_pano(scene, c[None, :])
        ax.plot(u, v, "*", color="w", ms=11, mec="k", mew=0.7)
    ax.set_title("③ 同 ② 但底图为斜距：星星处的斜距应与物体真实距离一致"
                 "（若系统性偏大/偏小，说明 3D 反投影或体素融合有问题）", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).ax.tick_params(labelsize=7)

    fig.suptitle(f"{scene.room}：物体坐标在全景里的位置（光心 {np.round(C, 2)}）",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ==========================================================================
# 图 4：俯视图
# ==========================================================================
def fig_topdown(scene: PanoScene, map_objects: Sequence[Any], out: Path,
                *, max_range: float) -> Path:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    C = np.asarray(scene.center, dtype=np.float64).reshape(3)
    vis = visible_gt(scene, max_range_m=max_range)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ---- 左：全景覆盖在水平面上的投影（"看到的方向"）+ 物体 ----
    ax = axes[0]
    cov = scene.panorama.depth_m > 0
    az_grid = (np.arange(scene.panorama.rgb.shape[1]) + 0.5) / \
        scene.panorama.rgb.shape[1] * 360.0 - 180.0
    ring = cov.mean(axis=0)
    for a, c in zip(az_grid, ring):
        if c <= 0:
            continue
        th = np.radians(a)
        ax.plot([0, np.cos(th) * max_range], [0, np.sin(th) * max_range],
                color=plt.cm.viridis(c), lw=2.2, alpha=0.85, solid_capstyle="butt")
    used = {}
    for o in vis:
        b = o["box"]
        col = CLASS_COLORS.get(o["label"], "#ff0000")
        ax.add_patch(Rectangle((b[0] - b[3] / 2, b[1] - b[4] / 2), b[3], b[4],
                               fill=False, ec=col, lw=1.0, alpha=0.9))
        used[o["label"]] = col
    for ob in map_objects:
        c = np.asarray(ob.center, dtype=np.float64).reshape(3)
        ax.plot(c[0] - C[0], c[1] - C[1], "*", ms=13, color="k", mec="w", mew=0.7)
    ax.plot(0, 0, "o", ms=9, color="w", mec="k", mew=1.6)
    ax.annotate("光心（所有视角共享）", (0, 0), fontsize=8, xytext=(6, -14),
                textcoords="offset points")
    ax.set_aspect("equal")
    lim = max_range * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.25)
    ax.set_xlabel("Δx (m)"); ax.set_ylabel("Δy (m)")
    ax.set_title("俯视图（以光心为原点）：\n"
                 "放射线 = 全景各方位是否有覆盖（颜色越亮覆盖越好）；\n"
                 "方框 = GT 物体，★ = 地图输出物体", fontsize=9)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in sorted(used.items())],
              loc="upper right", fontsize=6.5, ncol=2, framealpha=0.75)

    # ---- 右：各输入视角的位置与朝向 ----
    ax = axes[1]
    ids = list(scene.meta.get("frame_ids") or [])
    for fid in ids:
        try:
            R = np.asarray(loc_R(loc=scene, fid=fid), dtype=np.float64)
        except Exception:                                  # noqa: BLE001
            continue
        fwd = R.T @ np.array([0.0, 0.0, 1.0])
        th = np.arctan2(fwd[1], fwd[0])
        ax.arrow(0, 0, np.cos(th) * max_range * 0.5, np.sin(th) * max_range * 0.5,
                 head_width=0.06 * max_range, length_includes_head=True,
                 color="#1f77b4", alpha=0.5, lw=0.7)
    for o in vis:
        b = o["box"]
        col = CLASS_COLORS.get(o["label"], "#ff0000")
        ax.add_patch(Rectangle((b[0] - b[3] / 2, b[1] - b[4] / 2), b[3], b[4],
                               fill=False, ec=col, lw=1.0, alpha=0.9))
    ax.plot(0, 0, "o", ms=9, color="w", mec="k", mew=1.6)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.25)
    ax.set_xlabel("Δx (m)"); ax.set_ylabel("Δy (m)")
    ax.set_title(f"同上的俯视图 + {len(ids)} 个输入视角的「光轴方向」（蓝箭头）\n"
                 "箭头绕满一圈，说明确实是原地转一圈的多视角，而不是同一方向拍多次",
                 fontsize=9)

    fig.suptitle(f"{scene.room}：世界系俯视图（坐标已平移到光心）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def loc_R(loc: PanoScene, fid: int) -> np.ndarray:
    """取某个输入视角的 world→camera 旋转（供俯视图画箭头）。"""
    return loc.meta["_loc"].pose(fid).R


# ==========================================================================
# 建图（与脚本 37 同样的注入方式：用 GT 框当检测器，只评 2D→3D→地图这一段）
# ==========================================================================
def build_map(cfg, scene: PanoScene, max_range: float):
    from roboground.mapping import MapBuilder
    from roboground.types import Detection2D

    vis = visible_gt(scene, max_range_m=max_range)
    dets = []
    for o in vis:
        for (u0, v0, u1, v1) in o.get("uv_parts") or [o["uv"]]:
            dets.append(Detection2D(label=str(o["label"]), score=1.0,
                                    bbox=np.array([u0, v0, u1, v1], dtype=np.float64),
                                    prompt=str(o["label"])))
    frame = scene.frame()
    b = MapBuilder(cfg, prompts=sorted({d.label for d in dets}) or PROMPTS)
    b.pipeline.run = lambda f, prompts=None: dets          # type: ignore[assignment]
    smap = b.build_from_frames([frame])
    return smap, len(dets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--room", default="office_6")
    ap.add_argument("--views", type=int, default=12, help="图 1 里展示多少个输入视角")
    ap.add_argument("--max-views", type=int, default=32, help="最多用多少个视角做融合")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--json", type=Path, default=Path("runs/38_visualize_pipeline.json"),
                    help="验收数字落盘位置（与图一起构成可核对的产物）")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[FAIL] 数据不在 {args.root}")
        return 2

    _setup_chinese_font()
    args.outdir.mkdir(parents=True, exist_ok=True)

    loc = select_location(args.root, room=args.room, min_frames=8)
    print(f"[1/5] 采集点 {loc.uuid[:12]} / {loc.room}，可用视角 {len(loc.frame_ids)}")
    scene = load_scene(args.root, loc, width=args.width, height=args.height,
                       max_frames=args.max_views, max_depth=args.max_range)
    # 供俯视图取各视角朝向（load_scene 不保留 Location 对象）
    scene.meta["_loc"] = loc
    print(f"      融合 {scene.frames_used} 个视角，像素覆盖 {scene.coverage*100:.1f}%，"
          f"立体角覆盖 {scene.panorama.solid_angle_coverage*100:.1f}%")

    # 图 1 用均匀抽稀的一小批视角，避免网格过大
    all_ids = list(loc.frame_ids)
    n = min(int(args.views), len(all_ids))
    show_ids = [all_ids[i] for i in np.unique(np.linspace(0, len(all_ids) - 1, n).round().astype(int))]
    scene.meta["frame_ids"] = show_ids

    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("project.verbose", False)

    print("[2/5] 建图（用 GT 框注入当检测器，评的是 2D→3D→地图这一段）")
    smap, n_det = build_map(cfg, scene, args.max_range)
    print(f"      注入检测 {n_det} 个 → 地图物体 {smap.num_objects} 个 / "
          f"体素 {smap.num_voxels}")

    # ---- ★ 数字验收：物体坐标画在全景上到底对不对 ----
    chk = verify_projection(scene, smap.objects)
    print("\n[验收] 物体中心投回全景后，与全景实测斜距的一致性：")
    print(f"      物体 {chk['n_objects']} 个，其中投影落在**有覆盖**方向的 "
          f"{chk['n_covered']} 个（{chk['covered_frac']*100:.0f}%）")
    if chk["abs_diff_median_m"] is not None:
        print(f"      |实测斜距 − 物体距离| 中位 {chk['abs_diff_median_m']:.4f} m，"
              f"p90 {chk['abs_diff_p90_m']:.4f} m")
        print(f"      落在『物体半尺寸或 25 cm』内的：{chk['within_half_size_or_25cm']}")
    else:
        print("      [WARN] 没有一个物体的投影落在有覆盖的方向上，图的正确性存疑")

    outs = []
    print("\n[3/5] 图 1：原始输入视角")
    outs.append(fig_inputs(scene, loc, args.outdir / "pipeline_1_inputs.png"))
    print("[4/5] 图 2：融合全景")
    outs.append(fig_fusion(scene, args.outdir / "pipeline_2_fusion.png"))
    print("[5/5] 图 3/4：物体位置")
    outs.append(fig_objects(scene, smap.objects, args.outdir / "pipeline_3_objects.png",
                            max_range=args.max_range))
    outs.append(fig_topdown(scene, smap.objects, args.outdir / "pipeline_4_topdown.png",
                            max_range=args.max_range))

    # 把验收数字与图一起落盘（"图 + 数"才是可核对的产物）
    import json
    meta = {
        "room": scene.room, "uuid": scene.uuid,
        "frames_used": scene.frames_used, "n_frames_available": scene.n_frames_available,
        "width": args.width, "height": args.height, "max_range_m": args.max_range,
        "pixel_coverage": scene.coverage,
        "solid_angle_coverage": scene.panorama.solid_angle_coverage,
        "elevation_span_deg": scene.panorama.elevation_span_deg(),
        "n_gt_visible": len(visible_gt(scene, max_range_m=args.max_range)),
        "n_map_objects": int(smap.num_objects), "n_voxels": int(smap.num_voxels),
        "projection_check": {k: v for k, v in chk.items() if k != "per_object"},
        "figures": [str(o) for o in outs],
    }
    js = Path(args.json)
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {js}")

    print("\n产物：")
    for o in outs:
        print(f"  {o}  ({o.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
