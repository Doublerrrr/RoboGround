#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""26 · 多视角可视化：把「单帧 → 虚拟相机重渲染 → 建图」这条链路画出来。

为什么需要这个脚本
================
项目文档里有一句"真实 + 虚拟相机 4 视角，定位误差 0.262 m（−52%）"。
这句话**容易被读成"我们有 4 个真实视角"**，但实际上：

    SUN RGB-D 每个场景**只有一张** RGB-D 图。
    所谓"4 视角"是：把这一张图反投影成点云，再在新视角**重渲染**出来。

本脚本把这件事**画出来并量化**，让读者一眼看到：
  1. 真实输入只有一帧（唯一的真实观测）；
  2. 新视角是从这**同一份点云**渲染的（不是新采集的信息）；
  3. 新视角有**遮挡空洞**（原视角看不到的区域没有点）—— 量化成比例。

产出：`runs/vis_multiview_real.png` 与 `runs/vis_multiview_output.png`

用法::

    python scripts/26_visualize_multiview.py --scene 1236 --views 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import FancyArrow                                     # noqa: E402

from roboground.config import load_config                            # noqa: E402
from roboground.utils.logging import get_logger, set_verbosity       # noqa: E402

log = get_logger("vis.multiview")


# --------------------------------------------------------------------------
# 中文字体：Windows 上优先用雅黑/黑体；拿不到就退回英文标签
# --------------------------------------------------------------------------
def setup_font() -> bool:
    """配置 matplotlib 中文字体；返回是否成功（False 表示要用英文标签）。"""
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in have:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            log.info(f"使用中文字体：{name}")
            return True
    log.warn("没找到中文字体，图内标签改用英文")
    return False


def L(zh: str, en: str, zh_ok: bool) -> str:
    return zh if zh_ok else en


# --------------------------------------------------------------------------
def load_one_scene(cfg, idx: int, max_side: int):
    """按场景下标加载一个 SUN RGB-D 场景（可指定长边缩放，加快渲染）。"""
    from roboground.data.sunrgbd import load_scene_index, load_sunrgbd_scene

    index_path = str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz")
    index = load_scene_index(index_path)

    scene = load_sunrgbd_scene(index, idx, max_depth=8.0)
    if max_side and max(scene.shape) > max_side:
        h, w = scene.shape
        s = max_side / float(max(h, w))
        scene = load_sunrgbd_scene(index, idx, max_depth=8.0,
                                  resize=(int(round(w * s)), int(round(h * s))))
    return scene


def draw_boxes_2d(ax, boxes, labels, *, zh_ok: bool, max_n: int = 12):
    """在俯视图（x-y）上画 GT 框的矩形投影 + 中心点。"""
    for b, lab in list(zip(boxes, labels))[:max_n]:
        cx, cy, _cz, dx, dy, _dz = (float(v) for v in b[:6])
        yaw = float(b[6]) if boxes.shape[1] >= 7 else 0.0
        # 按 yaw 旋转的矩形四角
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        corners = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2],
                            [-dx / 2, -dy / 2], [-dx / 2, dy / 2]])
        pts = (R @ corners.T).T + np.array([cx, cy])
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.2, color="#e4572e")
        ax.text(cx, cy, str(lab)[:9], fontsize=6, color="#7a1f0f",
                ha="center", va="bottom")


def render_view_metrics(frame) -> float:
    """返回该视角的**遮挡空洞率**（深度为 0 的像素占比）。"""
    d = np.asarray(frame.depth_m)
    return float((d <= 0).mean())


# --------------------------------------------------------------------------
# 图 1：真实输入 + 虚拟相机怎么造的
# --------------------------------------------------------------------------
def figure_inputs(scene, frames, poses, target, out_path: Path, zh_ok: bool):
    from roboground.geometry.projection import frame_to_pointcloud

    base = scene.to_frame()
    pts, cols = frame_to_pointcloud(base, min_depth=0.2, max_depth=8.0,
                                    max_points=60_000, colors=True, seed=0)
    idx = np.random.default_rng(0).choice(pts.shape[0], size=min(20000, pts.shape[0]),
                                         replace=False)
    P = pts[idx]

    n = len(frames)
    fig = plt.figure(figsize=(15, 3.3 + 2.4 * ((n + 2) // 3)))
    gs = fig.add_gridspec(3 + (n + 2) // 3, 3, hspace=0.35, wspace=0.18)

    # --- (1) 真实 RGB ---
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(scene.color)
    ax.set_title(L("① 真实输入：SUN RGB-D 的**唯一一帧** RGB", "1. Real input: the ONLY RGB frame", zh_ok),
                 fontsize=9)
    ax.axis("off")

    # --- (2) 真实深度 ---
    ax = fig.add_subplot(gs[0, 1])
    dm = np.asarray(scene.depth_m)
    ax.imshow(np.where(dm > 0, dm, np.nan), cmap="turbo")
    ax.set_title(L("② 对应深度图（这就是全部真实观测）",
                   "2. Its depth (all the real observation there is)", zh_ok), fontsize=9)
    ax.axis("off")

    # --- (3) 点云俯视图 + 虚拟相机布局 ---
    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(P[:, 0], P[:, 1], c=P[:, 2], s=0.6, cmap="viridis")
    plt.colorbar(sc, ax=ax, fraction=0.046, label=L("高度 z (m)", "height z (m)", zh_ok))
    cx, cy = float(target[0]), float(target[1])
    ax.plot([cx], [cy], "k+", ms=10, mew=2,
            label=L("环绕中心", "orbit center", zh_ok))
    for i, pose in enumerate(poses):
        eye = -pose.R.T @ pose.t                       # 相机在 map 中的位置
        ax.plot([eye[0]], [eye[1]], "o", color="#e4572e", ms=6)
        ax.annotate("", xy=(cx, cy), xytext=(eye[0], eye[1]),
                    arrowprops=dict(arrowstyle="->", color="#e4572e", lw=1.0, alpha=0.8))
        ax.text(eye[0], eye[1], f" v{i}", fontsize=6, color="#e4572e")
    ax.plot([0], [0], "ks", ms=5, label=L("原始相机位置", "original camera", zh_ok))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(L("③ 点云俯视图 + 虚拟相机环绕位置（全部是渲染出来的视角）",
                   "3. Cloud top-view + virtual camera orbit (all views are rendered)", zh_ok),
                 fontsize=9)
    ax.legend(fontsize=6, loc="best")

    # --- 逐视角 RGB + 深度 ---
    r = 1
    for i, fr in enumerate(frames):
        row = r + i // 3
        col = i % 3
        hole = render_view_metrics(fr)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(fr.color)
        ax.set_title(L(f"虚拟视角 v{i}（RGB）　空洞 {hole * 100:.1f}%",
                       f"virtual v{i} (RGB)  holes {hole * 100:.1f}%", zh_ok), fontsize=8)
        ax.axis("off")

    fig.suptitle(L(
        f"SUN RGB-D 场景「{scene.sequence}」：1 张真实图 → {n} 个虚拟视角（点云重渲染，非真实多视角）",
        f"SUN RGB-D scene '{scene.sequence}': 1 real frame -> {n} rendered views", zh_ok),
        fontsize=11, y=0.995)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"已保存 {out_path}")


# --------------------------------------------------------------------------
# 图 2：输出 —— 单视角 vs 多视角融合
# --------------------------------------------------------------------------
def figure_output(scene, cfg, out_path: Path, max_views: int, zh_ok: bool):
    from roboground.data.virtual_camera import scene_sequence_from_cloud
    from roboground.eval.benchmark import map_level_localization
    from roboground.mapping import MapBuilder

    prompts = sorted({str(x).lower() for x in scene.labels})
    gt = scene.boxes_3d
    labels = list(scene.labels)

    seq = scene_sequence_from_cloud(scene, num_frames=max_views, radius=1.6,
                                    max_points=120_000, splat=2)
    per_view_err, fused_err = [], []
    maps = {}
    for k in range(1, max_views + 1):
        b = MapBuilder(cfg, prompts=prompts)
        smap = b.build_from_frames(seq[:k])
        maps[k] = smap
        # 与 GT 的定位对比（按标签匹配，避免"找最近"虚报）
        errs = []
        for o in smap.objects:
            cand = [np.linalg.norm(np.asarray(o.center) - np.asarray(g[:3]))
                    for g, l in zip(gt, labels) if str(l).lower() == str(o.label).lower()]
            if cand:
                errs.append(min(cand))
        fused_err.append(float(np.median(errs)) if errs else float("nan"))
        loc = map_level_localization(smap, [gt])
        per_view_err.append(loc)

    # 现实对照：**真实**那一帧本身的无效深度占比。
    # 有了这个分母，"虚拟视角 77% 空洞"才有意义 ——
    # 真实的 SUN RGB-D 帧本来就有约 4 成像素无效（超出量程/吸光材质/边界），
    # 重渲染把它抬高到约 8 成，即**丢掉了一半以上的可用像素**。
    real_hole = float((np.asarray(scene.depth_m) <= 0).mean())

    fig = plt.figure(figsize=(14, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.25)

    for ax_i, k in ((0, 1), (1, max_views)):
        ax = fig.add_subplot(gs[0, ax_i])
        smap = maps[k]
        for o in smap.objects:
            c = np.asarray(o.center)
            vids = np.asarray(getattr(o, "voxel_ids", []), dtype=np.int64)
            grid = getattr(smap, "voxel_grid", None)
            if grid is not None and vids.size and hasattr(grid, "centers"):
                cc = np.asarray(grid.centers)
                if cc.shape[0] > int(vids.max()):
                    vv = cc[vids]
                    ax.scatter(vv[:, 0], vv[:, 1], s=2, alpha=0.5)
            ax.plot([c[0]], [c[1]], "k+", ms=9, mew=2)
            ax.text(c[0], c[1], str(o.label)[:8], fontsize=6)
        draw_boxes_2d(ax, gt, labels, zh_ok=zh_ok)
        ax.set_aspect("equal")
        ax.set_title(L(f"{k} 帧建图：{smap.num_objects} 个物体",
                       f"map from {k} frames: {smap.num_objects} objects", zh_ok), fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = fig.add_subplot(gs[0, 2])
    ks = list(range(1, max_views + 1))
    med = [e.get("median_error_m", e.get("median_m", np.nan)) if isinstance(e, dict) else np.nan
           for e in per_view_err]
    ax.plot(ks, med, "o-", color="#3b6ea5",
            label=L("地图级中位误差（按标签匹配）", "map-level median error", zh_ok))
    ax.plot(ks, fused_err, "s--", color="#e4572e",
            label=L("同名物体最小误差中位", "median per-object error", zh_ok))
    ax.set_xlabel(L("参与融合的视角数", "#views fused", zh_ok))
    ax.set_ylabel(L("误差 (m)", "error (m)", zh_ok))
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title(L("融合视角数 vs 定位误差", "#views vs localization error", zh_ok), fontsize=9)
    ax.text(0.02, 0.03,
            L(f"参考：真实帧无效深度 {real_hole * 100:.1f}%",
              f"ref: real frame invalid depth {real_hole * 100:.1f}%", zh_ok),
            transform=ax.transAxes, fontsize=7, color="#555")

    fig.suptitle(L(
        "输出：同一份点云渲染出的多视角，视角越多越稳（但并未引入新信息）",
        "Output: more rendered views -> more stable (but no NEW information)", zh_ok),
        fontsize=11, y=1.02)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"已保存 {out_path}")
    return per_view_err, fused_err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=int, default=1236,
                    help="SUN RGB-D 场景下标（索引里的位置）")
    ap.add_argument("--views", type=int, default=6, help="虚拟视角数")
    ap.add_argument("--max-side", type=int, default=320, help="图像长边上限（加速）")
    ap.add_argument("--out-dir", default="runs", help="图片输出目录")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.quiet:
        set_verbosity(0)
    zh_ok = setup_font()
    cfg = load_config()
    cfg.set("perception.detector", "stub")
    cfg.set("perception.segmenter", "box")
    cfg.set("perception.encoder", "color_hist")
    cfg.set("project.verbose", False)

    scene = load_one_scene(cfg, args.scene, args.max_side)
    print("=" * 78)
    print(f"场景 #{args.scene}  {scene.sequence}")
    print(f"  图像 {scene.color.shape} / 有效深度 {int((scene.depth_m > 0).sum())} px")
    print(f"  GT：{len(scene.labels)} 个物体  {sorted(set(map(str, scene.labels)))}")
    print("  ★ 这是**一帧** RGB-D —— 数据集里同一个场景没有第二个视角")
    print("=" * 78)

    from roboground.data.virtual_camera import orbit_poses, scene_sequence_from_cloud

    base = scene.to_frame()
    target = scene.boxes_3d[:, :3].mean(axis=0) if scene.boxes_3d.shape[0] else np.zeros(3)
    poses = orbit_poses(target, radius=1.6, num=args.views,
                        start_deg=-25.0, end_deg=25.0)

    frames = scene_sequence_from_cloud(scene, num_frames=args.views, radius=1.6,
                                       max_points=120_000, splat=2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_inputs(scene, frames, poses, target, out_dir / "vis_multiview_real.png", zh_ok)

    holes = [render_view_metrics(f) for f in frames]
    real_hole = float((np.asarray(scene.depth_m) <= 0).mean())
    print("\n遮挡空洞率（深度为 0 的像素占比）：")
    print(f"  ★ 真实那一帧本身 ： {real_hole * 100:5.1f}%   ← 分母（真实传感器也会有无效像素）")
    for i, h in enumerate(holes):
        print(f"    虚拟视角 v{i}    ： {h * 100:5.1f}%")
    print(f"  虚拟视角平均 {np.mean(holes) * 100:.1f}%"
          f"（比真实帧多丢 {(np.mean(holes) - real_hole) * 100:.1f} 个百分点）")
    print("  ← 多出来的空洞来自**原视角看不到的区域**：那里本来就没有点云，"
          "重渲染补不上")

    per_view, fused = figure_output(scene, cfg, out_dir / "vis_multiview_output.png",
                                    args.views, zh_ok)
    print("\n融合视角数 → 误差：")
    for k in range(1, args.views + 1):
        e = fused[k - 1]
        print(f"  {k} 视图：同名物体误差中位 {e:.3f} m" if np.isfinite(e) else f"  {k} 视图：无匹配")
    print("\n★ 结论：视角越多定位越稳，但**信息量没有增加** —— "
          "所有视角都来自同一张真实图的点云。")
    print("  要验证真正的多视角，需要数据集中有同一场景的多个真实视点"
          "（例如移动相机轨迹）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
