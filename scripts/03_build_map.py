#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""03 · 用真实数据建图（SUN RGB-D 单场景 / 虚拟相机多视角）。

两条数据路径
-----------
1. `--source sunrgbd`：用 SUN RGB-D 的**真实单帧** RGB-D 建图（一张图一个视角）；
2. `--source virtual`：把 SUN RGB-D 场景的点云用**虚拟相机**重渲染成多视角序列，
   用来验证"多视角融合"这条链路（因为原始数据每个场景只有一帧）。

用法::

    # 真实单帧建图（挑一个物体多的场景）
    python scripts/03_build_map.py --source sunrgbd --save runs/map_sunrgbd.npz

    # 虚拟相机多视角（6 个视角，绕场景中心环绕）
    python scripts/03_build_map.py --source virtual --views 6 --save runs/map_virtual.npz

    # 指定 prompts（开放词汇的核心：换 prompt 就换"要找什么"）
    python scripts/03_build_map.py --prompts chair table monitor door window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboground.config import load_config                             # noqa: E402
from roboground.data.sunrgbd import (                                 # noqa: E402
    SUNRGBDDataset,
    load_scene_index,
    load_sunrgbd_scene,
)
from roboground.data.virtual_camera import scene_sequence_from_cloud  # noqa: E402
from roboground.mapping import MapBuilder                             # noqa: E402
from roboground.reasoning import RuleEngine                           # noqa: E402
from roboground.utils.logging import get_logger                       # noqa: E402

DEFAULT_PROMPTS = ["chair", "table", "desk", "monitor", "door", "window",
                   "bookshelf", "trash can", "box", "bottle", "sofa", "bed"]


def pick_scene(index, *, min_boxes: int = 5, min_classes: int = 3,
               wanted: list | None = None, max_scan: int = 4000):
    """挑一个 GT 框多、类别丰富的场景（可选要求含指定类别）。"""
    counts = np.diff(np.asarray(index["box_offset"], dtype=np.int64))
    wanted_set = {w.lower() for w in (wanted or [])}

    for pos in np.argsort(-counts):
        i = int(pos)
        if i >= max_scan:
            continue
        off0, off1 = int(index["box_offset"][i]), int(index["box_offset"][i + 1])
        labels = [str(x) for x in np.asarray(index["label_flat"][off0:off1]).ravel()]
        if wanted_set and not (wanted_set & {l.lower() for l in labels}):
            continue
        if counts[i] < min_boxes or len(set(labels)) < min_classes:
            continue
        scene = load_sunrgbd_scene(index, i, max_depth=8.0)
        if scene is not None and scene.boxes_3d.shape[0] > 0:
            return i, scene
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sunrgbd", "virtual"], default="sunrgbd")
    ap.add_argument("--index", default=None, help="场景索引路径")
    ap.add_argument("--scene", type=int, default=None, help="指定场景下标（默认自动挑）")
    ap.add_argument("--views", type=int, default=6, help="虚拟相机的视角数（仅 virtual）")
    ap.add_argument("--radius", type=float, default=1.8, help="虚拟相机环绕半径（米）")
    ap.add_argument("--resize", type=int, default=480, help="长边缩放到该尺寸（加速）")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--voxel-size", type=float, default=None, help="覆盖体素边长（米）")
    ap.add_argument("--save", default="runs/map.npz")
    ap.add_argument("--list", action="store_true", help="只列出物体清单，不建图")
    args = ap.parse_args()

    log = get_logger("build_map")
    cfg = load_config()
    prompts = list(args.prompts) if args.prompts else DEFAULT_PROMPTS
    cfg.set("perception.prompts", prompts)
    if args.voxel_size:
        cfg.set("geometry.voxel_size", float(args.voxel_size))

    index_path = args.index or str(Path(str(cfg.get("data.root", "data"))) / "cache" / "sunrgbd_index.npz")
    try:
        index = load_scene_index(index_path)
    except FileNotFoundError as exc:
        log.error(str(exc))
        log.info("请先运行：python scripts/02_build_sunrgbd_index.py")
        return 2

    log.info(f"prompts（决定「要找什么」）：{prompts}")

    # ---------------- 选场景 ----------------
    if args.scene is not None:
        scene = load_sunrgbd_scene(index, args.scene, max_depth=8.0)
        if scene is None:
            log.error(f"场景 {args.scene} 加载失败")
            return 2
        scene_idx = args.scene
    else:
        scene_idx, scene = pick_scene(index, wanted=prompts)
    if scene is None:
        log.error("没找到合适的场景（可放宽 --prompts 或检查数据）")
        return 2

    # ---------------- 缩放 ----------------
    if args.resize:
        h, w = scene.shape
        scale = args.resize / float(max(h, w))
        if scale < 1.0:
            scene = load_sunrgbd_scene(
                index, scene_idx, max_depth=8.0,
                resize=(int(round(w * scale)), int(round(h * scale))),
            )

    log.info(f"场景 #{scene_idx}：{scene.sequence}")
    log.info(f"  图像 {scene.color.shape}，有效深度 "
             f"{int((scene.depth_m > 0).sum())} px")
    log.info(f"  GT 类别：{sorted(set(scene.labels))}")

    # ---------------- 造帧序列 ----------------
    if args.source == "virtual":
        log.info(f"用虚拟相机生成 {args.views} 个视角（半径 {args.radius} m）")
        frames = scene_sequence_from_cloud(
            scene, num_frames=args.views, radius=args.radius,
            max_points=120_000, splat=2,
        )
    else:
        frames = [scene.to_frame()]

    # ---------------- 建图 ----------------
    log.info(f"建图：{len(frames)} 帧")
    builder = MapBuilder(cfg, prompts=prompts)
    smap = builder.build_from_frames(frames)

    log.kv("地图概况", {
        "观测数": smap.meta["num_observations"],
        "物体轨迹": smap.meta.get("object_mode"),
        "体素数": smap.num_voxels,
        "物体数": smap.num_objects,
        "体素边长(m)": smap.voxel_grid.voxel_size,
        "特征维度": smap.feature_dim,
    })

    print()
    print(smap.describe(max_objects=25))
    print()

    # ---------------- 与 GT 对比（定位质量）----------------
    from roboground.eval.benchmark import map_level_localization

    gt_boxes = [f.meta["boxes_3d"] for f in frames if "boxes_3d" in f.meta]
    if gt_boxes:
        loc = map_level_localization(smap, gt_boxes)
        from roboground.eval.metrics import format_metrics

        print("与 GT 的定位对比（地图级）：")
        print(format_metrics(loc))
        print()

    if args.list:
        return 0

    # ---------------- 查询示例 ----------------
    engine = RuleEngine(smap, cfg=cfg)
    print("示例问答：")
    for q in ["描述一下场景"] + [f"{lab} 在哪" for lab in smap.labels[:3]]:
        res = engine.answer(q)
        print(f"  Q: {q}\n  A: {res.answer}")
    print()

    path = smap.save(args.save)
    log.ok(f"地图已保存：{path}")
    log.info(f"接下来：python scripts/04_query_demo.py {path} \"桌子在哪\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
