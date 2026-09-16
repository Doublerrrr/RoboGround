# -*- coding: utf-8 -*-
"""2D-3D-S 采集点 → 全景场景（真实数据的**唯一**入口）。

这个模块把"数据集"和"项目其余部分"接起来，链路是：

    一个采集点（同一个 camera_uuid）
      └─ N 个真实视角（各自内参 + 位姿，**共享同一光心**）
           └─ fuse_to_equirect()  多视角 → 1 张 360° 全景（RGB + 斜距）
                └─ frame_for_panorama()  → RGBDFrame（精确等距柱状投影）
                     └─ MapBuilder → SemanticMap → 查询 / 语言接地

**这里没有任何"虚拟视角"**：每个全景像素都来自一个**真实采集的视角**，
信息量随视角数**真的增加**。覆盖率是**测出来**的（`Panorama.coverage`），
不是渲染出来的。

为什么不是"单帧造点云再重渲染"
============================
那样做只是把同一份观测重采样：新视角里原视角看不到的地方**没有点云**。
实测无效深度像素从 42.8% 涨到 77.0%（见 `docs/多视角数据核查报告.md`）。
本模块走的是相反方向 —— 把 N 个真实观测融合到同一个球面坐标系。

关键约定（细节见 `panorama.py` 与 `docs/2D3D-S数据集核验报告.md`）
==============================================================
· 位姿 `camera_rt_matrix` 是 **(3,4) = [R|t]**，语义是 **world→camera**
  （实测：`C = −Rᵀt` 与 `camera_location` 差 1.2e-06 m，反向假设差 38 m）。
  与项目 `CameraPose` 完全一致，**不需要转换**。
· 深度 `depth_m = raw / 512.0`，`raw == 65535` 是**无数据**（不是 128 米）。
· 全景的 `depth_m` 是**从光心起算的斜距**，不是针孔 z 深度。
· GT 的 `Bbox` 是**轴对齐**盒（yaw 恒 0），所以**不能用有向 3D IoU 评**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.data.frame_gt import box_surface_points
from roboground.data.panorama import Panorama, frame_for_panorama, fuse_to_equirect
from roboground.data.stanford2d3d import (
    Location,
    list_locations,
    load_objects,
    objects_as_boxes,
)
from roboground.utils.logging import get_logger

logger = get_logger("data.pano_scene")


# ==========================================================================
# 场景容器
# ==========================================================================
@dataclass
class PanoScene:
    """一个采集点的全景 + 它的标注（GT）与统计。"""

    uuid: str
    room: str
    panorama: Panorama
    n_frames_available: int = 0
    frames_used: int = 0
    #: GT 物体（原始字典，含 name/cls/bbox/room）
    objects: List[Dict[str, Any]] = field(default_factory=list)
    #: GT 轴对齐框 `(N, 7) = [cx, cy, cz, sx, sy, sz, yaw]`，yaw 恒 0
    gt_boxes: Optional[np.ndarray] = None
    #: 每个框对应的类别名
    gt_labels: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------------- 便捷访问 ----------------
    @property
    def center(self) -> Optional[np.ndarray]:
        return self.panorama.center

    @property
    def coverage(self) -> float:
        return self.panorama.coverage

    def frame(self):
        """包成 `RGBDFrame`（下游 `MapBuilder` 直接可用）。"""
        return frame_for_panorama(
            self.panorama,
            frame_id=f"2d3ds_{self.uuid[:8]}_{self.room}")

    def gt_within(self, max_range_m: float = 8.0,
                  classes: Optional[Sequence[str]] = None
                  ) -> Tuple[np.ndarray, List[str]]:
        """取光心 `max_range_m` 内的 GT 框（可选按类别过滤）。

        为什么要按距离筛：GT 覆盖整个房间（含背后的墙、隔壁的物体），
        而全景只看到 360° 视野内、且 `max_depth` 以内的东西。
        不筛的话评测会把"根本看不见"的物体算成漏检。

        空结果返回 `(0, 7)` 数组 + `[]`（**不是** `None`），
        且 dtype 与 `gt_boxes` 一致（上游 `objects_as_boxes` 给的是 float32；
        早先这里写死 `np.zeros((0,7))` 会变成 float64，
        下游拼接/比较时可能因 dtype 不同而产生意外行为）。
        """
        dt = np.float32 if self.gt_boxes is None else self.gt_boxes.dtype
        if self.gt_boxes is None or self.gt_boxes.shape[0] == 0:
            return np.zeros((0, 7), dtype=dt), []
        B = self.gt_boxes
        c = np.asarray(self.center, dtype=np.float64).reshape(3)
        # 用"框中心到光心的距离 − 框外接球半径"作为下界，保守地筛
        center_d = np.linalg.norm(B[:, :3] - c[None, :], axis=1)
        radius = 0.5 * np.linalg.norm(B[:, 3:6], axis=1)
        near = (center_d - radius) <= float(max_range_m)
        want = None
        if classes:
            want = {str(s).strip().lower() for s in classes if str(s).strip()}
            # ★ 全是空白/空串的 `classes` 视为"不过滤"，与
            #   `stanford2d3d.filter_objects(objs, [])` 的语义保持一致
            #   （那里空 prompts = 全要）。不这样统一的话，
            #   `classes=["  "]` 会静默返回空结果，很难排查。
            if not want:
                want = None
        keep = [i for i in range(B.shape[0])
                if near[i] and (want is None or self.gt_labels[i].lower() in want)]
        if not keep:
            return np.zeros((0, 7), dtype=dt), []
        idx = np.asarray(keep, dtype=int)
        return B[idx], [self.gt_labels[i] for i in idx]

    def describe(self) -> Dict[str, Any]:
        el_lo, el_hi = self.panorama.elevation_span_deg()
        return {
            "uuid": self.uuid[:12],
            "room": self.room,
            "可用视角": self.n_frames_available,
            "参与融合": self.frames_used,
            "全景分辨率": f"{self.panorama.rgb.shape[1]}x{self.panorama.rgb.shape[0]}",
            "像素覆盖": f"{self.coverage * 100:.1f}%",
            "立体角覆盖": f"{self.panorama.solid_angle_coverage * 100:.1f}%",
            "仰角范围": f"{el_lo:+.1f}° ~ {el_hi:+.1f}°",
            "GT物体": len(self.objects),
            "光心离散": f"{self.panorama.meta.get('center_spread_m', 0.0):.2e} m",
        }


# ==========================================================================
# 选择采集点
# ==========================================================================
def select_location(root: str | Path, *, room: Optional[str] = None,
                    uuid: Optional[str] = None, min_frames: int = 8,
                    index: int = 0) -> Location:
    """挑一个采集点（视角数足够多才有融合价值）。

    Parameters
    ----------
    room
        房间名（如 `office_6`）；给定时取该房间视角最多的那个采集点。
    uuid
        直接指定采集点 id（前缀匹配即可）。
    min_frames
        至少要有这么多视角，否则融合出来会有大片空洞。
    index
        不指定 room/uuid 时，按"视角数从多到少"排序后取第几个。
    """
    locs = list_locations(root)
    if not locs:
        raise FileNotFoundError(f"{root} 下没有解析到任何采集点（目录结构不对？）")

    if uuid:
        cand = [l for l in locs if l.uuid.startswith(uuid)]
        if not cand:
            raise KeyError(f"没有 uuid 以 {uuid!r} 开头的采集点")
        # `min_frames` 对显式指定的 uuid 同样生效（否则会静默给出视角过少、
        # 融合出大片空洞的采集点）。若因此一个都不剩，就明确报错而不是将就。
        enough = [l for l in cand if len(l.frame_ids) >= int(min_frames)]
        if not enough:
            raise ValueError(
                f"uuid={uuid!r} 的采集点只有 {max(len(l.frame_ids) for l in cand)} "
                f"个视角，少于 min_frames={min_frames}")
        return enough[0]

    if room:
        cand = [l for l in locs if l.room == room]
        if not cand:
            rooms = sorted({l.room for l in locs})
            raise KeyError(f"没有房间 {room!r}；可用房间示例：{rooms[:12]}")
    else:
        cand = locs
    cand = sorted(cand, key=lambda l: -len(l.frame_ids))
    cand = [l for l in cand if len(l.frame_ids) >= int(min_frames)]
    if not cand:
        raise ValueError(f"没有视角数 ≥ {min_frames} 的采集点")
    if index >= len(cand):
        raise IndexError(f"index={index} 越界（只有 {len(cand)} 个候选）")
    return cand[index]


# ==========================================================================
# 建场景
# ==========================================================================
def load_scene(root: str | Path, location: Optional[Location] = None, *,
               room: Optional[str] = None, uuid: Optional[str] = None,
               width: int = 2048, height: int = 1024,
               max_frames: Optional[int] = None,
               resize: Optional[Tuple[int, int]] = None,
               max_depth: float = 8.0,
               min_cos: float = 0.35,
               weight_power: float = 2.0,
               range_outlier_m: float = 0.5,
               rgb_mode: str = "weighted_mean",
               with_gt: bool = True,
               mat_path: Optional[str | Path] = None) -> PanoScene:
    """把一个真实采集点读成 `PanoScene`（N 个真实视角 → 1 张全景）。

    Parameters
    ----------
    max_frames
        最多用多少个视角（`None` = 全部）。视角越多覆盖越全、越慢。
        **均匀抽稀**而不是取前 N 个：同一采集点的视角是按 pitch/yaw 排列的，
        只取前面若干个会只覆盖一部分方向。
    resize
        把源视角缩放后再融合（如 `(540, 540)`），用于加速；
        会同步缩放内参。全景分辨率不随之改变。
    max_depth
        超过该**斜距**的点丢弃（米）。注意这是工程取舍：
        实测数据最远有 51.6 m 的真实长走廊（`office_28`），默认 8 m 会裁掉。
    """
    root = Path(root)
    if location is None:
        location = select_location(root, room=room, uuid=uuid)

    n_avail = len(location.frame_ids)
    ids = _pick_frame_ids(location.frame_ids, max_frames)

    frames = []
    for fid in ids:
        fr = location.frame(fid, resize=resize)
        if fr is not None:
            frames.append(fr)
    if not frames:
        raise RuntimeError(f"采集点 {location.uuid} 一帧都没读出来（缺 rgb/depth？）")

    logger.info(f"融合 {len(frames)}/{n_avail} 个真实视角 → "
                f"{width}x{height} 全景（{location.room}）")

    pano = fuse_to_equirect(frames, width=width, height=height,
                            weight_power=weight_power, min_cos=min_cos,
                            max_depth=max_depth, range_outlier_m=range_outlier_m,
                            rgb_mode=rgb_mode)

    scene = PanoScene(uuid=location.uuid, room=location.room, panorama=pano,
                      n_frames_available=n_avail, frames_used=len(frames),
                      meta={"root": str(root), "max_depth": max_depth,
                            "resize": resize, "frame_ids": ids})

    if with_gt:
        mp = Path(mat_path) if mat_path else (root / "3d" / "pointcloud.mat")
        if mp.exists():
            try:
                objs = load_objects(mp)
                # 只保留属于**本采集点所在房间**的物体
                room_objs = [o for o in objs if _room_matches(o, location)]
                scene.objects = room_objs
                boxes, labels = objects_as_boxes(room_objs)
                scene.gt_boxes, scene.gt_labels = boxes, labels
                scene.meta["gt_source"] = str(mp)
                scene.meta["gt_room_objects"] = len(room_objs)
            except Exception as e:                       # noqa: BLE001
                logger.warn(f"GT 读取失败（{type(e).__name__}: {e}），场景仍可用但无标注")
        else:
            logger.warn(f"没有 GT 文件 {mp}；场景仍可用但无标注")
    return scene


def _pick_frame_ids(ids: Sequence[int],
                    max_frames: Optional[int]) -> List[int]:
    """均匀抽稀视角 id（保持覆盖方向尽量完整）。"""
    ids = list(ids)
    if max_frames is None or len(ids) <= int(max_frames):
        return ids
    n = int(max_frames)
    # 等间隔取样，保证覆盖整个序列（而不是只取前 n 个方向）
    sel = np.linspace(0, len(ids) - 1, num=n).round().astype(int)
    return [ids[i] for i in np.unique(sel)]


def _room_matches(obj: Dict[str, Any], location: Location) -> bool:
    """GT 物体的 room 与采集点的 room 是否指同一个房间。

    两边命名**不完全一致**：文件名里是 `office_6`，json/mat 里常带区号
    （如 `office_6_1`）。所以用"前缀 + 下划线"匹配，而不是全等。
    """
    o_room = str(obj.get("room", ""))
    l_room = str(location.room)
    if not o_room:
        return True                     # 没房间信息就不排除
    return o_room == l_room or o_room.startswith(l_room + "_")


def load_scenes(root: str | Path, *, n: int = 5, min_frames: int = 8,
                rooms: Optional[Sequence[str]] = None, **kwargs) -> List[PanoScene]:
    """一次读多个采集点（用于小批量评测）。`rooms` 给定时按房间名取。"""
    root = Path(root)
    out: List[PanoScene] = []
    if rooms:
        for r in rooms:
            try:
                loc = select_location(root, room=r, min_frames=min_frames)
            except (KeyError, ValueError) as e:
                logger.warn(f"跳过房间 {r}：{e}")
                continue
            out.append(load_scene(root, loc, **kwargs))
        return out

    locs = sorted(list_locations(root), key=lambda l: -len(l.frame_ids))
    locs = [l for l in locs if len(l.frame_ids) >= int(min_frames)]
    if not locs:
        raise ValueError(f"没有视角数 ≥ {min_frames} 的采集点")
    # 沿"视角数降序"取前 n 个会集中在同一批房间；改成等间隔取，房间更分散
    idx = np.linspace(0, len(locs) - 1, num=min(int(n), len(locs))).round().astype(int)
    for i in np.unique(idx):
        out.append(load_scene(root, locs[int(i)], **kwargs))
    return out


# ==========================================================================
# GT 框 → 全景图像：投影 + 可见性
# ==========================================================================
# 采样点定义搬到了 `frame_gt.box_surface_points`：针孔版可见性判定
# （`frame_gt.box_to_frame`）必须与全景版用**同一套采样、同一个分母**，
# 否则"用一帧 vs 用多帧融合"的消融会数不到一块去。这里保留旧名字。
_box_surface_points = box_surface_points


def box_to_pano(scene: PanoScene, box: np.ndarray, *,
                grid: int = 6,
                occ_tol_abs: float = 0.30,
                occ_tol_rel: float = 0.10,
                min_visible_frac: float = 0.25) -> Optional[Dict[str, Any]]:
    """把一个世界系轴对齐 GT 框投到全景图上，并判定**可见性**。

    返回 `dict`（完全不可见时返回 None），含：

    · `uv`：`(u0, v0, u1, v1)` 像素框（已处理 ±180° 接缝）
    · `range_m`：框表面点到光心的平均距离
    · `visible_frac`：采样点中"既被覆盖、又没被更近的东西挡住"的比例
    · `covered_frac`：只算"该方向有有效深度"的比例

    为什么要判可见性：一个采集点只能看到房间的一部分（我们的全景本身
    只覆盖约 39~56% 的像素），如果拿**房间全部** GT 去算召回，
    会把"根本看不见"的物体算成漏检，指标就没有意义了。

    ⚠️ `visible_frac` 的分母是**包围盒表面采样点**，而一个盒子总有一半左右
    的表面是**背对相机的**（例如天花板盒子的顶面、柜子的背面）。
    所以"完全看得见"的物体也只会有 ~50% 左右，**不会接近 100%**。
    默认阈值 `min_visible_frac=0.25` 的含义就是"朝向相机的那一面至少看到一半"。
    不要把这个数读成"看到了物体的百分之多少"。
    该阈值会**透传给 `box_to_pano`**，保证条目里的 `visible` 字段
    与本函数的筛选口径一致。

    遮挡判据：某采样点被判定为"可见"，当且仅当该方向**有有效深度**，
    且观测到的斜距**不小于**该点的斜距（允许 `occ_tol` 容差）——
    也就是"前面没有更近的东西挡住它"。
    注意 GT 框是**轴对齐包围盒**，物体真实表面在盒内，所以观测距离
    通常**小于**盒面距离，这是正常的，不能据此判为遮挡（所以只看
    "有没有更近的东西"，而不是要求两者相等）。
    """
    if scene.center is None:
        return None
    C = np.asarray(scene.center, dtype=np.float64).reshape(3)
    box = np.asarray(box, dtype=np.float64).reshape(7)
    lo, hi = box[:3] - box[3:6] / 2.0, box[:3] + box[3:6] / 2.0
    # 相机**在盒子里**：这个物体真的包住了相机，不存在有意义的单框
    if bool(np.all(C >= lo) and np.all(C <= hi)):
        return None

    pts = _box_surface_points(box, grid=grid)
    d = pts - C[None, :]
    r = np.linalg.norm(d, axis=1)
    ok = r > 1e-6
    if ok.sum() < 4:
        return None
    d, r = d[ok], r[ok]

    az = np.arctan2(d[:, 1], d[:, 0])                   # [-π, π)
    el = np.arcsin(np.clip(d[:, 2] / r, -1.0, 1.0))

    # ---- 方位角接缝：用"最大空隙"法求**最小环绕区间** ----
    #
    # ★ 这里踩过一个坑：第一版直接看 `max(az) − min(az)`，一旦超过 180°
    #   就判成"物体包住相机"并返回 None。但这在**近极点**时完全错：
    #   相机正上方的小盒子，采样点的 xy 偏移方向绕了整整一圈
    #   （实测跨度 337°），可它的**立体角很小** —— 结果天花板灯、
    #   吸顶设备这类物体**永远进不了候选集**，是系统性的假阴性。
    #
    #   正确做法：方位角是**圆**，物体占据的是"去掉最大空隙"的那一段弧。
    #   近极点物体最大空隙很小 → 区间接近整圈（在等距柱状图上，
    #   极点附近的区域本来就横跨整幅图，这是投影的固有性质，不是 bug）。
    az_sorted = np.sort(az)
    gaps = np.diff(np.concatenate([az_sorted, [az_sorted[0] + 2.0 * np.pi]]))
    gi = int(np.argmax(gaps))
    az_start = float(az_sorted[(gi + 1) % az_sorted.size])
    az_span = float(2.0 * np.pi - gaps[gi])

    H, W = scene.panorama.rgb.shape[:2]
    # 与 directions_to_equirect / equirect_directions 同一约定（连续坐标）
    u_c = lambda a: (a + np.pi) / (2.0 * np.pi) * W          # noqa: E731
    v_c = lambda e: (np.pi / 2.0 - e) / np.pi * H            # noqa: E731
    u0f = u_c(az_start)
    u1f = u_c(az_start + az_span)
    v0 = int(np.clip(np.floor(v_c(float(el.max()))), 0, H - 1))
    v1 = int(np.clip(np.floor(v_c(float(el.min()))), 0, H - 1))

    # 展开后的 u 可能超过 W（跨接缝）。给出**拆成两段**的、都落在图内的框，
    # 因为 `Detection2D` 的 bbox 必须落在图像内才有效。
    parts: List[Tuple[int, int, int, int]] = []
    a0, a1 = int(np.floor(u0f)), int(np.floor(u1f))
    if a1 < W:
        parts.append((max(0, a0), v0, min(W - 1, a1), v1))
    else:
        parts.append((max(0, a0), v0, W - 1, v1))
        parts.append((0, v0, min(W - 1, a1 - W), v1))
    parts = [p for p in parts if p[2] >= p[0]]

    # 采样点投到像素上（取模后才能查深度）
    ui = np.mod(np.floor(u_c(az)).astype(np.int64), W)
    vi = np.clip(np.floor(v_c(el)).astype(np.int64), 0, H - 1)
    obs = scene.panorama.depth_m[vi, ui].astype(np.float64)
    covered = obs > 0
    not_occluded = obs >= (r - np.maximum(occ_tol_abs, occ_tol_rel * r))
    vis = covered & not_occluded

    return {
        "uv": parts[0],
        "uv_parts": parts,
        "uv_wrapped": bool(a1 >= W),
        "az_span_deg": float(np.degrees(az_span)),
        "range_m": float(np.median(r)),
        "visible_frac": float(vis.mean()),
        "covered_frac": float(covered.mean()),
        # ★ 门限由参数传入，**不再硬编码** 0.25：否则
        #   `visible_gt(min_visible_frac=0.5)` 会用它筛完，
        #   而条目里的 `visible` 仍按 0.25 判定，同一份数据两处结论不一致。
        "visible": bool(vis.mean() >= float(min_visible_frac)),
    }


def visible_gt(scene: PanoScene, *, max_range_m: float = 8.0,
               classes: Optional[Sequence[str]] = None,
               min_visible_frac: float = 0.25,
               grid: int = 6) -> List[Dict[str, Any]]:
    """列出**全景里真的看得见**的 GT 物体（带投影框）。

    每条含 `box` / `label` / `uv` / `uv_parts` / `visible_frac` / `range_m`。
    `uv_parts` 是**落在图内**的 1~2 个框（跨 ±180° 接缝的物体会被拆成两段，
    因为一个 bbox 不能越出图像边界）。

    这是"在真实数据上评语言接地"的候选集 —— 只有看得见的物体
    才有资格参与召回统计。
    """
    boxes, labels = scene.gt_within(max_range_m, classes)
    out: List[Dict[str, Any]] = []
    for i in range(boxes.shape[0]):
        info = box_to_pano(scene, boxes[i], grid=grid,
                           min_visible_frac=min_visible_frac)
        if info is None or info["visible_frac"] < float(min_visible_frac):
            continue
        out.append({"box": boxes[i], "label": labels[i],
                    "index": i, **info})
    out.sort(key=lambda o: o["range_m"])
    return out
