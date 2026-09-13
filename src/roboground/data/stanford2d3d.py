# -*- coding: utf-8 -*-
"""Stanford 2D-3D-S 数据源适配器。

数据从哪来
=========
Redivis（斯坦福 Doerr 学院数据仓库）→ `area_X_no_xyz.tar` → 选择性抽取。
获取/核验过程见 `docs/2D3D-S数据集核验报告.md`，脚本 `scripts/28~33`。

本模块只负责"把文件读成 `RGBDFrame`"，不掺算法。

★ 三条**实测**出来的约定（照官方文档写会错，见核验报告）
=====================================================
1. **位姿是 `(3, 4)` 的 `camera_rt_matrix` = [R | t]**，
   官方 README 说 4×3，**实测是 3×4**。
   方向经**可判定验算**确认为 **world→camera**：
   `C = −Rᵀ·t` 与 json 里的 `camera_location` 误差 **1.2 µm**（反向假设差 38 m）。
   ⇒ 正好与本项目 `CameraPose` 的约定一致，**直接构造即可，不做任何转换**。
2. **深度 `depth_m = raw / 512.0`**（官方说明的 1/512 m）。
   验算依据：按 512 得 0.60~3.09 m（办公室合理），按 1000 得 0.31~1.58 m（不合理）。
   且**完全稠密**（缺失值 0%），不需要空洞处理。
3. **文件名是单下划线**（官方写的是双下划线）：
   `camera_{uuid}_{room}_{i}_frame_{j}_domain_{modality}.{ext}`

「多视角」在这里是什么
==================
同一 `camera_uuid`（采集点）下的 ~54 个帧**共享同一个 `camera_location`**，
但**朝向与 FOV 各不相同** —— 也就是"站在同一个点、朝不同方向看"。
把它们融合起来就是一张 **360° 全景**，而官方 `/pano/` 下有**参考答案**可以逐像素比对。

⚠️ 像素来源说明（不要含糊）
=========================
`/data` 的 RGB 是官方在**真实采集位姿**上渲染的（官方描述："synthesized but
accurate RGB images"）。**视角是真的，像素是重建渲染的。**
真实传感器原图在 `/raw`（需 HDR 色调映射，本项目暂未采用）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame
from roboground.utils.logging import get_logger

logger = get_logger("data.stanford2d3d")

#: 深度换算：`depth_m = raw / DEPTH_SCALE`。
#: 512 是**验算**出来的（见模块 docstring 第 2 条），不是照抄文档。
DEPTH_SCALE = 512.0

#: ★ 「无数据」编码值。**实测出来的**，不是照抄文档：
#:   抽 40 张深度图共 4665.6 万像素，`raw == 65535` 占 **0.88%**，
#:   而 `raw == 0` 占 0%。65535 / 512 = **128.0 m**，正好是官方说的量程上限 ——
#:   所以 65535 就是"这里没有数据"，不是"这里离 128 米"。
#:
#: ⚠️ 踩过的坑：第一版核验脚本用 `%.1f%%` 打印，把 0.88% 显示成 "0.0%"，
#:    于是我在报告里写了"完全稠密、不需要空洞处理"—— **那个结论是错的**。
#:    不处理它的话，全景融合会吃到 128 m 的假深度。
INVALID_RAW = 65535

#: 文件名解析。实测命名：`camera_{uuid}_{room}_{i}_frame_{j}_domain_{modality}.{ext}`
#: `frame` 为 `equirectangular` 时表示全景。
#: 两处 `(?!_)` 是**故意的严苛**：官方 README 把文件名写成了双下划线
#: （`..._{uuid}__{room}__frame...`、`..._domain__rgb`），若照它拼路径，
#: 松散的正则会把 `_office_6` / `_rgb` 当成合法的房间名和模态收下，
#: 然后在下游查找时才莫名其妙地 KeyError/找不到文件。这里直接拒掉，
#: 保证"解析成功 ⇒ 文件名是真的长这样"。
FILENAME_RE = re.compile(
    r"^camera_(?P<uuid>[0-9a-f]{32})_(?!_)(?P<room>.+?)_frame_"
    r"(?P<frame>\d+|equirectangular)_domain_(?!_)(?P<modality>[a-z_]+)\.(?P<ext>[a-z]+)$"
)

#: 我们会用到的模态 → 文件名里的 `domain` 后缀
MODALITY = {
    "rgb": "rgb",
    "depth": "depth",
    "semantic": "semantic",
    "pose": "pose",
    "normal": "normals",
}


#: 逐像素语义图里「无数据」的标记色（官方说明：`#0D0D0D`）。
#: 它编码成整数索引后约为 856845，**远大于标签总数 9816**，所以可以据此判无效。
SEMANTIC_MISSING_RGB = (13, 13, 13)


def load_semantic_labels(path: str | Path) -> List[str]:
    """读官方 `assets/semantic_labels.json`（9816 条）。

    ⚠️ **数据集 tar 里不带这个文件**，要单独从官方仓库取：
        https://raw.githubusercontent.com/alexsax/2D-3D-Semantics/master/assets/semantic_labels.json

    标签格式：`{类}_{实例号}_{房间类型}_{房号}_{区号}`，例如 `beam_10_hallway_6_1`。
    第 0 条是 `<UNK>_0_<UNK>_0_0`（占位/未知）。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 {p}；该文件不在数据 tar 里，需从官方仓库"
            "（github.com/alexsax/2D-3D-Semantics）单独下载："
            "assets/semantic_labels.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{p} 不是标签列表（实际 {type(data).__name__}）")
    return [str(x) for x in data]


def semantic_class(index: int, labels: Sequence[str]) -> Optional[str]:
    """语义图里的整数索引 → 类别名（如 `beam`）；无效索引返回 None。"""
    i = int(index)
    if i < 0 or i >= len(labels):
        return None                       # 含 #0D0D0D 那种"无数据"标记
    name = labels[i]
    return re.sub(r"[-_]?\d+$", "", name.split("_")[0]) or None


# ==========================================================================
# 文件名解析
# ==========================================================================
@dataclass(frozen=True)
class NameParts:
    """文件名解析结果。"""

    uuid: str
    room: str          # 例如 "office_6"（注意：json 里的 room 多一节 areaNum）
    frame: int         # -1 表示 equirectangular（全景）
    modality: str
    ext: str
    is_panorama: bool


def parse_name(filename: str) -> Optional[NameParts]:
    """解析一个 2D-3D-S 文件名；不匹配返回 None。"""
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    f = m.group("frame")
    return NameParts(
        uuid=m.group("uuid"),
        room=m.group("room"),
        frame=(-1 if f == "equirectangular" else int(f)),
        modality=m.group("modality"),
        ext=m.group("ext"),
        is_panorama=(f == "equirectangular"),
    )


# ==========================================================================
# 位姿
# ==========================================================================
def pose_from_json(obj: Dict[str, Any]) -> CameraPose:
    """位姿 json → `CameraPose`（**直接构造，不做约定转换**）。

    实测：`camera_rt_matrix` 是 `(3, 4) = [R | t]`，且 `p_cam = R·p_world + t`，
    与本项目 `CameraPose` 的约定一致（验算误差 1.2 µm）。
    """
    rt = np.asarray(obj["camera_rt_matrix"], dtype=np.float64)
    if rt.shape == (3, 4):
        R, t = rt[:, :3], rt[:, 3]
    elif rt.shape == (4, 3):                 # 官方文档说的形状（实测没见到，兜底）
        R, t = rt[:3, :], rt[3, :]
    else:
        raise ValueError(f"camera_rt_matrix 形状异常：{rt.shape}")
    return CameraPose(R, t)


def intrinsics_from_json(obj: Dict[str, Any]) -> CameraIntrinsics:
    """内参 json → `CameraIntrinsics`（逐帧不同，必须逐帧读）。"""
    K = np.asarray(obj["camera_k_matrix"], dtype=np.float64).reshape(3, 3)
    return CameraIntrinsics(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        width=int(obj.get("image_width", 1080)),
        height=int(obj.get("image_height", 1080)),
    )


# ==========================================================================
# 采集点（scan location）
# ==========================================================================
@dataclass
class Location:
    """一个采集点：同一个 `camera_uuid` 下的全部帧。

    这些帧**共享同一个相机位置**、但**朝向不同** —— 融合起来就是 360° 全景。
    """

    uuid: str
    room: str                                   # 来自文件名（如 "office_6"）
    root: Path                                  # 解压后的 area 目录
    frame_ids: List[int] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    def _path(self, modality: str, frame: int, *, pano: bool = False) -> Optional[Path]:
        domain = MODALITY[modality]
        fam = "equirectangular" if pano else str(frame)
        base = f"camera_{self.uuid}_{self.room}_frame_{fam}_domain_{domain}"
        ext = "json" if modality == "pose" else "png"
        sub = "pano" if pano else "data"
        p = self.root / sub / modality / f"{base}.{ext}"
        return p if p.exists() else None

    # ---------------- 单个模态 ----------------
    def pose(self, frame: int) -> CameraPose:
        p = self._path("pose", frame)
        if p is None:
            raise FileNotFoundError(f"没找到位姿：uuid={self.uuid} frame={frame}")
        return pose_from_json(json.loads(p.read_text(encoding="utf-8")))

    def intrinsics(self, frame: int) -> CameraIntrinsics:
        p = self._path("pose", frame)
        if p is None:
            raise FileNotFoundError(f"没找到内参：uuid={self.uuid} frame={frame}")
        return intrinsics_from_json(json.loads(p.read_text(encoding="utf-8")))

    def depth_m(self, frame: int) -> Optional[np.ndarray]:
        """深度（米）。`raw / 512`；**`raw == 65535` 记为无效（0）**。

        本项目的约定是 `depth_m == 0` 表示无效（见 `RGBDFrame` 与几何层），
        所以这里把 65535、0 以及 ≥128 m 的值统一归零。
        实测无效占 **0.88%**（4665.6 万像素统计，40 张图）。
        """
        from PIL import Image

        p = self._path("depth", frame)
        if p is None:
            return None
        raw = np.asarray(Image.open(p)).astype(np.float32)
        d = raw / DEPTH_SCALE
        d[(raw <= 0) | (raw >= INVALID_RAW)] = 0.0
        return d

    def depth_valid_mask(self, frame: int) -> Optional[np.ndarray]:
        """有效深度掩码（`raw == 65535` 等无数据位置为 False）。"""
        from PIL import Image

        p = self._path("depth", frame)
        if p is None:
            return None
        raw = np.asarray(Image.open(p))
        return (raw > 0) & (raw < INVALID_RAW)

    def rgb(self, frame: int) -> Optional[np.ndarray]:
        from PIL import Image

        p = self._path("rgb", frame)
        return None if p is None else np.asarray(Image.open(p).convert("RGB"))

    def semantic(self, frame: int) -> Optional[np.ndarray]:
        """逐像素实例标签。官方说明：RGB 三通道编码 24-bit 整数索引。

        缺失像素被编码为 `#0D0D0D`（其值大于标签总数）。
        """
        from PIL import Image

        p = self._path("semantic", frame)
        if p is None:
            return None
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.int64)
        idx = a[..., 0] * 65536 + a[..., 1] * 256 + a[..., 2]
        return idx

    # ---------------- 组装成 RGBDFrame ----------------
    def frame(self, frame_id: int, *, resize: Optional[Tuple[int, int]] = None,
              frame_name: Optional[str] = None) -> Optional[RGBDFrame]:
        """读一帧 → `RGBDFrame`（带**真实**内参与位姿）。"""
        color = self.rgb(frame_id)
        depth = self.depth_m(frame_id)
        if color is None or depth is None:
            return None
        K = self.intrinsics(frame_id)
        pose = self.pose(frame_id)

        if resize is not None:
            tw, th = int(resize[0]), int(resize[1])
            if (color.shape[1], color.shape[0]) != (tw, th):
                from PIL import Image

                color = np.asarray(
                    Image.fromarray(color).resize((tw, th), Image.BILINEAR))
                depth = np.asarray(
                    Image.fromarray(depth).resize((tw, th), Image.NEAREST))
                sx, sy = tw / float(K.width), th / float(K.height)
                K = CameraIntrinsics(fx=K.fx * sx, fy=K.fy * sy,
                                     cx=K.cx * sx, cy=K.cy * sy,
                                     width=tw, height=th)

        return RGBDFrame(
            color=color.astype(np.uint8),
            depth_m=depth.astype(np.float32),
            intrinsics=K, pose=pose,
            frame_id=(frame_name or f"2d3ds_{self.uuid[:8]}_f{frame_id}"),
            meta={"source": "stanford2d3d", "scan_uuid": self.uuid,
                  "room": self.room, "frame": frame_id},
        )

    def frames(self, *, resize: Optional[Tuple[int, int]] = None,
               limit: Optional[int] = None) -> List[RGBDFrame]:
        out = []
        for fid in self.frame_ids[: (limit or len(self.frame_ids))]:
            fr = self.frame(fid, resize=resize)
            if fr is not None:
                out.append(fr)
        return out

    # ---------------- 官方全景（融合结果的参考答案）----------------
    def official_panorama(self) -> Optional[Dict[str, np.ndarray]]:
        """读官方等距柱状全景（rgb / depth）。**这是我们融合结果的参考答案。**"""
        from PIL import Image

        out: Dict[str, np.ndarray] = {}
        pr = self._path("rgb", -1, pano=True)
        pd_ = self._path("depth", -1, pano=True)
        if pr is not None:
            out["rgb"] = np.asarray(Image.open(pr).convert("RGB"))
        if pd_ is not None:
            out["depth_m"] = np.asarray(Image.open(pd_)).astype(np.float32) / DEPTH_SCALE
        return out or None


# ==========================================================================
# 目录级入口
# ==========================================================================
def list_locations(root: str | Path) -> List[Location]:
    """扫 `data/pose/` 目录，按 `camera_uuid` 聚成采集点。

    `root` 是解压后的 **area 目录**（即含 `data/`、`pano/`、`3d/` 的那一层）。
    """
    root = Path(root)
    pose_dir = root / "data" / "pose"
    if not pose_dir.is_dir():
        raise FileNotFoundError(
            f"找不到 {pose_dir}；root 应指向解压后的 area 目录（含 data/ pano/ 3d/）")

    groups: Dict[str, Location] = {}
    for p in sorted(pose_dir.glob("*.json")):
        parts = parse_name(p.name)
        if parts is None or parts.is_panorama:
            continue
        loc = groups.get(parts.uuid)
        if loc is None:
            loc = Location(uuid=parts.uuid, room=parts.room, root=root)
            groups[parts.uuid] = loc
        if parts.frame not in loc.frame_ids:
            loc.frame_ids.append(parts.frame)
    for loc in groups.values():
        loc.frame_ids.sort()
    return [groups[k] for k in sorted(groups)]


def describe(root: str | Path, *, sample: int = 3) -> Dict[str, Any]:
    """给一个 area 目录做概览（采集点数、每点帧数、是否有全景/GT）。"""
    root = Path(root)
    locs = list_locations(root)
    n_frames = [l.n_frames for l in locs]
    pano_dir = root / "pano" / "rgb"
    n_pano = len(list(pano_dir.glob("*.png"))) if pano_dir.is_dir() else 0
    has_mat = (root / "3d" / "pointcloud.mat").exists()
    return {
        "area_dir": str(root),
        "采集点数": len(locs),
        "帧数合计": int(sum(n_frames)),
        "每点帧数": (f"min {min(n_frames)} 中位 {int(np.median(n_frames))} "
                     f"max {max(n_frames)}" if n_frames else "—"),
        "官方全景数": n_pano,
        "有 GT(pointcloud.mat)": has_mat,
        "样例采集点": [
            {"uuid": l.uuid, "room": l.room, "n_frames": l.n_frames}
            for l in locs[:sample]
        ],
    }


# ==========================================================================
# 物体级 GT（MAT v7.3 / HDF5）
# ==========================================================================
def load_objects(mat_path: str | Path,
                 *, max_objects: Optional[int] = None) -> List[Dict[str, Any]]:
    """从 `3d/pointcloud.mat` 读物体级 GT。

    返回 `[{"name","cls","bbox","room"}, ...]`，`bbox` 是
    `[Xmin, Ymin, Zmin, Xmax, Ymax, Zmax]`（**世界系，与位姿同坐标系**）。

    ⚠️ 两个 MATLAB v7.3 存储坑（不处理会读出莫名其妙的数字）：
      1. struct array 是**按字段存成引用数组**：
         `Area_1/Disjoint_Space/name` 是 (44,1) 引用数组，第 i 个元素是
         `h[h[...]['name'][i,0]]`；
      2. **字符串是 char 数组（uint16 码点）**，直接取 `[0]` 得到的是
         **首字符的码点**（例如 'c' = 99），看起来像"名字是 99"。
    """
    import h5py

    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"找不到 {mat_path}")

    def _decode(node) -> str:
        a = np.asarray(node)
        if a.dtype.kind in ("U", "S"):
            return str(a.ravel()[0])
        if a.dtype.kind in ("u", "i"):
            return "".join(chr(int(c)) for c in a.ravel() if int(c) != 0)
        return str(a.ravel()[0])

    def _deref(h, node):
        if np.asarray(node).dtype == h5py.ref_dtype:
            r = np.asarray(node).ravel()[0]
            return h[r] if r else None
        return node

    out: List[Dict[str, Any]] = []
    with h5py.File(mat_path, "r") as h:
        area_key = next((k for k in h.keys() if not k.startswith("#")), None)
        if area_key is None:
            raise ValueError(f"{mat_path} 里没有 area 组（顶层 {list(h.keys())}）")
        area = h[area_key]
        ds_key = next((k for k in area.keys() if "disjoint" in k.lower()), None)
        if ds_key is None:
            raise ValueError(f"{area_key} 里没有 Disjoint_Space（{list(area.keys())}）")
        ds = area[ds_key]
        name_ds = ds["name"]
        obj_ds = next(ds[k] for k in ds.keys() if k.lower() == "object")

        for i in range(name_ds.shape[0]):
            room = _decode(_deref(h, name_ds[i, 0]))
            grp = _deref(h, obj_ds[i, 0])
            if grp is None or not hasattr(grp, "keys"):
                continue
            gname = next((grp[k] for k in grp.keys() if k.lower() == "name"), None)
            gbbox = next((grp[k] for k in grp.keys() if k.lower() == "bbox"), None)
            if gname is None:
                continue
            for j in range(np.asarray(gname).shape[0]):
                nm = _decode(_deref(h, np.asarray(gname)[j, 0]))
                bb = np.zeros(6, dtype=np.float64)
                if gbbox is not None:
                    bb = np.asarray(_deref(h, np.asarray(gbbox)[j, 0]),
                                    dtype=np.float64).ravel()
                # 类别名：去掉末尾的实例序号（chair-1 → chair；floor_1 → floor）
                cls = re.sub(r"[-_]?\d+$", "", nm)
                out.append({"name": nm, "cls": cls, "bbox": bb, "room": room})
                if max_objects is not None and len(out) >= max_objects:
                    return out
    return out


def objects_as_boxes(objects: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """把 GT 物体转成 `(boxes, labels)` —— 与本项目 SUN RGB-D 的接口一致。

    `boxes` 形状 `(N, 7)`：`[cx, cy, cz, dx, dy, dz, yaw]`（**yaw 恒为 0**，
    因为 2D-3D-S 给的是**轴对齐**包围盒；这一点必须记住，
    它意味着**不能用 yaw 有向框 IoU 去评它**）。
    """
    boxes, labels = [], []
    for o in objects:
        bb = np.asarray(o["bbox"], dtype=np.float64).ravel()
        if bb.size != 6 or not np.all(np.isfinite(bb)):
            continue
        lo, hi = bb[:3], bb[3:]
        size = np.clip(hi - lo, 1e-6, None)
        center = (lo + hi) / 2.0
        boxes.append(np.concatenate([center, size, [0.0]]))
        labels.append(str(o["cls"]))
    if not boxes:
        return np.zeros((0, 7), dtype=np.float32), []
    return np.stack(boxes).astype(np.float32), labels


def filter_objects(objects: Sequence[Dict[str, Any]],
                   prompts: Sequence[str]) -> List[Dict[str, Any]]:
    """按关心的类别过滤 GT（类别名大小写不敏感，支持子串匹配）。"""
    want = {p.strip().lower() for p in prompts if p and p.strip()}
    if not want:
        return list(objects)
    out = []
    for o in objects:
        cls = str(o["cls"]).lower()
        if cls in want or any(w in cls or cls in w for w in want):
            out.append(o)
    return out
