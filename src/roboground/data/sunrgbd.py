"""SUN RGB-D 加载器（自建，输出 RoboGround 需要的 RGB-D 帧格式）。

为什么自己写而不用现成的预处理结果
----------------------------------
`G:\\Embodied3D\\data\\processed\\` 里已有 10335 个 npz，但它服务于 3D 检测
（存的是**下采样后的点云** `pc` + 224×224 的缩略图），而 RoboGround 需要的是：

1. **全分辨率、像素对齐的 RGB + 深度图** —— 因为要按检测掩码反投影；
2. **相机内外参** —— 因为要把 2D 语义投到 3D 世界系；
3. **GT 的 3D 框 + 类别名** —— 用作离线 stub 检测器的输入和评测基准。

SUN RGB-D 的几个易踩坑点（都已在本模块处理）
-------------------------------------------
1. **深度尺度是 `/10000` 而不是 `/1000`**：实测 depth PNG 数值 1.2 万~4.5 万，
   `depth_m = raw / 10000`。写错一个数量级，整个点云尺度全错。
2. **opencv 读 16 位 PNG 会返回 None**：必须用 PIL 读深度图，否则静默拿到空数组。
3. **坐标系（最容易错的一点）**：SUN RGB-D 的 GT 3D 框**不在 OpenCV 相机系**，
   而在"y 为深度、z 为竖直"的重力对齐世界系里。经
   `scripts/calibrate_sunrgbd_geometry.py` 用角点闭环一致性标定，
   世界→相机为 `R = Rtiltᵀ @ Pᵀ`（`P` 是固定 90° 轴置换），
   已封装为 `CameraPose.from_sunrgbd`。
4. **`coeffs` 是全边长不是半边长**（见 `build_scene_index` 里的标定依据）。
5. **meta 结构**：`SUNRGBDMeta` 形状是 `(1, 10335)`，需要 `.ravel()`；
   字段名是 `K`（不是 `intrinsics`）、`sequenceName`、`depthname`、`rgbname`。
6. **加载 .mat 很慢**：所以本模块提供**场景索引缓存** —— 首次扫描后把
   每帧的 K / Rtilt / GT 框 / 相对路径存成一个紧凑 npz，后续秒开。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from roboground.types import CameraIntrinsics, CameraPose, RGBDFrame
from roboground.utils.io import ensure_dir, load_npz, save_npz
from roboground.utils.logging import get_logger

logger = get_logger("data.sunrgbd")


# ==========================================================================
# 常量
# ==========================================================================
#: SUN RGB-D 深度图的单位换算：`depth_m = raw / DEPTH_SCALE`
DEPTH_SCALE = 10000.0

#: 3D 检测的标准 10 类（Embodied3D 就是按这 10 类做的，便于横向对比）
STANDARD_10 = (
    "bed", "table", "sofa", "chair", "toilet",
    "desk", "dresser", "night_stand", "bookshelf", "bathtub",
)

#: 把 SUN RGB-D 的原始类名规范化，让词法查询能对上双语别名表
CLASS_CANON: Dict[str, str] = {
    "night_stand": "nightstand",
    "bookshelf": "bookshelf",
    "dresser": "cabinet",
    "desk": "desk",
    "sofa": "sofa",
    "table": "table",
    "chair": "chair",
    "bed": "bed",
    "toilet": "toilet",
    "bathtub": "bathtub",
    "box": "box",
    "cup": "cup",
    "bottle": "bottle",
    "monitor": "monitor",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "book": "book",
    "lamp": "lamp",
    "picture": "picture",
    "clock": "clock",
    "pillow": "pillow",
    "trash_can": "trash_can",
    "sink": "sink",
    "refrigerator": "refrigerator",
    "curtain": "curtain",
    "door": "door",
    "window": "window",
    "mirror": "mirror",
    "person": "person",
    "shelf": "shelf",
    "cabinet": "cabinet",
    "whiteboard": "whiteboard",
    "paper": "paper",
    "towel": "towel",
    "paper_bag": "bag",
    "backpack": "bag",
    "bag": "bag",
    "basket": "basket",
    "telephone": "phone",
    "printer": "printer",
    "speaker": "speaker",
    "tv": "monitor",
    "computer": "laptop",
}


def canon_class(name: str) -> str:
    """规范化类别名（无法识别时原样返回并转小写）。"""
    key = str(name).strip().lower().replace(" ", "_")
    return CLASS_CANON.get(key, key)


# ==========================================================================
# 场景对象
# ==========================================================================
@dataclass
class SUNRGBDScene:
    """一个 SUN RGB-D 场景（单帧 RGB-D + 相机参数 + GT 3D 框）。

    Attributes
    ----------
    sequence : str
        原始 `sequenceName`（含 `/`，用于拼路径）。
    color : (H,W,3) uint8
    depth_m : (H,W) float32
        已换算成米。
    intrinsics : CameraIntrinsics
    pose : CameraPose
        **world → camera**，与世界系 (x右, y前, z上) 配套。
    boxes_3d : (K,7) float32
        `[cx, cy, cz, dx, dy, dz, heading]`，世界系、z 轴为竖直、heading 绕 z。
    labels : list[str]
        规范化后的类别名。
    """

    sequence: str
    color: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    pose: CameraPose
    boxes_3d: np.ndarray = field(default_factory=lambda: np.zeros((0, 7), np.float32))
    labels: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.color.shape[0]), int(self.color.shape[1])

    def filter_classes(self, keep: Sequence[str]) -> "SUNRGBDScene":
        """只保留指定类别的 GT 框（用于对齐标准评测协议）。"""
        keep_set = {canon_class(k) for k in keep}
        idx = [i for i, lab in enumerate(self.labels) if canon_class(lab) in keep_set]
        return SUNRGBDScene(
            sequence=self.sequence,
            color=self.color,
            depth_m=self.depth_m,
            intrinsics=self.intrinsics,
            pose=self.pose,
            boxes_3d=self.boxes_3d[idx] if idx else np.zeros((0, 7), np.float32),
            labels=[self.labels[i] for i in idx],
            meta=dict(self.meta),
        )

    def to_frame(self, *, frame_id: Optional[str] = None) -> RGBDFrame:
        """转成 `RGBDFrame`，并把 GT 框放进 `meta`（供离线 stub 检测器使用）。"""
        return RGBDFrame(
            color=self.color,
            depth_m=self.depth_m,
            intrinsics=self.intrinsics,
            pose=self.pose,
            frame_id=frame_id or self.sequence.replace("/", "_"),
            meta={
                "boxes_3d": self.boxes_3d,
                "labels": list(self.labels),
                "source": "sunrgbd",
                "sequence": self.sequence,
            },
        )

    def cloud(self, *, min_depth: float = 0.2, max_depth: float = 8.0,
              max_points: Optional[int] = 100_000, seed: int = 0):
        """从深度图生成世界系彩色点云（可视化/重渲染用）。"""
        from roboground.geometry.projection import frame_to_pointcloud  # noqa: PLC0415

        return frame_to_pointcloud(
            self.to_frame(), min_depth=min_depth, max_depth=max_depth,
            max_points=max_points, colors=True, seed=seed,
        )


# ==========================================================================
# 场景索引（避免每次加载 13.9MB 的 .mat）
# ==========================================================================
_INDEX_REQUIRED = ("sequences", "K", "Rtilt", "box_flat", "box_offset", "label_flat")


def build_scene_index(
    root: str = r"G:\sunrgbd_raw",
    meta_path: str = r"G:\sunrgbd_raw\SUNRGBDtoolbox\Metadata\SUNRGBDMeta.mat",
    out_path: str = r"G:\RoboGround\data\cache\sunrgbd_index.npz",
    *,
    limit: Optional[int] = None,
    require_files: bool = True,
) -> Path:
    """扫描 SUN RGB-D 元数据，生成紧凑场景索引（一次性，之后秒开）。

    Parameters
    ----------
    limit
        只处理前 N 个场景（调试用；None = 全部）。
    require_files
        是否校验 depth/rgb 文件真实存在（慢但可靠）。

    Returns
    -------
    Path
        索引文件路径。
    """
    try:
        from scipy.io import loadmat  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 scipy 来读取 SUNRGBDMeta.mat：pip install scipy") from exc

    meta_file = Path(meta_path)
    if not meta_file.exists():
        raise FileNotFoundError(
            f"找不到 SUN RGB-D 元数据：{meta_file}\n"
            "请确认数据集已下载，或在 config 里改 data.sunrgbd_meta"
        )

    root_path = Path(root)
    logger.info(f"读取元数据（约 10-30s）：{meta_file}")
    mat = loadmat(str(meta_file), squeeze_me=False)
    if "SUNRGBDMeta" not in mat:
        raise KeyError(f"元数据里没有 SUNRGBDMeta 字段，实际字段：{list(mat)}")
    items = np.asarray(mat["SUNRGBDMeta"]).ravel()
    if limit:
        items = items[: int(limit)]
    logger.info(f"场景总数：{len(items)}")

    sequences: List[str] = []
    Ks: List[np.ndarray] = []
    Rt: List[np.ndarray] = []
    box_flat: List[np.ndarray] = []
    box_offset: List[int] = [0]
    label_flat: List[str] = []
    depth_rel: List[str] = []
    rgb_rel: List[str] = []
    skipped = 0

    def _str_field(obj: Any, name: str) -> str:
        try:
            val = np.asarray(obj[name]).ravel()
            return str(val[0]).strip() if val.size else ""
        except Exception:
            return ""

    for item in items:
        seq = _str_field(item, "sequenceName")
        dname = _str_field(item, "depthname")
        rname = _str_field(item, "rgbname")
        if not seq or not dname:
            skipped += 1
            continue

        d_rel = f"{seq}/depth/{dname}"
        r_rel = f"{seq}/image/{rname}" if rname else ""
        if require_files and not (root_path / d_rel).exists():
            skipped += 1
            continue

        try:
            K = np.asarray(item["K"], dtype=np.float64).reshape(3, 3)
            Rtilt = np.asarray(item["Rtilt"], dtype=np.float64).reshape(3, 3)
        except Exception:
            skipped += 1
            continue

        boxes: List[np.ndarray] = []
        labels: List[str] = []
        gt = item["groundtruth3DBB"] if "groundtruth3DBB" in item.dtype.names else None
        if gt is not None and np.asarray(gt).size > 0:
            for obj in np.atleast_1d(gt).ravel():
                cls = _str_field(obj, "classname")
                if not cls:
                    continue
                try:
                    centroid = np.asarray(obj["centroid"], dtype=np.float64).ravel()[:3]
                    coeffs = np.asarray(obj["coeffs"], dtype=np.float64).ravel()[:3]
                    basis = np.asarray(obj["basis"], dtype=np.float64).reshape(3, 3)
                    heading = float(np.arctan2(basis[1, 0], basis[0, 0]))
                except Exception:
                    continue
                # ⚠️ 关键：`coeffs` 是**全边长**，不是半边长！
                # 标定依据：某场景 "bed" 的 coeffs = (1.89, 2.30, 1.97)
                #   - 当全边长 → 1.9×2.3m 的床 ✔ 合理
                #   - 当半边长 → 3.8×4.6m 的床 ✘ 荒谬
                # "nightstand" coeffs=(0.58,0.55,0.90) → 0.58m 床头柜 ✔ 同样印证。
                # （注：Embodied3D 里写的 `size = 2.0 * coeffs` 会把框放大一倍。）
                size = coeffs
                if np.any(size <= 0):
                    continue
                boxes.append(np.concatenate([centroid, size, [heading]]).astype(np.float32))
                labels.append(canon_class(cls))

        sequences.append(seq)
        Ks.append(K)
        Rt.append(Rtilt)
        if boxes:
            box_flat.append(np.stack(boxes, axis=0))
            box_offset.append(box_offset[-1] + len(boxes))
            label_flat.extend(labels)
        else:
            box_offset.append(box_offset[-1])
        depth_rel.append(d_rel)
        rgb_rel.append(r_rel)

    if not sequences:
        raise RuntimeError(
            "索引为空：没有任何有效场景。请检查 dataset 路径与元数据是否匹配"
        )

    out = Path(out_path)
    ensure_dir(out.parent)
    save_npz(
        out,
        sequences=np.asarray(sequences, dtype="U256"),
        K=np.stack(Ks, axis=0).astype(np.float64),
        Rtilt=np.stack(Rt, axis=0).astype(np.float64),
        box_flat=(np.concatenate(box_flat, axis=0).astype(np.float32)
                  if box_flat else np.zeros((0, 7), np.float32)),
        box_offset=np.asarray(box_offset, dtype=np.int64),
        label_flat=np.asarray(label_flat, dtype="U64"),
        depth_rel=np.asarray(depth_rel, dtype="U256"),
        rgb_rel=np.asarray(rgb_rel, dtype="U256"),
        root=np.asarray([str(root_path)], dtype="U256"),
        meta_path=np.asarray([str(meta_file)], dtype="U256"),
    )
    logger.ok(
        f"索引已保存：{out}（{len(sequences)} 个有效场景，跳过 {skipped} 个）"
    )
    return out


def load_scene_index(path: str) -> Dict[str, Any]:
    """加载场景索引。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"场景索引不存在：{p}\n请先运行：python scripts/build_sunrgbd_index.py"
        )
    data = load_npz(p)
    missing = [k for k in _INDEX_REQUIRED if k not in data]
    if missing:
        raise KeyError(f"索引文件缺少字段 {missing}；建议删除后重建")
    return data


# ==========================================================================
# 单场景加载
# ==========================================================================
def load_sunrgbd_scene(
    index: Dict[str, Any],
    i: int,
    *,
    root: Optional[str] = None,
    max_depth: float = 8.0,
    min_depth: float = 0.2,
    resize: Optional[Tuple[int, int]] = None,
    classes: Optional[Sequence[str]] = None,
) -> Optional[SUNRGBDScene]:
    """按索引位置加载一个场景。

    Parameters
    ----------
    index
        `load_scene_index` 的返回。
    i
        场景下标。
    root
        数据根目录（默认取索引里记录的）。
    resize
        `(width, height)`，可选。会同步缩放内参与 GT 框投影所需的一切。
    classes
        只保留这些类别的 GT 框。

    Returns
    -------
    SUNRGBDScene or None
        读取失败返回 None（例如文件损坏），不抛异常。
    """
    from PIL import Image  # noqa: PLC0415

    n = len(index["sequences"])
    if not (0 <= i < n):
        raise IndexError(f"场景下标 {i} 越界（共 {n} 个）")

    root_path = Path(root) if root else Path(str(index["root"][0]))
    depth_rel = str(index["depth_rel"][i])
    rgb_rel = str(index["rgb_rel"][i])

    # ---- 深度（必须 PIL，opencv 读 16bit PNG 会失败）----
    depth_path = root_path / depth_rel
    if not depth_path.exists():
        logger.warn(f"深度图不存在，跳过：{depth_path}")
        return None
    try:
        raw = np.asarray(Image.open(depth_path))
    except Exception as exc:
        logger.warn(f"读取深度失败（{depth_path}）：{exc}")
        return None
    if raw.ndim != 2:
        logger.warn(f"深度图维度异常 {raw.shape}，跳过：{depth_path}")
        return None

    height, width = raw.shape
    depth_m = raw.astype(np.float32) / DEPTH_SCALE
    depth_m[(depth_m < min_depth) | (depth_m > max_depth) | ~np.isfinite(depth_m)] = 0.0

    # ---- RGB（对齐到深度尺寸）----
    color = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_path = root_path / rgb_rel if rgb_rel else None
    if rgb_path is not None and rgb_path.exists():
        try:
            img = Image.open(rgb_path).convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.BILINEAR)
            color = np.asarray(img, dtype=np.uint8)
        except Exception as exc:
            logger.warn(f"读取 RGB 失败（{rgb_path}）：{exc}")
    else:
        logger.debug(f"RGB 缺失，使用全黑图：{rgb_path}")

    # ---- 相机参数 ----
    K = np.asarray(index["K"][i], dtype=np.float64)
    intrinsics = CameraIntrinsics.from_matrix(K, width=width, height=height)
    # 标定过的 SUN RGB-D 位姿（world → camera = Rtiltᵀ @ Pᵀ），见 CameraPose.from_sunrgbd
    pose = CameraPose.from_sunrgbd(np.asarray(index["Rtilt"][i], dtype=np.float64))

    # ---- GT 3D 框 ----
    off0 = int(index["box_offset"][i])
    off1 = int(index["box_offset"][i + 1])
    boxes = np.asarray(index["box_flat"][off0:off1], dtype=np.float32).reshape(-1, 7)
    labels = [str(x) for x in np.asarray(index["label_flat"][off0:off1]).ravel().tolist()]

    scene = SUNRGBDScene(
        sequence=str(index["sequences"][i]),
        color=color,
        depth_m=depth_m.astype(np.float32),
        intrinsics=intrinsics,
        pose=pose,
        boxes_3d=boxes,
        labels=labels,
        meta={"index": int(i)},
    )

    # ---- 可选：缩放（内参必须同步缩放，否则投影全错）----
    if resize is not None:
        scene = _resize_scene(scene, resize)

    if classes:
        scene = scene.filter_classes(classes)

    return scene


def _resize_scene(scene: SUNRGBDScene, size: Tuple[int, int]) -> SUNRGBDScene:
    """缩放到目标 (width, height)，同步更新内参与深度图。"""
    from PIL import Image  # noqa: PLC0415

    target_w, target_h = int(size[0]), int(size[1])
    if (target_w, target_h) == scene.shape[::-1]:
        return scene

    color = np.asarray(
        Image.fromarray(scene.color).resize((target_w, target_h), Image.BILINEAR),
        dtype=np.uint8,
    )
    depth = np.asarray(
        Image.fromarray(scene.depth_m).resize((target_w, target_h), Image.NEAREST),
        dtype=np.float32,
    )
    sx = target_w / float(scene.intrinsics.width or target_w)
    sy = target_h / float(scene.intrinsics.height or target_h)
    intrinsics = CameraIntrinsics(
        fx=scene.intrinsics.fx * sx,
        fy=scene.intrinsics.fy * sy,
        cx=scene.intrinsics.cx * sx,
        cy=scene.intrinsics.cy * sy,
        width=target_w,
        height=target_h,
    )
    return SUNRGBDScene(
        sequence=scene.sequence,
        color=color,
        depth_m=depth,
        intrinsics=intrinsics,
        pose=scene.pose,
        boxes_3d=scene.boxes_3d,
        labels=list(scene.labels),
        meta=dict(scene.meta),
    )


def scene_to_frame(
    scene: SUNRGBDScene,
    *,
    frame_id: Optional[str] = None,
) -> RGBDFrame:
    """便捷函数：场景 → `RGBDFrame`（等价于 `scene.to_frame()`）。"""
    return scene.to_frame(frame_id=frame_id)


# ==========================================================================
# 数据集迭代器
# ==========================================================================
class SUNRGBDDataset:
    """轻量数据集：支持按索引/切片/随机取样迭代。

    Examples
    --------
    >>> ds = SUNRGBDDataset(index)                 # doctest: +SKIP
    >>> scene = ds[0]                              # doctest: +SKIP
    >>> frames = [s.to_frame() for s in ds.sample(3, seed=0)]  # doctest: +SKIP
    """

    def __init__(
        self,
        index: Dict[str, Any],
        *,
        root: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
        max_depth: float = 8.0,
        min_depth: float = 0.2,
        resize: Optional[Tuple[int, int]] = None,
        require_gt: bool = False,
    ) -> None:
        self.index = index
        self.root = root
        self.classes = classes
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.resize = resize
        n = len(index["sequences"])
        self._indices: List[int] = list(range(n))
        if require_gt:
            offsets = np.asarray(index["box_offset"], dtype=np.int64)
            self._indices = [i for i in self._indices if offsets[i + 1] > offsets[i]]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, key) -> SUNRGBDScene:
        if isinstance(key, slice):
            return [self[i] for i in self._indices[key]]
        i = self._indices[key] if isinstance(key, int) else int(key)
        scene = load_sunrgbd_scene(
            self.index, i,
            root=self.root,
            max_depth=self.max_depth,
            min_depth=self.min_depth,
            resize=self.resize,
            classes=self.classes,
        )
        if scene is None:
            raise RuntimeError(
                f"场景 {i} 加载失败（文件可能缺失或损坏）。"
                "可用 ds.sample() 的重试逻辑绕过，或重建索引"
            )
        return scene

    def sample(self, k: int = 1, *, seed: int = 0, attempts: int = 5) -> List[SUNRGBDScene]:
        """随机取 k 个**能成功加载**的场景（自动重试）。"""
        rng = np.random.default_rng(seed)
        out: List[SUNRGBDScene] = []
        tried = set()
        budget = max(k * attempts, 20)
        while len(out) < k and budget > 0:
            budget -= 1
            pos = int(rng.integers(0, len(self._indices)))
            if pos in tried:
                continue
            tried.add(pos)
            scene = load_sunrgbd_scene(
                self.index, self._indices[pos],
                root=self.root, max_depth=self.max_depth, min_depth=self.min_depth,
                resize=self.resize, classes=self.classes,
            )
            if scene is not None:
                out.append(scene)
        return out

    def label_histogram(self, top: int = 20) -> Dict[str, int]:
        """GT 标签分布（选场景/做实验分组时很有用）。"""
        from collections import Counter  # noqa: PLC0415

        counter: Counter = Counter()
        for i in self._indices:
            off0 = int(self.index["box_offset"][i])
            off1 = int(self.index["box_offset"][i + 1])
            for lab in np.asarray(self.index["label_flat"][off0:off1]).ravel().tolist():
                counter[str(lab)] += 1
        return dict(counter.most_common(top))

    def __repr__(self) -> str:
        return (
            f"SUNRGBDDataset(scenes={len(self)}, classes={self.classes}, "
            f"resize={self.resize})"
        )
