"""配置系统：YAML → 嵌套 Config 对象，支持点号访问与深度合并。

设计目标
--------
1. **零配置可跑**：`load_config()` 不传参数时返回完整默认配置，
   所有模块都能在没有任何 yaml 文件的情况下运行（测试/CI 友好）。
2. **部分覆盖**：用户的 yaml 只需要写想改的字段，其余从 DEFAULT_CONFIG 继承。
3. **点号访问**：`cfg.geometry.voxel_size` 与 `cfg.get("geometry.voxel_size")` 等价。
4. **命令行友好**：`apply_overrides(["geometry.voxel_size=0.02"])` 支持 CLI 覆盖。

示例
----
>>> cfg = load_config("configs/default.yaml", overrides=["mapping.voxel_size=0.03"])
>>> cfg.mapping.voxel_size
0.03
>>> cfg.get("perception.detector")
'stub'
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

try:  # PyYAML 是硬依赖，但保持友好的报错
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "需要 PyYAML：pip install PyYAML（lxr 环境已自带）"
    ) from exc


# --------------------------------------------------------------------------
# 默认配置：所有可调项的单一事实来源（single source of truth）
# --------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "RoboGround",
        "seed": 42,
        "device": "cuda",           # cuda | cpu（无 GPU 时自动回退）
        "output_dir": "runs/default",
        "verbose": True,
    },

    # ---------------- Stage 1: 开放词汇 2D 感知 ----------------
    "perception": {
        # 三个后端彼此独立、可插拔；每个都有离线降级实现
        "detector": "stub",         # stub | grounding_dino | yolo
        "segmenter": "box",         # box | sam | stub
        "encoder": "color_hist",    # color_hist | dinov2 | clip

        # 自然语言提示词（开放词汇的核心：用文本而不是固定 label 指定类别）
        "prompts": ["cup", "table", "chair", "door", "monitor", "bottle"],

        "detector_kwargs": {
            "box_threshold": 0.30,
            "text_threshold": 0.25,
            "model_id": "IDEA-Research/grounding-dino-tiny",
            "max_detections": 64,
        },
        "segmenter_kwargs": {
            "model_id": "facebook/sam-vit-base",
            "multimask_output": False,
        },
        "encoder_kwargs": {
            "model_id": "facebook/dinov2-small",   # 或 openai/clip-vit-base-patch32
            "feature_dim": 256,                     # 统一投影维度（各后端对齐）
            "pool": "mask_mean",                    # mask_mean | bbox_mean | cls
        },
        # 感知结果缓存目录（避免重复推理）
        "cache_dir": "data/cache/perception",
    },

    # ---------------- 几何层 ----------------
    "geometry": {
        "depth_scale": 1000.0,      # raw depth → meters：depth_m = raw / depth_scale
        "min_depth": 0.10,          # 米，过近丢弃（噪声大）
        "max_depth": 8.00,          # 米，过远丢弃（精度差）
        "voxel_size": 0.05,         # 米，体素边长（3D 特征场分辨率）
        "max_points_per_frame": 200_000,
        "depth_trunc_percentile": 99.0,   # 幽灵点过滤：超过该分位数的深度丢弃
        "ransac_plane_filter": False,     # 是否用 RANSAC 剔除地面（耗时）
    },

    # ---------------- Stage 2: 3D 语义地图 ----------------
    "mapping": {
        "fusion": "mean",           # mean | max | confidence_weighted
        "min_observations": 1,      # 一个体素被观测到几次才写入地图
        "conf_threshold": 0.0,      # 低于该置信度的观测丢弃
        # --- 物体构建方式 ---
        # "association"（默认，推荐）：按"实例身份"跨帧关联 —— 每个检测就是一个
        #     实例，用 标签兼容性 + 中心距离/IoU 把不同帧的同一物体串起来。
        #     这是真实机器人系统的做法，不会把相邻的两个物体粘成一个。
        # "clustering"：对体素做 DBSCAN 连通域聚类。**当没有实例信息时**才用
        #     （例如只加载了一张体素网格）；它的问题是两个靠近的物体会被合并。
        "object_mode": "association",
        "assoc_radius": 0.60,       # 米，关联同一物体的中心距离阈值
        "assoc_iou": 0.10,          # 3D bbox IoU 超过此值也判为同一物体（>1 = 关闭）
        "assoc_require_label": True,  # 关联时是否要求标签语义兼容
        # 关联的**分配方式**：
        # "hungarian"（默认）：逐帧全局最优一对一匹配（scipy linear_sum_assignment），
        #     即"同一条轨迹在一帧里最多认领一个观测"。
        # "greedy"：逐观测贪心（旧行为），每个观测独立挑最高分轨迹。
        # 两者共用同一个打分函数（`MapBuilder._match_score`），唯一区别是
        # "一帧里一条轨迹能不能认领多个观测"。消融依据（12 个真实采集点，
        # scripts/42_ablate_association.py）：greedy 碎裂率 0.79 / 命中 95.1% /
        # 中心误差中位 0.502 m；hungarian 碎裂率 1.04 / 命中 98.1% / 0.425 m。
        "assoc_strategy": "hungarian",
        # 同帧内"其实是同一个物体"的重复检测合并门限（轴对齐 3D IoU）。
        # ⚠️ 默认 **None = 关闭**：实测在针孔帧上把"同标签 + IoU ≥ 0.5"的
        # 检测也合并会变差（命中率 98.1% → 96.7%）—— 并排的两个真不同物体
        # 包围盒本来就会重叠。真正需要合并的是**等距柱状跨接缝**那一种，
        # 由图像边界签名判（`MapBuilder._merge_same_frame`），与本项无关。
        "assoc_merge_iou": None,
        "use_3d_clustering": True,  # object_mode=clustering 时是否启用
        "cluster_eps": 0.08,        # 米，DBSCAN 邻域半径
        "cluster_min_samples": 3,
        "max_voxels": 2_000_000,    # 内存保护上限
        # 物体级聚合
        "object_min_voxels": 5,
        "object_bbox_percentile": 2.0,   # bbox 用分位数裁剪，抗离群点
        "object_max_points": 10_000,     # 每个物体保留的点采样上限（算 bbox 用）
    },

    # ---------------- 语言查询 ----------------
    "query": {
        "top_k": 5,
        "score": "cosine",          # cosine | dot | l2
        # 词法路径的接受阈值。
        #
        # ⚠️ 曾经的注释写着"命中就是 1.0、不命中就是 0，所以阈值可以很低"——
        # **这个假设是错的**，也是本项长期带病的原因。词法打分实际有三层：
        #   ① 规范概念命中 1.00  ② 子串包含 0.85  ③ 字符 bigram Jaccard ≤ 0.80
        # 第三层对**任意**两个词都给部分分（bicycle↔bottle 共享 "le" → 0.145），
        # 属于纯噪声。阈值必须**高于噪声层上限、低于真信号下限**：
        # 实测 0.05 → 27.8% 的未见类别查询返回错误物体；0.5 → 拒识率 100%
        # 且命名类命中率不降（综合分 0.603→0.828）。
        # 复现：`python scripts/16_eval_openvocab_query.py`
        # 回归锁：tests/test_query.py::test_lexical_threshold_rejects_ngram_noise
        #         tests/test_pipeline_e2e.py::test_e2e_config_path_does_not_lower_lexical_threshold
        "min_score_lexical": 0.5,
        # 旧键名，含义等同 min_score_lexical，仅为兼容历史配置保留。
        # **不要再往这里写低值** —— 它会在 builder 里覆盖上面的阈值。
        "min_score": None,
        # 嵌入路径阈值：**null 表示按编码器自适应**。
        # 不同后端分数尺度差两个数量级（CLIP 校准后命中 ~0.9，
        # SigLIP sigmoid 概率命中只有 ~0.03），硬编码一个值必然坏掉一边。
        # 需要固定值时在这里写数字。
        "min_score_embedding": None,
        "spatial_rerank": True,     # 用空间关系对候选重排
    },

    # ---------------- Stage 3: 推理 ----------------
    "reasoning": {
        "backend": "rules",         # rules | qwen2vl
        # --- rules 后端：纯几何推理，离线可用、可解释 ---
        "relations": {
            "near_threshold": 0.50,     # 米，两个物体小于此距离算 "near"
            "above_delta": 0.10,        # 米，z 差超过此值算 "above"
            "inside_ratio": 0.60,       # bbox 包含比例超过此值算 "inside"
            "axis_ratio": 0.15,         # 归一化分离量低于此值视为"无明显方向"
            "adjacent_gap": 0.15,       # 米，水平表面间隙小于此值算"旁边"
        },
        # --- VLM 后端 ---
        "vlm": {
            # ★ 总开关。默认关：开箱即用不需要显存，也不会在没下模型时报错。
            #   `deployment/ros2/nodes.py` 的 QueryNode 会按 ROS2 参数 `use_vlm` 覆盖它。
            #   （这个键曾经被代码 `cfg.set` 出来但**没写进默认配置** —— 又一个
            #    "代码里有开关、配置文件里看不见"，由 tests/test_config_paths.py 抓出。）
            "enabled": False,
            "model_id": "Qwen/Qwen2-VL-2B-Instruct",
            "adapter_path": None,       # LoRA 权重路径（训练后填）
            "max_new_tokens": 256,
            "load_in_4bit": True,       # 8GB 显存必须开
            "device_map": "auto",
        },
    },

    # ---------------- 数据 ----------------
    "data": {
        "root": "data",
        # 直接复用 Embodied3D 已预处理好的 SUN RGB-D（点云+RGB+内参+3D box）
        "sunrgbd_processed": r"G:\Embodied3D\data\processed",
        "sunrgbd_meta": r"G:\sunrgbd_raw\SUNRGBDtoolbox\Metadata\SUNRGBD2Dseg\SUNRGBD",
        "sequence_length": 8,       # 一次建图用多少帧
        "image_size": [480, 640],   # H, W
        "num_workers": 0,           # ⚠️ 本机沙箱必须为 0，别改
        # 自动标注（`data/auto_label.py`）的四道闸门参数。
        # ★ 这两项曾被 `auto_label.py` 读取但**没写进本默认配置** ——
        #   于是"代码里有开关、YAML 里配不了"。已由
        #   `runs/_audit_config_paths.py` 审计出并补齐。
        "auto_label_min_points": 20,      # 少于这么多点的检测框丢弃
        "auto_label_check_scale": True,   # 是否用 GT box 尺度做合理性检查
    },

    # ---------------- Stage 4: 部署 ----------------
    "deploy": {
        "fast_tier_hz": 10.0,       # 高频感知（避障级）
        "slow_tier_hz": 1.0,        # 低频 VLM 推理
        "queue_size": 4,
        "quantize": "none",         # none | fp16 | int8
        "onnx_opset": 17,
        "ros2": {
            "node_namespace": "/roboground",
            "topics": {
                "color_image": "/camera/color/image_raw",
                "depth_image": "/camera/depth/image_raw",
                "camera_info": "/camera/color/camera_info",
                "semantic_map": "/roboground/semantic_map",
                "query": "/roboground/query",
                "answer": "/roboground/answer",
            },
            # 相机位姿来源（决定建的是"世界地图"还是"局部地图"）
            "pose": {
                # tf | odometry | static | identity
                # ⚠️ identity 只用于占位与测试：建出来的地图以第一帧相机为原点，
                #    真机上不可用（不同时刻的帧会被错误地叠在同一位置 → 重影）
                "source": "identity",
                "target_frame": "map",
                # 用 *_optical_frame 可以省掉轴纠正（推荐，相机驱动都会发布它）
                "source_frame": "camera_color_optical_frame",
                # 只有查到的是 camera_link（非 optical）时才需要设 True
                "optical_frame_correction": False,
                "timeout_s": 0.1,
                # 启动期等待位姿源就绪的最长时间（秒）。
                # 真机启动顺序常常是"感知节点先起、TF/里程计后到"，
                # 不等一下就会把开头若干帧**静默丢弃**（只是地图少几帧观测）。
                # `TfPoseProvider.wait_ready` / `OdometryPoseProvider.wait_ready` 用它。
                "ready_timeout_s": 5.0,
                "odom_topic": "/odom",
                "translation": [0.0, 0.0, 0.0],
                "rpy": [0.0, 0.0, 0.0],
                # identity 兜底：查到位姿失败时是否退回恒等位姿。
                # ★ 默认 False = **宁可丢帧也不伪造位姿**（伪造会把不同时刻的
                #   帧错误地叠在同一处 → 地图重影，而且完全静默）。
                #   需要兜底必须显式开；`build_pose_provider` 里也做了硬校验。
                "fallback_identity": False,
            },
            # 三路传感器消息（彩色/深度/内参）的时间同步容差（秒）
            "sync_slop": 0.05,
            # 每 N 帧发布一次地图（省带宽；调试用 1 能看到每一帧的变化）
            # ★ 曾被 `nodes.py` 读取但没写进本默认配置 → 已补齐
            "publish_every_n": 10,
            # 深度图与彩色图分辨率不一致时，是否按内参自动对齐深度到彩色
            "autosync_depth": False,
        },
    },

    # ---------------- 评测 ----------------
    "eval": {
        "iou_thresholds": [0.25, 0.50],
        "top_k": [1, 5],
        "latency_warmup": 3,
        "latency_iters": 20,
    },
}


# --------------------------------------------------------------------------
# Config 对象
# --------------------------------------------------------------------------
class Config:
    """嵌套字典的薄封装，支持属性访问 + 点号路径访问。

    内部始终持有一个纯 dict（`_data`），嵌套 dict 在读取时才包装成 Config，
    因此 `cfg.geometry.voxel_size` 与 `cfg["geometry"]["voxel_size"]` 都能用。
    """

    __slots__ = ("_data",)

    def __init__(self, data: Optional[Mapping[str, Any]] = None) -> None:
        object.__setattr__(self, "_data", dict(data or {}))

    # ---------------- 构造 / 序列化 ----------------
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在：{path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"配置文件顶层必须是 mapping，实际是 {type(raw).__name__}")
        return cls(raw)

    def to_dict(self) -> Dict[str, Any]:
        """返回深拷贝的纯 dict（可直接 json.dump）。"""
        return copy.deepcopy(self._data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._data, indent=indent, ensure_ascii=False)

    def save(self, path: Union[str, Path]) -> Path:
        """把当前配置（含覆盖）落盘，便于实验可复现。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self._data, fh, allow_unicode=True, sort_keys=False)
        return path

    # ---------------- 访问 ----------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        for key, value in self._data.items():
            yield key, (Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        """点号路径读取：`cfg.get("perception.detector_kwargs.box_threshold")`。

        路径不存在时返回 default（不抛异常），方便写"可选配置"逻辑。
        """
        node: Any = self._data
        for part in str(key).split("."):
            if isinstance(node, Config):
                node = node._data
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def require(self, key: str) -> Any:
        """点号路径读取，缺失即报错（用于关键配置）。"""
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(f"缺少必需配置项：{key}")
        return value

    def set(self, key: str, value: Any) -> "Config":
        """点号路径写入，中间层级不存在会自动创建。"""
        parts = str(key).split(".")
        node = self._data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value
        return self

    # ---------------- 合并 ----------------
    def merge(self, other: Union[Mapping[str, Any], "Config"], *, inplace: bool = False) -> "Config":
        """深度合并：other 覆盖 self（递归到叶子）。"""
        base = self._data if inplace else copy.deepcopy(self._data)
        payload = other.to_dict() if isinstance(other, Config) else dict(other)
        merged = _deep_merge(base, payload)
        return self if inplace else Config(merged)

    def apply_overrides(self, overrides: Iterable[str]) -> "Config":
        """应用 `a.b.c=value` 形式的覆盖（CLI 常用）。返回自身便于链式调用。"""
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"覆盖项必须形如 key=value，收到：{item!r}")
            key, raw = item.split("=", 1)
            self.set(key.strip(), parse_scalar(raw.strip()))
        return self

    # ---------------- 属性访问 ----------------
    def __getattr__(self, name: str) -> Any:
        # 只在常规查找失败后被调用；下划线开头视为内部属性，避免污染
        if name.startswith("_"):
            raise AttributeError(name)
        data = object.__getattribute__(self, "_data")
        if name in data:
            value = data[name]
            return Config(value) if isinstance(value, dict) else value
        raise AttributeError(
            f"配置中没有 {name!r}；可用键：{sorted(data.keys())}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __repr__(self) -> str:
        return f"Config({self.to_json(indent=2)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Config):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented


# --------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并两个 dict；override 中的 dict 与 base 中 dict 逐层合并。

    非 dict 值（含 list）一律整体替换 —— 这是有意的：
    列表语义通常是"完整替换"而非"逐元素合并"。
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_scalar(raw: str) -> Any:
    """把 CLI 字符串解析成 python 标量（bool/int/float/None/list/dict/str）。"""
    lowered = raw.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", "~"}:
        return None
    # 尝试 YAML 解析（能覆盖 int/float/list/dict/带引号字符串）
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    if isinstance(parsed, (int, float, list, dict)) or parsed is None or isinstance(parsed, bool):
        return parsed
    return raw


def resolve_device(requested: str = "cuda") -> str:
    """把 'cuda' 解析成实际可用设备；无 GPU 时静默回退到 'cpu'。

    注意：只有真的尝试 import torch 才判断，避免无 torch 环境下 import 本模块失败。
    """
    if requested != "cuda":
        return requested
    if os.environ.get("ROBOGROUND_FORCE_CPU") == "1":
        return "cpu"
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - 无 torch 时的降级
        return "cpu"


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Iterable[str]] = None,
    *,
    use_env: bool = True,
) -> Config:
    """加载配置：DEFAULT_CONFIG ← yaml ← overrides ← 环境变量。

    Parameters
    ----------
    path
        yaml 路径。为 None 时只用默认配置。
    overrides
        `["a.b=1", "c.d=e"]` 形式的覆盖列表。
    use_env
        是否读取 `ROBOGROUND_<SECTION>__<KEY>` 形式的环境变量覆盖，
        例如 `ROBOGROUND_DATA__ROOT=D:/data`。

    Returns
    -------
    Config
    """
    cfg = Config(copy.deepcopy(DEFAULT_CONFIG))

    if path is not None:
        file_cfg = Config.from_yaml(path)
        cfg = cfg.merge(file_cfg)

    if overrides:
        cfg.apply_overrides(overrides)

    if use_env:
        _apply_env_overrides(cfg)

    return cfg


def _apply_env_overrides(cfg: Config) -> None:
    """`ROBOGROUND_<SEC>__<KEY>=value` → cfg.<sec>.<key>（全小写）。"""
    prefix = "ROBOGROUND_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix) or env_key in {"ROBOGROUND_FORCE_CPU"}:
            continue
        dotted = env_key[len(prefix):].lower().replace("__", ".")
        if len(dotted.split(".")) < 2:
            continue
        cfg.set(dotted, parse_scalar(env_val))
