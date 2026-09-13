"""RoboGround —— 面向服务机器人的开放词汇 3D 场景理解与语言接地系统。

Pipeline
--------
Stage 1  perception   开放词汇 2D 感知（检测 / 分割 / 特征编码）
Stage 2  mapping      RGB-D → 开放词汇 3D 语义地图（本项目核心）
Stage 3  reasoning    语言查询 + 空间关系 + 米制距离推理
Stage 4  deployment   量化 / ONNX 导出 / 异步流水线 / ROS2 节点
Stage 5  data         自动化标注 + 合成数据 + 评测闭环

快速上手
--------
>>> from roboground import load_config, SemanticMap, MapBuilder
>>> cfg = load_config("configs/default.yaml")
>>> builder = MapBuilder(cfg)
>>> semantic_map = builder.build_from_frames(frames)
>>> hits = semantic_map.query_text("杯子在哪", top_k=3)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "load_config",
    "Config",
    "DEFAULT_CONFIG",
    "SemanticMap",
    "SemanticObject",
    "MapBuilder",
    "QueryResult",
]

from roboground.config import DEFAULT_CONFIG, Config, load_config
from roboground.mapping.builder import MapBuilder
from roboground.mapping.semantic_map import QueryResult, SemanticMap, SemanticObject
