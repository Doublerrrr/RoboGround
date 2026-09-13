# -*- coding: utf-8 -*-
"""RoboGround 的 ROS2 封装包。

算法本体在 `roboground` 包里；本包只做 ROS2 侧接线：

| 模块 | 入口 | 作用 |
|---|---|---|
| `perception_node` | `ros2 run roboground_ros perception` | RGB-D + 位姿 → 3D 语义地图 |
| `query_node` | `ros2 run roboground_ros query` | 自然语言问题 → 结构化答案 |
| `launch_args` | — | 两个 launch 文件共用的参数/节点构造（唯一构造点） |
| `tf_spec` | — | 完整 TF 树的纯数据描述（零依赖，可离线验算） |

参数集中在 `perception_node.PARAM_MAP` 一张表里（`参数名 → Config 路径`）。

⚠️ `__init__.py` **必须保持空导入**：ROS2 侧模块（`launch_args`）依赖 `launch`，
一旦在这里 import 就会让离线测试也无法 import 本包。
"""
__version__ = "0.1.0"
