# -*- coding: utf-8 -*-
"""感知节点 launch：`ros2 launch roboground_ros perception.launch.py`

launch 的价值（对比"手敲 ros2 run"）
==================================
1. **一次起全套** —— 节点 + 可选 TF 树 + 参数文件，不用开好几个终端；
2. **参数可版本化** —— 不用每次记一长串 `--ros-args -p`；
3. **可组合** —— `full.launch.py` 复用同一个构造模块，参数不可能两处不一致。

用法::

    ros2 launch roboground_ros perception.launch.py
    ros2 launch roboground_ros perception.launch.py pose_source:=static   # 没有 TF 时
    ros2 launch roboground_ros perception.launch.py static_tf:=true       # 顺带发完整 TF 树
    ros2 launch roboground_ros perception.launch.py params_file:=/path/to/my.yaml

参数优先级（与 nav2 相同的约定）
=============================
    内置默认  <  params_file（默认 config/roboground.yaml）  <  命令行 key:=value
                                                             <  运行时 ros2 param set

即**本文件的 launch 参数默认值会覆盖 params_file 的同名项**，所以两处默认值
必须一致 —— 由 `tests/test_ros2_package.py` 逐项断言（见 `LAUNCH_OVERRIDES`）。
"""
from launch import LaunchDescription
from launch.actions import LogInfo
from launch.substitutions import LaunchConfiguration

from roboground_ros.launch_args import (
    launch_arguments,
    perception_node,
    static_tf_include,
)


def generate_launch_description():
    return LaunchDescription([
        *launch_arguments(),
        LogInfo(msg=["启动感知节点，位姿来源=", LaunchConfiguration("pose_source"),
                     "（pose_source=tf 需要 TF 树；没有就加 static_tf:=true）"]),
        perception_node(),
        static_tf_include(),
    ])
