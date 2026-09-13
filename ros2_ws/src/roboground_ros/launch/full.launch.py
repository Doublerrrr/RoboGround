# -*- coding: utf-8 -*-
"""全链路 launch：`ros2 launch roboground_ros full.launch.py`

一次起：感知节点（快档）+ 问答节点（慢档）+ 可选完整 TF 树。

**这正是 ROS2 相比"手敲脚本"的价值** —— 真机上一条命令起整个感知子系统，
参数集中、可版本化、可被上层（导航 / 调度）拉起。

用法::

    ros2 launch roboground_ros full.launch.py
    ros2 launch roboground_ros full.launch.py static_tf:=true       # 无真机时
    ros2 launch roboground_ros full.launch.py use_vlm:=true         # 开 VLM 推理
    ros2 launch roboground_ros full.launch.py use_sim_time:=true    # 回放 bag

**与 `perception.launch.py` 的关系**：两者都从 `roboground_ros.launch_args`
取参数与节点构造，所以参数名/默认值/类型处理**不可能不一致**
（历史教训：两处各写一份，改了一处忘了另一处）。
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration

from roboground_ros.launch_args import (
    launch_arguments,
    perception_node,
    query_node,
    static_tf_include,
)
from roboground_ros.tf_spec import frame_names


def generate_launch_description():
    return LaunchDescription([
        *launch_arguments(),
        DeclareLaunchArgument("use_vlm", default_value="false",
                              description="问答节点是否启用 VLM（需要显存，默认关）"),
        LogInfo(msg=["启动 RoboGround 全链路：感知（快档）+ 问答（慢档）"]),
        LogInfo(msg=["TF 树帧：" + ", ".join(frame_names())
                     + "（static_tf:=true 时由本 launch 发布，真机上由驱动/URDF 负责）"]),
        perception_node(),
        query_node(),
        static_tf_include(),
    ])
