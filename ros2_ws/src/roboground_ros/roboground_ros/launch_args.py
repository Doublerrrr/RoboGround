# -*- coding: utf-8 -*-
"""launch 文件共用的构造逻辑（`perception.launch.py` 与 `full.launch.py` 都用它）。

为什么单独抽出来
==============
两个 launch 文件如果各写一份参数名/默认值，就一定会出现
"改了一个忘了另一个"。历史事故已经发生过一次（两处 `sync_slop` 一个 0.05
一个 0.1，改哪里生效取决于用哪个 launch）。本模块是**唯一**构造点。

本模块依赖 ROS2（launch / launch_ros），所以**不能**被离线测试直接 import；
纯数据（`PARAM_MAP` / `PARAM_DEFAULTS` / `LAUNCH_OVERRIDES`）放在
`perception_node.py` 里，那边零依赖、可离线测试。
"""
from __future__ import annotations

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from roboground_ros.perception_node import LAUNCH_OVERRIDES


def launch_arguments(with_knobs: bool = True):
    """所有 launch 参数（含默认值 = 与 YAML 一致的 `LAUNCH_OVERRIDES`）。"""
    pkg = FindPackageShare("roboground_ros")
    args = [
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([pkg, "config", "roboground.yaml"]),
            description="ROS2 参数文件（默认用包内 config/roboground.yaml）"),
        DeclareLaunchArgument("config_file", default_value="",
                              description="roboground 自己的 YAML 配置（留空用默认配置）"),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false",
            description="回放 rosbag / 仿真时必须设为 true，否则 TF 按墙上时钟查、必然失败"),
        DeclareLaunchArgument(
            "static_tf", default_value="false",
            description="是否顺带发布完整 TF 树（无真机时用；真机上由驱动/URDF 负责）"),
    ]
    if with_knobs:
        args += [
            DeclareLaunchArgument("pose_source",
                                  default_value=LAUNCH_OVERRIDES["pose_source"],
                                  description="位姿来源：tf | odometry | static | identity"),
            DeclareLaunchArgument("sync_slop",
                                  default_value=LAUNCH_OVERRIDES["sync_slop"],
                                  description="三路传感器时间同步容差（秒）"),
            DeclareLaunchArgument("publish_every_n",
                                  default_value=LAUNCH_OVERRIDES["publish_every_n"],
                                  description="每处理 N 帧发布一次地图"),
        ]
    return args


def _param_overrides(*, include_knobs: bool = True):
    """命令行参数 → 节点参数的映射（**显式声明类型**）。

    为什么必须用 `ParameterValue(..., value_type=...)`：
    launch 的 `LaunchConfiguration` 求值结果是**字符串**，直接塞进参数字典会让
    `sync_slop` 变成 `"0.05"`（字符串）而不是 `0.05`（double）。下游恰好有
    `float()` 兜住纯属侥幸 —— 一旦有消费方不做转换就会炸在真机上。
    """
    overrides = {
        "config_file": LaunchConfiguration("config_file"),
        "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"),
                                       value_type=bool),
    }
    if include_knobs:
        overrides.update({
            "pose_source": LaunchConfiguration("pose_source"),
            "sync_slop": ParameterValue(LaunchConfiguration("sync_slop"), value_type=float),
            "publish_every_n": ParameterValue(LaunchConfiguration("publish_every_n"),
                                              value_type=int),
        })
    return overrides


def perception_node():
    """感知节点（快档）。"""
    return Node(
        package="roboground_ros",
        executable="perception",
        name="perception",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), _param_overrides()],
    )


def query_node():
    """问答节点（慢档）。默认关 VLM —— 开箱即用优先。"""
    return Node(
        package="roboground_ros",
        executable="query",
        name="query",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            _param_overrides(include_knobs=False),
            {"use_vlm": ParameterValue(LaunchConfiguration("use_vlm"), value_type=bool)},
        ],
    )


def static_tf_include():
    """可选：把完整 TF 树一起起起来（`static_tf:=true`）。"""
    pkg = FindPackageShare("roboground_ros")
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg, "launch", "tf_tree.launch.py"])),
        condition=IfCondition(LaunchConfiguration("static_tf")),
    )
