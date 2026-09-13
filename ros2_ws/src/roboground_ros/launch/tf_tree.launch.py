# -*- coding: utf-8 -*-
"""完整 TF 树 launch：`ros2 launch roboground_ros tf_tree.launch.py`

为什么需要它
===========
真机上 TF 树由驱动 + URDF + 定位/里程计节点共同维护。本 launch 用**静态发布器**
把同一条 REP-105 标准链搭出来，用途只有一个：
**在没有机器人时验证位姿链路本身是对的**。

    逻辑链: map ──► odom ──► base_link ──► camera_link ──► camera_color_optical_frame
    真机归属: 定位    里程计      URDF 外参        相机驱动

每段边在真机上的归属与是否动态，写在 `roboground_ros/tf_spec.py` 的 `EDGES` 里
（那张表同时被 `tests/test_ros2_package.py` 用来**离线验算**四元数方向）。

⚠️ 诚实说明：这里四条边全是**静态**发布器，所以相机在整个运行期不动。
它验证的是「链路方向 / 四元数 / 单位 / 坐标系约定」，**不是**"移动机器人建图"。
真机移动场景由真机 TF 或 `pose_source:=odometry` 覆盖。
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from roboground_ros.tf_spec import EDGES, resolve_frames


def _q(values):
    """四元数元组 → 六个 flag 字符串列表。"""
    return ["--qx", str(values[0]), "--qy", str(values[1]),
            "--qz", str(values[2]), "--qw", str(values[3])]


def _static_node(index, parent, child, translation, quaternion, owner, logical_child):
    return Node(
        package="tf2_ros", executable="static_transform_publisher",
        # ⚠️ 节点名只能用**逻辑名**（纯字符串）：`LaunchConfiguration` 是要在
        #    launch 期才求值的对象，直接拼进 name 会让 launch 报
        #    "Invalid node name" 而整棵树都起不来（真机上第一次跑就是这么挂的）。
        name=f"tf_{index}_{logical_child}",
        output="log",
        arguments=[
            "--frame-id", parent, "--child-frame-id", child,
            "--x", str(translation[0]), "--y", str(translation[1]),
            "--z", str(translation[2]),
            *_q(quaternion),
        ],
        # 真机归属写在参数里，方便 `ros2 param dump` 时看到"这条边本来是谁发的"
        parameters=[{"roboground_owner": owner}],
    )


def generate_launch_description():
    # 允许把帧名换成自家机器人的命名（不同厂商差别很大）
    args = [
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("odom_frame", default_value="odom"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("camera_link_frame", default_value="camera_link"),
        DeclareLaunchArgument("optical_frame", default_value="camera_color_optical_frame"),
    ]
    rename = {
        "map": LaunchConfiguration("map_frame"),
        "odom": LaunchConfiguration("odom_frame"),
        "base_link": LaunchConfiguration("base_frame"),
        "camera_link": LaunchConfiguration("camera_link_frame"),
        "optical": LaunchConfiguration("optical_frame"),
    }

    nodes = [
        _static_node(i, rename[parent], rename[child], t, q, owner, child)
        for i, (parent, child, t, q, owner, _dyn) in enumerate(EDGES)
    ]

    frames = resolve_frames()
    return LaunchDescription([
        *args,
        LogInfo(msg=["发布完整 TF 树（"
                     + str(len(EDGES)) + " 条边）："
                     + " → ".join([frames["map"], frames["odom"], frames["base_link"],
                                   frames["camera_link"], frames["optical"]])]),
        LogInfo(msg=["注意：全部为静态发布器，真机上 map→odom 与 odom→base_link "
                     "应由定位/里程计模块**动态**发布"]),
        *nodes,
    ])
