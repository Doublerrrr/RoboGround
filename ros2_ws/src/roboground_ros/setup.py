# -*- coding: utf-8 -*-
"""ament_python 打包配置。

为什么需要这个文件（而不是"直接 python 跑脚本就行"）
==================================================
`python -c "from roboground... import spin_perception"` 这种跑法有三个问题：

1. **不进入 ROS2 的包索引** —— `ros2 pkg list` 看不到、`ros2 run` 起不来，
   别人拿到代码不知道怎么用；
2. **没有参数声明** —— 想调阈值只能改 yaml 重启，真机上不可接受；
3. **没有 launch 编排** —— 起一个系统要开好几个终端。

打成 ament 包之后：`colcon build` → `ros2 run roboground_ros perception`
→ `ros2 launch roboground_ros perception.launch.py`，
这才是一个**能交付**的形态。

依赖 `roboground`（算法本体）通过 pip 安装，见 `package.xml` 的说明与
`ros2_ws/README.md` 的安装步骤。
"""
from setuptools import setup

package_name = "roboground_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        # ament 资源索引：`ros2 pkg` 靠它找到这个包
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # launch 与配置要装到 share 下，`ros2 launch` 才能按包名找到
        ("share/" + package_name + "/launch", [
            "launch/perception.launch.py",
            "launch/full.launch.py",
            "launch/tf_tree.launch.py",
        ]),
        ("share/" + package_name + "/config", ["config/roboground.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RoboGround",
    maintainer_email="roboground@example.com",
    description="RoboGround 的 ROS2 封装（感知 / 问答节点）",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # 这两个是 `ros2 run roboground_ros <name>` 的入口
            "perception = roboground_ros.perception_node:main",
            "query = roboground_ros.query_node:main",
        ],
    },
)
