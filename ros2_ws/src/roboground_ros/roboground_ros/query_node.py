# -*- coding: utf-8 -*-
"""问答节点入口：`ros2 run roboground_ros query`。

参数（尤其 `use_vlm`）由 `QueryNode` 自己声明 —— 理由见
`roboground/deployment/ros2/params.py` 的模块 docstring：
**声明参数的节点必须是干活的节点**，否则 `--params-file` 按节点名匹配不上。
"""
from __future__ import annotations

import sys

from roboground.deployment.ros2.params import (  # noqa: F401
    COLD_PARAMS,
    HOT_PARAMS,
    PARAM_MAP,
    declare_params,
    plan_param_update,
)


def main(args=None) -> int:
    import rclpy

    from roboground.deployment.ros2.nodes import QueryNode, ROS2_AVAILABLE

    if not ROS2_AVAILABLE:
        print("[FAIL] 环境里没有 rclpy。请先 source ROS2 的 setup.bash。", file=sys.stderr)
        return 2

    rclpy.init(args=args)
    node = QueryNode(node_name="query")
    node.get_logger().info(
        f"问答节点已启动（VLM={node.use_vlm}）；"
        "开关 VLM：ros2 param set /query use_vlm false（需重启生效：推理器按需构造）")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
