# -*- coding: utf-8 -*-
"""感知节点入口：`ros2 run roboground_ros perception`。

为什么这个文件很短（刻意的）
==========================
ROS2 侧的接线逻辑（参数表、参数声明、冷热参数分类）都在
`roboground.deployment.ros2.params` 里，因为**声明参数的节点必须是干活的节点**：

    `--params-file` 的顶层键就是节点名，必须与运行时节点名完全一致。

早期实现是"另起一个 `roboground_param_bootstrap` 节点读参数、算 Config，
再交给真节点用"，实测（`scripts/wsl/42_probe_param_matching.py`）证明
bootstrap 读到的是默认值、真节点读到的是文件里的值，**静默不一致**。
改成"真节点自己声明参数"之后：

  · `ros2 param get/list/dump /perception` 能看到**真实生效值**；
  · `ros2 param set /perception sync_slop 0.42` 在线生效（热参数）；
  · 冷参数会被**显式拒绝**并说明原因，而不是静默忽略。

所以本文件只剩"起节点、spin、收尾"三件事 —— ament 包该有的薄。
"""
from __future__ import annotations

import sys

#: 参数名 → roboground Config 路径（**唯一接线点**在库里，这里只是再导出，
#: 方便外部脚本与测试从一个显然的地方拿到它）。
from roboground.deployment.ros2.params import (  # noqa: F401
    COLD_PARAMS,
    HOT_PARAMS,
    PARAM_MAP,
    declare_params,
    plan_param_update,
)

#: launch 文件传给节点的默认值（字符串形式，因为 launch 参数都是字符串）。
#:
#: ★ 必须与 `config/roboground.yaml` 里同名项**逐项一致**：
#:   launch 命令行参数会覆盖 params_file，两处不一致就会出现
#:   "我明明改了 YAML 怎么没生效"。`tests/test_ros2_package.py` 断言两者相等。
LAUNCH_OVERRIDES = {
    "pose_source": "tf",
    "sync_slop": "0.05",
    "publish_every_n": "10",
}


def main(args=None) -> int:
    import rclpy

    from roboground.deployment.ros2.nodes import PerceptionNode, ROS2_AVAILABLE

    if not ROS2_AVAILABLE:
        print("[FAIL] 环境里没有 rclpy。请先 source ROS2 的 setup.bash。", file=sys.stderr)
        return 2

    rclpy.init(args=args)
    # ★ 不传 cfg —— 让节点自己声明参数并据其构造配置。
    #   节点名由 rclpy 从 `__node` 解析（launch 的 `name:=` 也是走这个），
    #   所以参数文件必然匹配得上。
    node = PerceptionNode(node_name="perception")
    node.get_logger().info(
        "感知节点已启动：可用 `ros2 param list /perception` 查看参数；"
        "热参数（" + ", ".join(HOT_PARAMS) + "）支持 ros2 param set 在线修改")
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
