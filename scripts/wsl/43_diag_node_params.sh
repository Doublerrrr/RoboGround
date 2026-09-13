#!/usr/bin/env bash
# 43 · 诊断：真 perception 节点的参数服务（本地排查用）
set +u
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source /mnt/g/RoboGround/ros2_ws/install/setup.bash >/dev/null 2>&1
set -u

python3 - <<'PY' 2>&1 | tail -30
import rclpy
from roboground.deployment.ros2.nodes import PerceptionNode

rclpy.init()
n = PerceptionNode(node_name="perception")
print("OK node =", n.get_name())
print("参数个数 =", len(n._parameters))
print("sync_slop cfg      =", n.cfg.get("deploy.ros2.sync_slop"))
print("pose_source cfg    =", n.cfg.get("deploy.ros2.pose.source"))
print("publish_every_n    =", n.publish_every)
print("热属性 min_depth   =", n.min_depth)
n.destroy_node()
rclpy.shutdown()
PY
