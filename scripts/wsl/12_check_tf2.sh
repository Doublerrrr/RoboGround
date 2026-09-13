#!/usr/bin/env bash
# =============================================================================
# 12 · 检查 tf2 相关包是否齐备（在 WSL 内执行）
# =============================================================================
# 注意：**不能开 `set -u`** —— ROS2 的 setup.bash 会引用未定义变量
# （AMENT_TRACE_SETUP_FILES），开了 -u 会直接 "unbound variable" 退出。
set -o pipefail
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

echo "=== 0. ROS2 环境 ==="
echo "  ROS_DISTRO  = ${ROS_DISTRO:-未设置}"
echo "  ROS_VERSION = ${ROS_VERSION:-未设置}"
echo "  AMENT_PREFIX_PATH 首项 = ${AMENT_PREFIX_PATH%%:*}"
echo "  python3     = $(python3 --version 2>&1)"

echo
echo "=== 1. Python 包导入 ==="
for p in rclpy tf2_ros tf2_geometry_msgs tf2_py tf_transformations \
         geometry_msgs sensor_msgs nav_msgs message_filters std_msgs; do
    printf '  %-22s ' "$p"
    if python3 -c "import $p" 2>/dev/null; then
        echo "OK"
    else
        echo "FAIL"
    fi
done

echo
echo "=== 2. tf2_ros 关键类 ==="
python3 - <<'PY'
import tf2_ros
names = ("Buffer", "BufferInterface", "TransformListener", "TransformBroadcaster",
         "StaticTransformBroadcaster")
for n in names:
    print(f"  {n:<28}", "OK" if hasattr(tf2_ros, n) else "MISSING")
PY

echo
echo "=== 3. 关键消息类型 ==="
python3 - <<'PY'
from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_geometry_msgs import do_transform_point, do_transform_pose
print("  geometry_msgs / sensor_msgs / tf2_geometry_msgs  OK")
PY

echo
echo "=== 4. ros2 CLI 与 tf2 工具 ==="
for c in ros2 ros2- tf2_echo tf2_monitor static_transform_publisher; do
    :
done
command -v ros2 >/dev/null 2>&1 && echo "  ros2                     OK" || echo "  ros2                     FAIL"
ros2 pkg list 2>/dev/null | grep -c '^tf2' | xargs -I{} echo "  tf2 系列包数量           {}"
ros2 pkg executables tf2_ros 2>/dev/null | sed 's/^/    /' | head -8

echo
echo "=== 5. numpy / 项目依赖 ==="
python3 - <<'PY'
for m in ("numpy", "scipy", "yaml", "PIL"):
    try:
        mod = __import__(m)
        print(f"  {m:<12} OK  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  {m:<12} MISSING ({e})")
PY
