#!/usr/bin/env bash
# 60 · 完整 TF 树端到端验证：真起 `tf_tree.launch.py`，再跑感知节点走 TF 位姿。
#
# 为什么要"真起 launch"而不是在测试脚本里自己广播 TF
# =================================================
# 因为要验证的**正是那个 launch 文件**。自己广播一遍只能证明"我广播的数是对的"，
# 证明不了"交付给别人的 launch 文件是对的"。历史上 `${full.launch.py}` 里的
# 静态 TF 就带着一个方向错误（四元数写成转置），而且从来没被真跑过。
#
# 用法：
#     bash /mnt/g/RoboGround/scripts/wsl/60_verify_tf_tree.sh
set -uo pipefail

set +u
source /opt/ros/humble/setup.bash
source /mnt/g/RoboGround/ros2_ws/install/setup.bash
set -u

LOG=/tmp/_rg_tf_tree_launch.log
rm -f "$LOG"

echo "================================================================"
echo "60 · 完整 TF 树端到端验证"
echo "================================================================"

if [ ! -f /mnt/g/RoboGround/ros2_ws/install/setup.bash ]; then
    echo "[FAIL] 工作空间没构建，请先跑 40_build_ros2_pkg.sh"
    exit 2
fi

# ---- 1) 真起 launch -------------------------------------------------------
echo
echo "[1/4] 启动 ros2 launch roboground_ros tf_tree.launch.py"
ros2 launch roboground_ros tf_tree.launch.py >"$LOG" 2>&1 &
LAUNCH_PID=$!
cleanup() {
    if kill -0 "$LAUNCH_PID" 2>/dev/null; then
        kill -INT "$LAUNCH_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$LAUNCH_PID" 2>/dev/null || true
    fi
    # launch 会 fork 出 static_transform_publisher 子进程，一并收掉
    pkill -f static_transform_publisher 2>/dev/null || true
}
trap cleanup EXIT

# 等 /tf_static 上出现内容
echo "      等待 TF 就绪 ..."
READY=0
for _ in $(seq 1 40); do
    sleep 0.5
    if timeout 10 ros2 topic echo /tf_static --once >/dev/null 2>&1; then
        READY=1
        break
    fi
done
if [ "$READY" -ne 1 ]; then
    echo "[FAIL] /tf_static 上一直没数据，launch 日志："
    tail -30 "$LOG"
    exit 1
fi
echo "      ✓ /tf_static 有数据"
echo "      launch 日志关键行："
grep -E "发布完整 TF 树|注意：|process has finished|process started" "$LOG" \
    | head -8 | sed 's/^/        /'

# ---- 2) 数一数 /tf_static 上有几条边 -------------------------------------
echo
echo "[2/4] /tf_static 上的边"
python3 - <<'PY'
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

rclpy.init()
n = Node("rg_tf_edge_count")
seen = {}
qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                 history=HistoryPolicy.KEEP_LAST, depth=100)


def cb(msg):
    for t in msg.transforms:
        seen[(t.header.frame_id, t.child_frame_id)] = t


n.create_subscription(TFMessage, "/tf_static", cb, qos)
import time
t0 = time.time()
while time.time() - t0 < 5.0 and len(seen) < 4:
    rclpy.spin_once(n, timeout_sec=0.2)

for (p, c), t in sorted(seen.items()):
    tr, ro = t.transform.translation, t.transform.rotation
    print(f"  {p:>12} → {c:<30} "
          f"t=({tr.x:+.3f},{tr.y:+.3f},{tr.z:+.3f}) "
          f"q=({ro.x:+.3f},{ro.y:+.3f},{ro.z:+.3f},{ro.w:+.3f})")
print(f"  共 {len(seen)} 条边")
n.destroy_node()
rclpy.shutdown()
PY

# ---- 3) 逐位对比解析解 + 端到端建图 --------------------------------------
echo
echo "[3/4] 链路与解析解对比 + pose_source:=tf 端到端建图"
python3 /mnt/g/RoboGround/scripts/wsl/61_e2e_tf_tree.py
RC=$?

# ---- 4) 收尾 -------------------------------------------------------------
echo
echo "[4/4] 关闭 launch"
cleanup
trap - EXIT

echo
echo "================================================================"
if [ "$RC" -eq 0 ]; then
    echo "[PASS] 完整 TF 树验证通过"
else
    echo "[FAIL] 验证未通过（见上面 ✗ 项）"
fi
echo "================================================================"
exit "$RC"
