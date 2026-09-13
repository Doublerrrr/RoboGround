#!/usr/bin/env bash
# =============================================================================
# 04 · 在 WSL 里运行 ROS2 端到端测试（在 WSL 内执行）
# =============================================================================
# 用法：
#   wsl -d Ubuntu-22.04 -- bash -lc "bash /mnt/g/RoboGround/scripts/wsl/04_verify_ros2.sh"
# =============================================================================
# ⚠️ **不要开 `set -u`**：ROS2 的 setup.bash 会引用未定义变量
# （`AMENT_TRACE_SETUP_FILES`），开了 -u 会在 source 时直接
# "unbound variable" 退出 —— 这个坑在 12_check_tf2.sh 里也踩过一次。
set -o pipefail
set +u

PROJECT_WSL="${PROJECT_WSL:-/mnt/g/RoboGround}"
ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
SCRIPT="$PROJECT_WSL/scripts/wsl/ros2_e2e_test.py"

c_ok()   { printf '\033[32m[OK]   %s\033[0m\n' "$1"; }
c_info() { printf '\033[36m[..]   %s\033[0m\n' "$1"; }
c_warn() { printf '\033[33m[WARN] %s\033[0m\n' "$1"; }
c_err()  { printf '\033[31m[FAIL] %s\033[0m\n' "$1"; }

# ---- 1) source ROS2 ----
if [ -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
    c_ok "已 source /opt/ros/${ROS_DISTRO_NAME}/setup.bash"
else
    c_err "找不到 ROS2 ${ROS_DISTRO_NAME}。请先运行 03_provision_ubuntu.sh"
    exit 2
fi

# ---- 2) 避免与同网段其他 ROS 冲突 ----
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
c_info "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

# ---- 3) 检查脚本 ----
if [ ! -f "$SCRIPT" ]; then
    c_err "找不到测试脚本：$SCRIPT"
    exit 2
fi

# ---- 4) 检查 Python 依赖 ----
python3 - <<'PY' || exit 2
import sys
missing = []
for m in ("rclpy", "numpy"):
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    print(f"[FAIL] 缺少 Python 模块：{missing}")
    sys.exit(1)
print("[OK]   rclpy 与 numpy 可用")
PY

# ---- 5) 跑测试 ----
echo ""
c_info "开始 ROS2 端到端测试（首次运行 TF 静态广播需要 1~2 秒建立）..."
echo ""
python3 "$SCRIPT" "$@"
RC=$?

echo ""
if [ $RC -eq 0 ]; then
    c_ok "ROS2 端到端测试通过"
    echo ""
    echo "  这意味着项目里唯一未验证的部分（ROS2 真机链路）已被覆盖："
    echo "    · 静态 TF 树（map → camera_link → optical）被正确解析"
    echo "    · 三路传感器消息的时间同步生效"
    echo "    · 位姿按图像时间戳查询、位姿缺失丢帧的策略正确"
    echo "    · 建出的地图物体位置与几何真值一致"
else
    c_err "ROS2 端到端测试失败（返回码 $RC）"
    echo ""
    echo "  排查建议："
    echo "    1. 确认 TF 已发布： ros2 run tf2_ros tf2_echo map camera_color_optical_frame"
    echo "    2. 看话题是否有数据： ros2 topic hz /camera/color/image_raw"
    echo "    3. 加详细日志重跑： bash $0 --verbose --frames 8"
fi
exit $RC
