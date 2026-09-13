#!/usr/bin/env bash
# =============================================================================
# 03 · 在 Ubuntu(WSL2) 里安装 ROS2 Humble + Python 环境（在 WSL 内执行）
# =============================================================================
# 用法（由 02_enter_ubuntu.ps1 自动调用，也可手动）：
#   wsl -d Ubuntu-22.04 -- bash -lc "bash /mnt/g/RoboGround/scripts/wsl/03_provision_ubuntu.sh"
#
# 目标发行版：Ubuntu 22.04 + ROS2 Humble（官方配对）
# 幂等：重复运行安全。
# =============================================================================
set -euo pipefail

ROS_DISTRO_NAME="humble"
PROJECT_WSL="${PROJECT_WSL:-/mnt/g/RoboGround}"
UBUNTU_CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"

c_ok()   { printf '\033[32m[OK]   %s\033[0m\n' "$1"; }
c_info() { printf '\033[36m[..]   %s\033[0m\n' "$1"; }
c_warn() { printf '\033[33m[WARN] %s\033[0m\n' "$1"; }
c_err()  { printf '\033[31m[FAIL] %s\033[0m\n' "$1"; }
step()   { printf '\n\033[36m=== %s ===\033[0m\n' "$1"; }

# -----------------------------------------------------------------------------
step "0/7 环境确认"
# -----------------------------------------------------------------------------
echo "  发行版   : $(. /etc/os-release && echo "$PRETTY_NAME") ($UBUNTU_CODENAME)"
echo "  用户     : $(whoami)"
echo "  项目路径 : $PROJECT_WSL"
if [ "$UBUNTU_CODENAME" != "jammy" ]; then
    c_warn "ROS2 Humble 的官方目标是 Ubuntu 22.04 (jammy)，当前是 $UBUNTU_CODENAME。"
    c_warn "继续安装可能失败；若失败请改装 $UBUNTU_CODENAME 对应的 ROS2 发行版。"
fi
if [ ! -d "$PROJECT_WSL" ]; then
    c_err "项目路径不存在：$PROJECT_WSL（/mnt/g 是否挂载？）"
    exit 1
fi
c_ok "环境检查通过"

# -----------------------------------------------------------------------------
step "1/7 系统基础包与 locale"
# -----------------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq \
    locales curl gnupg lsb-release ca-certificates software-properties-common \
    build-essential git wget >/dev/null
# UTF-8 locale：否则 ros2 / colcon 遇到非 ASCII 会报错
if ! locale -a 2>/dev/null | grep -qi "en_US.utf8"; then
    sudo locale-gen en_US en_US.UTF-8 >/dev/null 2>&1 || true
fi
c_ok "基础包与 locale 就绪"

# -----------------------------------------------------------------------------
step "2/7 添加 ROS2 apt 源"
# -----------------------------------------------------------------------------
ROS_KEYRING="/usr/share/keyrings/ros-archive-keyring.gpg"
if [ ! -f "$ROS_KEYRING" ]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /tmp/ros.key
    sudo gpg --dearmor -o "$ROS_KEYRING" /tmp/ros.key
    c_ok "已导入 ROS2 签名密钥"
else
    c_ok "ROS2 签名密钥已存在"
fi

ROSLIST="/etc/apt/sources.list.d/ros2.list"
if [ ! -f "$ROSLIST" ]; then
    echo "deb [arch=$(dpkg --print-architecture) signed-by=$ROS_KEYRING] http://packages.ros.org/ros2/ubuntu $UBUNTU_CODENAME main" \
        | sudo tee "$ROSLIST" >/dev/null
    c_ok "已添加 ROS2 apt 源"
else
    c_ok "ROS2 apt 源已存在"
fi
sudo apt-get update -qq

# -----------------------------------------------------------------------------
step "3/7 安装 ROS2 $ROS_DISTRO_NAME"
# -----------------------------------------------------------------------------
PKGS=(
    "ros-${ROS_DISTRO_NAME}-ros-base"
    "ros-${ROS_DISTRO_NAME}-tf2-ros"
    "ros-${ROS_DISTRO_NAME}-tf2-tools"
    "ros-${ROS_DISTRO_NAME}-tf-transformations"
    "ros-${ROS_DISTRO_NAME}-message-filters"
    "ros-${ROS_DISTRO_NAME}-sensor-msgs"
    "ros-${ROS_DISTRO_NAME}-nav-msgs"
    "ros-${ROS_DISTRO_NAME}-vision-msgs"
    "ros-${ROS_DISTRO_NAME}-geometry-msgs"
    "ros-${ROS_DISTRO_NAME}-std-msgs"
    "python3-colcon-common-extensions"
    "python3-rosdep"
    "python3-argcomplete"
    "python3-pip"
)
MISSING=()
for p in "${PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    c_info "安装 ${#MISSING[@]} 个包（首次约 400MB，请耐心等待）..."
    sudo apt-get install -y -qq "${MISSING[@]}" >/dev/null
fi
c_ok "ROS2 $ROS_DISTRO_NAME 安装完成"

# -----------------------------------------------------------------------------
step "4/7 rosdep 初始化"
# -----------------------------------------------------------------------------
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init >/dev/null 2>&1 || true
fi
rosdep update --rosdistro "$ROS_DISTRO_NAME" >/dev/null 2>&1 || \
    c_warn "rosdep update 失败（不影响我们的节点，仅影响后续用 rosdep 装依赖）"
c_ok "rosdep 就绪"

# -----------------------------------------------------------------------------
step "5/7 让 ROS2 自动 source"
# -----------------------------------------------------------------------------
BASHRC="$HOME/.bashrc"
SOURCE_LINE="source /opt/ros/${ROS_DISTRO_NAME}/setup.bash"
if ! grep -qF "$SOURCE_LINE" "$BASHRC" 2>/dev/null; then
    {
        echo ""
        echo "# --- RoboGround: ROS2 $ROS_DISTRO_NAME ---"
        echo "$SOURCE_LINE"
        echo "export ROS_DOMAIN_ID=42          # 避免与同网段其他 ROS 冲突"
    } >> "$BASHRC"
    c_ok "已写入 ~/.bashrc"
else
    c_ok "~/.bashrc 已包含 ROS2 source"
fi

# -----------------------------------------------------------------------------
step "6/7 Python 依赖（供 roboground 使用）"
# -----------------------------------------------------------------------------
python3 -m pip install --quiet --upgrade pip 2>/dev/null || true
# 注意：不加 --break-system-packages 的发行版会拒绝装到系统 Python；
# Ubuntu 22.04 的 python3-pip 允许用户级安装，必要时退回 --user
PIP_FLAGS=""
if ! python3 -m pip install --quiet numpy 2>/dev/null; then
    PIP_FLAGS="--user"
fi
python3 -m pip install $PIP_FLAGS --quiet \
    numpy scipy pyyaml pillow matplotlib tqdm 2>&1 | tail -2 || \
    c_warn "部分 Python 包安装失败，请检查网络"

# torch：ROS2 节点本身不需要（离线后端是纯 numpy），
# 但要在 WSL 里跑真实感知模型就需要。装了才有 CUDA 透传。
if [ "${INSTALL_TORCH:-0}" = "1" ]; then
    c_info "安装 torch（CPU 版，约 200MB）..."
    python3 -m pip install $PIP_FLAGS --quiet \
        torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2 || \
        c_warn "torch 安装失败"
fi

c_ok "Python 依赖就绪"

# -----------------------------------------------------------------------------
step "7/7 验证"
# -----------------------------------------------------------------------------
set +u
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set -u

echo "  ROS_DISTRO      = ${ROS_DISTRO:-未设置}"
echo "  ROS_VERSION     = ${ROS_VERSION:-未设置}"
echo "  python3         = $(python3 --version 2>&1)"
python3 - <<'PY'
import sys
try:
    import rclpy
    print(f"  rclpy           = OK ({rclpy.__file__.split('site-packages')[-1]})")
except Exception as e:
    print(f"  rclpy           = FAIL: {e}")
    sys.exit(1)
for mod in ("numpy", "scipy", "PIL", "yaml", "matplotlib"):
    try:
        __import__(mod)
        print(f"  {mod:<15} = OK")
    except Exception as e:
        print(f"  {mod:<15} = MISSING ({e})")
PY

# TF 工具（用来确认 tf2 可用）
if command -v ros2 >/dev/null 2>&1; then
    c_ok "ros2 CLI 可用：$(ros2 --help >/dev/null 2>&1 && echo yes || echo no)"
else
    c_err "ros2 CLI 不可用"
    exit 1
fi

# GPU 透传（有 NVIDIA 驱动才有，没有也不影响节点测试）
if command -v nvidia-smi >/dev/null 2>&1; then
    c_ok "GPU 透传可用：$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    c_warn "WSL 内没有 nvidia-smi（GPU 未透传）。ROS2 节点测试不需要 GPU，可忽略。"
fi

echo ""
c_ok "provisioning 完成"
echo ""
echo "  下一步：跑真实的 ROS2 端到端测试"
echo "    bash $PROJECT_WSL/scripts/wsl/04_verify_ros2.sh"
echo ""
