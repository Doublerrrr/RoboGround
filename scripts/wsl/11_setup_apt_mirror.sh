#!/usr/bin/env bash
# =============================================================================
# 11 · 把 Ubuntu apt 源换成国内镜像（在 WSL 内以 root 执行）
# =============================================================================
# 为什么只换 Ubuntu 源、**不换** ROS2 源：
#   - Ubuntu 源要下 ~400MB 系统包，aliyun 实测 0.18s vs 官方 2.18s，差距明显；
#   - ROS2 的 packages.ros.org 实测 0.53s 已经够快，而第三方镜像**可能有同步滞后**，
#     会导致 ros-humble-* 版本与 keyring 不匹配，排查成本高。
#     "够快就别动" —— 镜像只在明显更快时才值得引入同步风险。
#
# 幂等：重复运行安全（已是镜像源时不会重复替换）。
# 会自动备份到 sources.list.bak-<时间戳>。
# =============================================================================
set -euo pipefail

MIRROR="${MIRROR:-https://mirrors.aliyun.com}"
SRC="/etc/apt/sources.list"

c_ok()   { printf '\033[32m[OK]   %s\033[0m\n' "$1"; }
c_info() { printf '\033[36m[..]   %s\033[0m\n' "$1"; }
c_warn() { printf '\033[33m[WARN] %s\033[0m\n' "$1"; }

echo "=== 换源为 $MIRROR ==="

if [ ! -f "$SRC" ]; then
    c_warn "$SRC 不存在，跳过"
    exit 0
fi

CUR=$(grep -cE '^deb ' "$SRC" || true)
echo "  当前源条目数: $CUR"

if grep -q "mirrors.aliyun.com" "$SRC"; then
    c_ok "已经是 aliyun 源，无需替换"
else
    BAK="${SRC}.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$SRC" "$BAK"
    c_info "已备份到 $BAK"

    # http://archive.ubuntu.com/ubuntu   -> https://mirrors.aliyun.com/ubuntu
    # http://security.ubuntu.com/ubuntu  -> https://mirrors.aliyun.com/ubuntu
    sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|${MIRROR}/ubuntu|g" \
        -e "s|http://security.ubuntu.com/ubuntu|${MIRROR}/ubuntu|g" \
        -e "s|https://archive.ubuntu.com/ubuntu|${MIRROR}/ubuntu|g" \
        -e "s|https://security.ubuntu.com/ubuntu|${MIRROR}/ubuntu|g" \
        "$SRC"
    c_ok "已替换为 $MIRROR"
fi

echo
echo "  替换后的源:"
grep -E '^deb ' "$SRC" | sed 's/^/    /'

echo
c_info "apt-get update ..."
export DEBIAN_FRONTEND=noninteractive
if apt-get update -qq 2>&1 | tail -5; then
    c_ok "apt-get update 成功"
else
    c_warn "apt-get update 有报错，请检查上面输出；可回滚：cp $SRC.bak-* $SRC"
    exit 1
fi

echo
c_ok "换源完成"
