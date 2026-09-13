#!/usr/bin/env bash
# =============================================================================
# 10 · WSL Ubuntu 环境探测（在 WSL 内以 root 执行）
# =============================================================================
# 目的：在正式 provisioning 之前，先把"能不能装"这件事查清楚：
#   - 基础工具（sudo/curl/gpg）是否齐
#   - 网络到各镜像源是否通、速度如何（决定用官方源还是国内镜像）
#   - 宿主代理在 NAT 模式下是否可达（已知 127.0.0.1 不可达）
# =============================================================================

echo "=== 0. 基础信息 ==="
. /etc/os-release
echo "  发行版   : $PRETTY_NAME ($VERSION_CODENAME)"
echo "  用户     : $(whoami)"
echo "  内核     : $(uname -r)"
echo "  架构     : $(dpkg --print-architecture)"
echo "  CPU      : $(nproc) 核"
echo "  内存     : $(free -m | awk '/Mem:/{print $2" MB"}')"
echo "  磁盘可用 : $(df -h / | awk 'NR==2{print $4}')"
echo "  python3  : $(python3 --version 2>&1 || echo '缺失')"

echo
echo "=== 1. 基础工具 ==="
for c in sudo curl wget gpg lsb_release apt-get; do
    if command -v "$c" >/dev/null 2>&1; then
        echo "  [OK]   $c -> $(command -v "$c")"
    else
        echo "  [MISS] $c"
    fi
done

echo
echo "=== 2. 现有 apt 源 ==="
grep -rhE '^deb ' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | sed 's/^/  /' || echo "  (无)"

echo
echo "=== 3. 网络连通性（超时 6s） ==="
probe() {
    local name="$1" url="$2"
    local code t0 t1
    t0=$(date +%s.%N)
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 "$url" 2>/dev/null || echo "ERR")
    t1=$(date +%s.%N)
    printf '  %-46s %-5s %5.2fs\n' "$name" "$code" "$(echo "$t1 - $t0" | bc)"
}
probe "archive.ubuntu.com (官方)"      "http://archive.ubuntu.com/ubuntu/dists/jammy/Release"
probe "mirrors.tuna.tsinghua.edu.cn"   "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/dists/jammy/Release"
probe "mirrors.aliyun.com"             "https://mirrors.aliyun.com/ubuntu/dists/jammy/Release"
probe "packages.ros.org (官方 ROS2)"   "http://packages.ros.org/ros2/ubuntu/dists/jammy/Release"
probe "mirrors.tuna 的 ros2 镜像"       "https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu/dists/jammy/Release"
probe "raw.githubusercontent.com"      "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key"

echo
echo "=== 4. 宿主代理可达性（已知 NAT 模式下 127.0.0.1 不通） ==="
HOST_IP=$(ip route show default | awk '{print $3}')
echo "  宿主 IP（默认网关）: ${HOST_IP:-未知}"
for p in 7897 7890 10809; do
    if timeout 3 bash -c "echo > /dev/tcp/${HOST_IP}/${p}" 2>/dev/null; then
        echo "  [OK]   ${HOST_IP}:${p} 可达"
    else
        echo "  [MISS] ${HOST_IP}:${p}"
    fi
done

echo
echo "=== 5. 判定 ==="
echo "  上面哪一行是 [OK] 就用哪个源。若只有国内镜像通，"
echo "  先跑 11_setup_apt_mirror.sh 换源再 provisioning。"
