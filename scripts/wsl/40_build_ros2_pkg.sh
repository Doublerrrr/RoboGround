#!/usr/bin/env bash
# 40 · 构建并验证 roboground_ros 这个 ament 包（在 WSL 的 ROS2 Humble 里跑）。
#
# 为什么必须真的 `colcon build` 一遍
# ================================
# 离线测试（`tests/test_ros2_package.py`）只能查"文件在不在、数字对不对"，
# 查不到"setuptools 到底能不能把它装进 ament 索引"。而 colcon 失败的典型原因
# 恰恰是那些纯结构问题：`resource/` 标记文件缺失、`setup.cfg` 的 script_dir
# 写错、data_files 指向不存在的文件。所以这一步是"结构测试"的**实证补充**。
#
# 用法（在 WSL 里）:
#     bash /mnt/g/RoboGround/scripts/wsl/40_build_ros2_pkg.sh
#
# 注意：
#   · `set +u` 是必须的 —— `set -u` 下 source ROS2 的 setup.bash 会因为
#     `AMENT_TRACE_SETUP_FILES` 未定义而报错（本项目踩过）。
#   · pip install 用 `--no-deps`：否则 pip 会去解析 torch 等重依赖，
#     而 torch 由 conda 环境提供，绝不能让 pip 动它。
set -uo pipefail

ROS_DISTRO_NAME="humble"
SETUP="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
PROJ="/mnt/g/RoboGround"
WS="${PROJ}/ros2_ws"

if [ ! -f "$SETUP" ]; then
    echo "[FAIL] 找不到 $SETUP —— 先按 docs/WSL_ROS2_安装指南.md 装 ROS2"
    exit 2
fi

# shellcheck disable=SC1090
set +u
source "$SETUP"
set -u

echo "================================================================"
echo "40 · roboground_ros 构建与验证"
echo "================================================================"
echo "ROS2 发行版 : ${ROS_DISTRO_NAME}"
echo "python3     : $(python3 -V 2>&1)  ($(command -v python3))"
echo "项目路径    : ${PROJ}"
echo "工作空间    : ${WS}"

PASS=0
FAIL=0
declare -a RESULTS=()

check() {  # check "描述" 命令...
    local desc="$1"; shift
    if "$@" >/tmp/_rg_check.log 2>&1; then
        RESULTS+=("✓ ${desc}")
        PASS=$((PASS + 1))
    else
        RESULTS+=("✗ ${desc}")
        FAIL=$((FAIL + 1))
        echo "---- ${desc} 的输出 ----"
        tail -25 /tmp/_rg_check.log
        echo "------------------------"
    fi
}

# ---------------------------------------------------------------- 1) 依赖
echo
echo "[1/6] 检查运行时依赖"
python3 - <<'PY'
import importlib, sys
missing = []
for mod in ("numpy", "scipy", "yaml", "tqdm", "PIL", "matplotlib"):
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {exc}")
print("python:", sys.version.split()[0])
if missing:
    print("[WARN] 缺以下依赖（roboground 本体需要）：")
    for m in missing:
        print("   ", m)
    print("  修：pip3 install numpy scipy PyYAML tqdm Pillow matplotlib")
else:
    print("依赖齐全")
PY

# ---------------------------------------------------- 2) pip install roboground
echo
echo "[2/6] pip install roboground（--no-deps，绝不动 torch）"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
if python3 -c "import roboground, sys; print(roboground.__file__)" >/dev/null 2>&1; then
    echo "  已经装过了：$(python3 -c 'import roboground; print(roboground.__file__)')"
    RESULTS+=("✓ roboground 已可 import")
    PASS=$((PASS + 1))
else
    # ⚠️ 这里**不能**加 `--no-build-isolation`。
    #   Ubuntu 22.04 的 setuptools 是 59.6，比 PEP 660（editable 安装，
    #   setuptools>=64）还老，会报：
    #     "build backend is missing the 'build_editable' hook"
    #   加上 build isolation 后 pip 会**临时**装一个新版 setuptools 来构建，
    #   既解决了问题又不用动系统里的 setuptools（不改坏 ROS2 环境）。
    echo "  尝试 editable 安装（pip 会在隔离环境里取新版 setuptools）..."
    if python3 -m pip install --no-deps -i "$PIP_MIRROR" -e "$PROJ" \
            >/tmp/_rg_pip.log 2>&1; then
        echo "  editable 安装成功：$(python3 -c 'import roboground; print(roboground.__file__)')"
        RESULTS+=("✓ pip install -e roboground（editable）")
        PASS=$((PASS + 1))
    else
        echo "  editable 安装失败，退而做普通安装（拷贝一份，非实时同步）..."
        tail -6 /tmp/_rg_pip.log | sed 's/^/    /'
        if python3 -m pip install --no-deps -i "$PIP_MIRROR" "$PROJ" \
                >>/tmp/_rg_pip.log 2>&1; then
            echo "  普通安装成功：$(python3 -c 'import roboground; print(roboground.__file__)')"
            RESULTS+=("✓ pip install roboground（普通安装；**改代码要重装**）")
            PASS=$((PASS + 1))
        else
            echo "  安装失败，最后 25 行："
            tail -25 /tmp/_rg_pip.log
            RESULTS+=("✗ pip install roboground")
            FAIL=$((FAIL + 1))
        fi
    fi
fi

check "import roboground" python3 -c "import roboground; print(roboground.__version__)"
check "import roboground.deployment.ros2.nodes（真 rclpy 路径）" \
    python3 -c "
from roboground.deployment.ros2.nodes import PerceptionNode, QueryNode, ROS2_AVAILABLE
assert ROS2_AVAILABLE, 'rclpy 不可用'
print('ROS2_AVAILABLE =', ROS2_AVAILABLE)
"

# ------------------------------------------------------------- 3) colcon build
echo
echo "[3/6] colcon build --packages-select roboground_ros"
if [ ! -d "$WS/src/roboground_ros" ]; then
    echo "[FAIL] 找不到 $WS/src/roboground_ros"
    exit 2
fi
rm -rf "$WS/build" "$WS/install" "$WS/log"
# 不用 --symlink-install：/mnt/g 是 DrvFs，符号链接行为不可靠
if (cd "$WS" && colcon build --packages-select roboground_ros) >/tmp/_rg_colcon.log 2>&1; then
    echo "  构建成功"
    grep -E "Finished|Starting|Summary" /tmp/_rg_colcon.log | tail -4
    RESULTS+=("✓ colcon build")
    PASS=$((PASS + 1))
else
    echo "  构建失败，输出："
    tail -40 /tmp/_rg_colcon.log
    RESULTS+=("✗ colcon build")
    FAIL=$((FAIL + 1))
fi

if [ ! -f "$WS/install/setup.bash" ]; then
    echo "[FAIL] 构建产物 install/setup.bash 不存在，后续检查无法进行"
    printf '\n%s\n' "${RESULTS[@]}"
    echo "通过 ${PASS} / 失败 ${FAIL}"
    exit 1
fi

set +u
# shellcheck disable=SC1091
source "$WS/install/setup.bash"
set -u

# --------------------------------------------------------- 4) 包索引与入口
echo
echo "[4/6] ament 索引与可执行入口"
check "ros2 pkg list 里有 roboground_ros" \
    bash -c "ros2 pkg list | grep -qx roboground_ros"
check "ros2 pkg prefix 能定位 share 目录" \
    bash -c "ros2 pkg prefix roboground_ros"
check "ros2 pkg executables 列出 perception/query" \
    bash -c "ros2 pkg executables roboground_ros | grep -q 'perception' && ros2 pkg executables roboground_ros | grep -q 'query'"
check "share 下 launch 与 config 已安装" \
    bash -c "
SHARE=\$(ros2 pkg prefix roboground_ros)/share/roboground_ros
test -f \$SHARE/launch/perception.launch.py || { echo missing perception.launch.py; exit 1; }
test -f \$SHARE/launch/full.launch.py || { echo missing full.launch.py; exit 1; }
test -f \$SHARE/launch/tf_tree.launch.py || { echo missing tf_tree.launch.py; exit 1; }
test -f \$SHARE/config/roboground.yaml || { echo missing config; exit 1; }
echo \$SHARE
"

# ------------------------------------------------------------- 5) launch 自检
echo
echo "[5/6] launch 参数自检（--show-args 不需要真的起节点）"
for L in perception.launch.py full.launch.py tf_tree.launch.py; do
    check "ros2 launch --show-args ${L}" \
        bash -c "timeout 60 ros2 launch roboground_ros ${L} --show-args"
done

# ------------------------------------------- 6) 真节点的参数服务与在线调参
echo
echo "[6/6] ★ 起**真正的** perception 节点，验证参数服务与在线调参"
echo "      （历史上这里有个静默缺陷：参数声明在另一个名字的 bootstrap 节点上，"
echo "        于是 ros2 param set /perception 完全无效）"
python3 - <<'PY' >/tmp/_rg_param.log 2>&1 &
import threading
import time

import rclpy
from roboground.deployment.ros2.nodes import PerceptionNode

rclpy.init()
# 不传 cfg：让节点自己声明参数（与 ros2 run 的路径完全一致）
node = PerceptionNode(node_name="perception")
node.get_logger().info("真节点已就绪，参数已声明")
stop = threading.Event()
threading.Thread(
    target=lambda: [rclpy.spin_once(node, timeout_sec=0.2) for _ in iter(
        lambda: not stop.is_set(), False)], daemon=True).start()
t0 = time.time()
while time.time() - t0 < 25:
    time.sleep(0.2)
stop.set()
print("最终 sync_slop =", node.cfg.get("deploy.ros2.sync_slop"), flush=True)
print("最终 min_depth =", node.cfg.get("geometry.min_depth"), flush=True)
print("最终 detector  =", node.cfg.get("perception.detector"), flush=True)
node.destroy_node()
rclpy.shutdown()
PY
PROBE_PID=$!
sleep 6

check "真节点 /perception 的参数列表非空" \
    bash -c "ros2 param list /perception 2>/dev/null | grep -q sync_slop"
check "ros2 param get 能看到**真实生效值**（不是 not set）" \
    bash -c "ros2 param get /perception sync_slop 2>&1 | grep -qE '[0-9]'"
check "★ ros2 param set 在线改热参数 sync_slop 成功" \
    bash -c "ros2 param set /perception sync_slop 0.42 2>&1 | grep -qi 'successful\|成功'"
check "★ 热参数改完立刻可读回 0.42" \
    bash -c "ros2 param get /perception sync_slop 2>&1 | grep -q '0.42'"
check "★ 热参数 publish_every_n 也能在线改" \
    bash -c "ros2 param set /perception publish_every_n 5 >/dev/null 2>&1 && ros2 param get /perception publish_every_n 2>&1 | grep -q '5'"
check "★ 冷参数 detector 被**拒绝**且原因点名（不是静默忽略）" \
    bash -c "ros2 param set /perception detector yolo 2>&1 | grep -q 'detector'"
check "冷参数被拒后值保持 stub（没有被改掉）" \
    bash -c "ros2 param get /perception detector 2>&1 | grep -q 'stub'"
check "ros2 param dump /perception 能导出全部参数" \
    bash -c "ros2 param dump /perception 2>/dev/null | grep -q sync_slop"

wait $PROBE_PID 2>/dev/null || true
echo "  真节点进程内的最终配置："
grep "^最终" /tmp/_rg_param.log | sed 's/^/    /'
echo "  ★ 期望：sync_slop 被在线改成 0.42、detector 仍是 stub（冷参数被拒绝）"

# ------------------------------------------------------------------ 汇总
echo
echo "================================================================"
echo "结果"
echo "================================================================"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo
echo "通过 ${PASS} / 失败 ${FAIL}"
if [ "$FAIL" -eq 0 ]; then
    echo "[PASS] roboground_ros 构建与验证全部通过"
    exit 0
else
    echo "[FAIL] 有 ${FAIL} 项未通过"
    exit 1
fi
