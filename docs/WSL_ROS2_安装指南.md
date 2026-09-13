# WSL2 + ROS2 安装指南（RoboGround）

> **✅ 状态（2026-09-11）：两条路径都已装好并验证通过，本文档转为"复盘 + 重建参考"。**
>
> | 路径 | 状态 | 覆盖 |
> |---|---|---|
> | **A. RoboStack**（conda，Windows 原生，无需管理员/WSL/重启） | ✅ 已跑通 | rclpy 端到端（里程计位姿） |
> | **B. WSL2 + Ubuntu 22.04 + ROS2 Humble** | ✅ 已跑通 | **`tf2_ros` 真实 TF 树** |
>
> 两条路径的实测结果**逐位一致**（461 体素 / 3 物体 / table 0.534m）。
>
> - 想**最快验证**（不装 WSL）→ 看「零、RoboStack」
> - 想**重建 WSL 环境**或看**踩过的 4 个坑** → 看「一、WSL 方案」
> - 想看 **tf2 抓出的 2 个 bug** → 看「三、端到端测试」

---

## 零、最快路径：RoboStack（已验证可用，无需管理员）

### 一条命令跑测试

```powershell
powershell -ExecutionPolicy Bypass -File "G:\RoboGround\scripts\wsl\05_run_ros2_windows.ps1"
```

**实测结果**（2026-09-10）：

```
已处理帧数     : 6
位姿失败帧数   : 0
位姿来源       : odometry
地图物体数     : 3
  ✓ table    误差=0.534m
  ✓ chair    误差=0.239m
  ✓ cup      误差=0.053m
[OK]   ROS2 end-to-end test PASSED
```

### 环境已经建好了

conda 环境 `ros2b`（位于 `G:\minigore\envs\ros2b`）：

| 项 | 值 |
|---|---|
| Python | 3.12.14 |
| ROS2 | Humble（RoboStack win-64） |
| rclpy | 3.3.21 |
| 可用 | `rclpy` / `message_filters` / `sensor_msgs` / `nav_msgs` / `geometry_msgs` / `std_msgs` / `tf2_msgs` |
| **不可用** | **`tf2_ros`（RoboStack 没有 win-64 构建）** |

### 如果环境损坏了要重建（约 90 秒）

```powershell
& 'G:\minigore\Scripts\conda.exe' create -n ros2b -y `
    --override-channels `
    -c https://conda.anaconda.org/robostack-staging -c conda-forge `
    python=3.12 ros-humble-rclpy=3.3.21 ros-humble-message-filters `
    ros-humble-sensor-msgs ros-humble-std-msgs ros-humble-geometry-msgs `
    ros-humble-nav-msgs "numpy>=2" pyyaml pillow
```

> ⚠️ **必须整组锁定**（python 3.12 + numpy>=2 + rclpy 3.3.21）。
> 我第一次用 python 3.11 + numpy 1.26 建出来的环境里，
> conda 混了 `_13` 和 `_14` 两个 build 变体，导致
> `DLL load failed ... _rclpy_pybind11`（`ERROR_PROC_NOT_FOUND`）——
> 是 ABI 不匹配，不是缺 DLL。

### 真实 rclpy 帮我抓出的两个 bug

离线单测全过、但真机必挂的两个问题 —— 这正是做端到端集成测试的价值：

1. **`CameraInfo` 的字段名大小写**
   ROS2 的 **Python** 消息类用小写 `msg.k`，C++ 侧才是大写 `K`。
   我按 C++ 约定写成 `msg.K`，离线假对象测试全过，
   真实 rclpy 上直接报 "CameraInfo 缺少 K 矩阵"。已改为两种都收。

2. **`message_filters.Subscriber` 的参数名**
   是 `qos_profile=`，不是 `qos=`。传错会在构造节点时直接抛
   `unexpected keyword argument 'qos'`。已修正 3 处。

### 这个方法覆盖了什么、没覆盖什么

| 环节 | 覆盖 |
|---|---|
| rclpy 发布/订阅 | ✅ |
| `ApproximateTimeSynchronizer` 三路时间同步 | ✅ |
| Image（rgb8 / 16UC1）→ numpy | ✅ |
| CameraInfo → 内参 | ✅ |
| 里程计位姿 → `CameraPose` | ✅ |
| 帧组装 + "位姿缺失丢帧"策略 | ✅ |
| 反投影 → 建图 → 话题发布 | ✅ |
| **物体落点 vs 几何真值** | ✅ 误差 0.053~0.534 m |
| `tf2_ros` 的 TF 查询 | ❌ 见下 |

**`tf2_ros` 在 Windows 上没有构建**，所以：
- TF 路径的**数学**由 `tests/test_ros2_pose.py` 的 46 个离线测试覆盖
  （包括"光学轴纠正矩阵转置"那个真 bug）；
- TF 路径的**rclpy 集成**需要 WSL/Linux —— 见下节。

---

## 一、WSL 方案（★ **2026-09-11 已完成，以下为复盘与重建参考**）

> ✅ **状态：已装好并验证通过。**
> 用户于 2026-09-11 10:30 重启后，WSL 功能生效，随后一次性装完
> Ubuntu 22.04 + ROS2 Humble，**`tf2_ros` 真实 TF 树已跑通（3/3 稳定 PASS）**。
>
> 本节保留下来是因为**重建时这些信息仍然有用**，尤其是那 4 条坑。

### 1.1 当时的阻塞与解除过程（值得记的教训）

| 检查项 | 结果 | 说明 |
|---|---|---|
| CPU 虚拟化 | ✅ 已启用 | AMD Ryzen 5 7500F，`VirtualizationFirmwareEnabled=True`，SLAT 支持 |
| Windows 版本 | ✅ 合适 | Windows 11，Build 26200 |
| **管理员权限** | ⚠️ 曾缺失 → 已提权 | 启用 Windows 可选功能属系统级改动，沙箱放不开 |
| `RebootPending` | ⚠️ 曾为 `True` → 已重启清除 | **这是最后一道卡口** |
| WSL 发行版 | ✅ 已装 Ubuntu-22.04 | WSL 2.7.13.0 / 内核 6.18.33.2 |

**关键教训**：启用 `Microsoft-Windows-Subsystem-Linux` 与 `VirtualMachinePlatform`
之后 **`wsl --version` 仍然报"未安装子系统"** —— 因为注册表里有
`HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending`。
**这个标记不重启清不掉，也没有任何绕过办法**（试过 `wsl --install` 直接装，
报同样的错）。判断方法：看 `Win32_OperatingSystem.LastBootUpTime` 是否晚于
启用功能的时间点。

### 1.2 实际执行的步骤（重建时照抄）

```powershell
# 1) 装发行版（--no-launch 避免交互式建用户；我们用 root）
wsl --install -d Ubuntu-22.04 --no-launch

# 2) 换 Ubuntu 源到 aliyun（ROS2 源保持官方，避免镜像同步滞后）
wsl -d Ubuntu-22.04 -u root -- bash /mnt/g/RoboGround/scripts/wsl/11_setup_apt_mirror.sh

# 3) 装 ROS2 Humble 全套
wsl -d Ubuntu-22.04 -u root -- bash -lc "bash /mnt/g/RoboGround/scripts/wsl/03_provision_ubuntu.sh"

# 4) 验证
wsl -d Ubuntu-22.04 -u root -- bash /mnt/g/RoboGround/scripts/wsl/12_check_tf2.sh
wsl -d Ubuntu-22.04 -u root -- bash /mnt/g/RoboGround/scripts/wsl/04_verify_ros2.sh --pose tf
```

> 前置探测脚本：`10_probe_env.sh`（查网络连通性/工具齐备度，决定用哪个源）。

### 1.3 ★ 4 条 WSL 踩坑（重建必看）

1. **`set -u` 会让 `source /opt/ros/humble/setup.bash` 直接失败** ——
   setup.bash 引用未定义的 `AMENT_TRACE_SETUP_FILES`，报 `unbound variable` 退出。
   所有 WSL 脚本必须 `set +u`（`set -o pipefail` 可保留）。踩了两次。
2. **宿主 localhost 代理在 NAT 模式下不可达** —— WSL 每次都警告
   "检测到 localhost 代理配置，但未镜像到 WSL"。**不影响本项目**：
   实测直连各源都通（aliyun **0.18s** / packages.ros.org 0.53s /
   raw.githubusercontent 0.48s）。要用代理需在 `.wslconfig` 开 `networkingMode=mirrored`。
3. **`.sh` 不能有 BOM、换行必须 LF** —— 否则 `bash` 报 `$'\r': command not found`。
   用 `UTF8Encoding($false)` 写 + 把 CRLF 换成 LF。
4. **`RGBDFrame` 的深度字段是 `depth_m` 不是 `depth`** ——
   写错会让 `AttributeError` 冒泡到 rclpy 的 spin 线程，
   表现得像"帧全丢了"，把真正的问题掩盖掉（我加了条调试探针结果自己踩进去）。

---

## 二、另一条安装路径：`01/02` 交互式脚本（本项目未采用，保留备用）

> ℹ️ 2026-09-11 实际用的是 **1.2 节那 4 条命令**（`wsl --install -d Ubuntu-22.04 --no-launch`
> + root 直接跑脚本），**没有**走 `01_install_wsl.ps1` / `02_enter_ubuntu.ps1`。
> 原因：`--no-launch` + root 更省事，不需要 UAC、也不需要交互式建用户。
> 下面这套保留给"想要一个普通 Linux 用户（`lxr`，免密 sudo）"的场景。

### 步骤 1：以管理员身份运行安装脚本（约 3 分钟 + 下载）

**方式 A（推荐）**：右键 `G:\RoboGround\scripts\wsl\01_install_wsl.ps1`
→ 选「使用 PowerShell 运行」→ UAC 弹窗点「是」。

**方式 B**：右键开始菜单 →「终端(管理员)」，粘贴：

```powershell
powershell -ExecutionPolicy Bypass -File "G:\RoboGround\scripts\wsl\01_install_wsl.ps1"
```

脚本会：启用两个系统功能 → 更新 WSL 内核 → 设默认版本为 2 → 下载安装 Ubuntu-22.04。

### 步骤 2：如果提示需要重启，就重启

启用 Windows 可选功能后**通常需要重启一次**。脚本会明确告诉你。
重启后再运行一次同一个脚本（幂等，已完成的步骤会跳过）。

### 步骤 3：跑第二条脚本（无需管理员，约 5 分钟）

```powershell
powershell -ExecutionPolicy Bypass -File "G:\RoboGround\scripts\wsl\02_enter_ubuntu.ps1"
```

它会自动创建 Linux 用户（`lxr`，免密 sudo）、开启 systemd、
然后调用 `03_provision_ubuntu.sh` 装好 ROS2 Humble。

> **如果脚本说"root 初始化失败，请手动完成一次"**：
> 说明你的 WSL 版本强制首次交互建用户。执行 `wsl -d Ubuntu-22.04`，
> 按提示输入用户名 `lxr` 和密码，退出后再重跑步骤 3 的脚本即可。

---

## 三、跑真正的 ROS2 端到端测试（★ 已完成）

### ★★ 实测结果（2026-09-11）

```powershell
wsl -d Ubuntu-22.04 -u root -- bash /mnt/g/RoboGround/scripts/wsl/04_verify_ros2.sh --pose tf
```

| 项 | 结果 |
|---|---|
| **`tf2_ros` 真实链路** | ✅ **PASSED**，**3/3 次稳定复现** |
| 位姿来源 | **tf**（`StaticTransformBroadcaster` → `TransformListener` → `Buffer`） |
| 已处理帧数 / 位姿失败 | 6 / **0** |
| 体素 / 物体 | 461 体素 → 3 物体 |
| 物体定位误差 | table **0.534 m** / chair **0.239 m** / cup **0.053 m** |
| 与里程计路径对比 | **逐位一致** —— 两套独立位姿来源产出完全相同的地图 |

> **"两条路径逐位一致"比"跑通一条"更有说服力**：它同时验证了
> TF 约定解析、轴纠正、时间戳处理三件事都对。

### ★ 这一项抓出的两个 bug（只有真 tf2 才会暴露）

RoboStack（Windows）**没有 `tf2_ros`**，测试会自动退回 odometry ——
整条 TF 路径被绕过，所以这两个 bug 潜伏了很久：

**Bug 1（根因）：`stamp` 类型不匹配 → 静默退回恒等位姿**

- 项目内部约定 `PoseProvider.get_pose(stamp)` 的 stamp 是 **float 秒**
  （`nodes.py`: `stamp = _stamp_to_seconds(color_msg.header.stamp)`）；
- 但 `tf2_ros.Buffer.lookup_transform()` 只接受 `rclpy.time.Time`；
- 传 float 会**抛异常**，而 `build_pose_provider` 配了
  `fallback=IdentityPoseProvider(warn=False)` → **异常被吞、退回恒等位姿**；
- 现象：**处理了 6 帧却 0 体素，而 `pose_failures` 还是 0**（位姿"拿到了"，只是完全不对）。
- 修：新增 `_stamp_to_ros_time()`；identity 兜底改为**默认关闭**。

**Bug 2：`optical_frame_correction` 被重复套用**

- `ros2_e2e_test.py` 查的是 `camera_color_optical_frame`，却又设 `correction=True`
  （注释写着"不需要轴纠正"，代码却设了 `True`，自相矛盾）；
- 量化（`13_diag_tf_pose.py` 的 A/B/C/D 四组对照）：

  | 用例 | 配置 | 与期望位姿误差 |
  |---|---|---|
  | A | 查 optical frame，**不**纠正 | **0.000000 m** ✅ |
  | B | 查 camera_link，**做**纠正 | **0.000000 m** ✅ |
  | C | 查 optical frame，**又**纠正 | **1.697 m** ❌ |

- **A vs B 平移差 = 0.000000 m** → 顺带证明 `R_OPTICAL_TO_LINK` 矩阵本身是对的，
  错的只是配置。

> 两个 bug 都补了**离线回归锁**（假 buffer + 假 rclpy 模块），
> 不装 WSL 也能守住：见 `tests/test_ros2_pose.py`。

---

或者先进入 WSL 再跑：

```bash
wsl -d Ubuntu-22.04
source /opt/ros/humble/setup.bash
bash /mnt/g/RoboGround/scripts/wsl/04_verify_ros2.sh --verbose
```

### 这个测试到底测了什么

项目此前只有 `bridge` 层的离线单测（消息转换、位姿数学），
**没有验证过 rclpy 真实链路**。这个测试补上了：

```
rclpy 消息 → ApproximateTimeSynchronizer 时间同步 → tf2 查询 →
位姿重建 → 帧组装（含"位姿缺失丢帧"）→ 反投影建图 → 话题发布
```

**防循环论证的设计**：
- **TF 用硬编码常量发布**（不用项目自己的辅助函数生成）：
  - `map → camera_link`：平移 `(0, 0, 1.2)`
  - `camera_link → camera_color_optical_frame`：标准 REP-103 光学旋转
    四元数 `(-0.5, 0.5, -0.5, 0.5)`
- 于是"相机在 map 的 (0,0,1.2)、朝 +x 看"完全由 TF 决定，
  **节点必须自己从 TF 树重建出这个位姿**；
- 场景用解析式渲染器生成，物体世界坐标精确已知；
- **断言**：建出的地图里物体位置必须落在容差内。
  位姿约定只要错一点（哪怕只是某个轴反了），物体就会跑到别处，测试必然失败。

---

## 四、已准备的文件

| 文件 | 作用 | 需要管理员 |
|---|---|---|
| `scripts/wsl/01_install_wsl.ps1` | 启用 WSL2 功能 + 装 Ubuntu-22.04 | ✅ |
| `scripts/wsl/02_enter_ubuntu.ps1` | 建 Linux 用户 + systemd + 触发 provisioning | ❌ |
| `scripts/wsl/03_provision_ubuntu.sh` | WSL 内装 ROS2 Humble + Python 依赖 | ❌（内部用 sudo） |
| `scripts/wsl/04_verify_ros2.sh` | 跑端到端测试 + 结果解读 | ❌ |
| `scripts/wsl/ros2_e2e_test.py` | 端到端测试本体（rclpy） | ❌ |

**全部幂等**：任何一步失败后重跑都是安全的。

---

## 五、常见问题

**Q：`wsl --version` 能用，但 `wsl -l -v` 说"未安装 Linux 子系统"？**
说明功能已启用但**发行版还没装**。直接 `wsl --install -d Ubuntu-22.04 --no-launch`。
（另一种可能：功能刚启用但没重启 —— 那时 `wsl --version` 也会失败。
判断方法：看 `LastBootUpTime` 是否晚于启用功能的时间点。）

**Q：脚本报 `/opt/ros/humble/setup.bash: line 8: AMENT_TRACE_SETUP_FILES: unbound variable`？**
脚本里开了 `set -u`。ROS 的 setup.bash 会引用未定义变量，必须 `set +u` 后再 source。

**Q：`bash` 报 `$'\r': command not found`？**
`.sh` 文件是 CRLF 换行（或带 BOM）。转成 LF、去掉 BOM 即可。

**Q：WSL 每次都警告"检测到 localhost 代理配置，但未镜像到 WSL"？**
NAT 模式下 WSL 用不了宿主的 `127.0.0.1` 代理。**本项目不受影响**（实测直连都通）。
非要用代理就在 `C:\Users\<你>\.wslconfig` 加：
```ini
[wsl2]
networkingMode=mirrored
```
然后 `wsl --shutdown` 重启。

**Q：`wsl --install` 卡住不动？**
首次要下载约 500MB 的 Ubuntu rootfs，网络慢时会等较久。
如果超过 15 分钟无进展，Ctrl+C 后改用：
```powershell
wsl --install -d Ubuntu-22.04 --no-launch
```

**Q：装完 `wsl -l -v` 里 STATE 是 `Installed` 而不是 `Running`？**
正常。`--no-launch` 装出来就是未启动状态，第一次 `wsl -d Ubuntu-22.04` 才启动。

**Q：ROS2 apt 源报 GPG 错误？**
多半是签名密钥下载不完整。删掉重来：
```bash
sudo rm /usr/share/keyrings/ros-archive-keyring.gpg /etc/apt/sources.list.d/ros2.list
```
然后重跑 `03_provision_ubuntu.sh`。

**Q：想用 GPU 跑真实感知模型？**
WSL2 的 CUDA 透传依赖 Windows 侧的 NVIDIA 驱动（不需要在 WSL 里装驱动）。
`03_provision_ubuntu.sh` 里设 `INSTALL_TORCH=1` 会装 CPU 版 torch；
要 GPU 版就装对应 CUDA 版本的 wheel，然后 `nvidia-smi` 在 WSL 里应该能看到显卡。

**Q：免密 sudo 安全吗？**
这是**开发机便利性取舍**：自动化脚本需要非交互式 sudo。
生产环境请删掉 `/etc/sudoers.d/90-lxr` 并改用普通密码 sudo。
脚本注释里已标注这一点。

**Q：/mnt/g 下的项目 I/O 很慢？**
Windows 盘通过 9p 挂载，I/O 明显慢于 WSL 原生文件系统。
对 ROS2 节点测试没影响；如果要跑大量训练，建议把项目复制到 `~/RoboGround`。

---

## 六、装完之后能解锁什么

1. **ROS2 节点真实验证** —— 项目 README 里"已知局限"的第 6 条**已经划掉**
   （三种位姿来源 + 完整 TF 树 + rosbag 回放全部跑通，见 `ros2_ws/README.md`）；
2. **`ros2 topic echo /roboground/answer`** 看结构化答案；
3. **rviz2 可视化**语义地图（`src/roboground/deployment/ros2/README.md` 里有说明）；
4. **完整 TF 树 + 参数在线调参** —— 用 `ros2_ws/` 的 ament 包：
   ```bash
   cd /mnt/g/RoboGround/ros2_ws && colcon build --packages-select roboground_ros
   source install/setup.bash
   ros2 launch roboground_ros full.launch.py static_tf:=true
   ros2 param set /perception sync_slop 0.42      # 热参数：下一帧生效
   ```
5. 后续接真机时，只需把 `pose_source` 改成 `tf` 并确认
   `source_frame` 用的是相机的 `*_optical_frame`（**不要**同时把
   `optical_frame_correction` 设成 true，那会**重复旋转**，实测误差 1.697 m）。
