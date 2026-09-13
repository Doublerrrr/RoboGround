# roboground_ros — RoboGround 的 ROS2 包

把 RoboGround 的算法本体（`roboground`）接成一个**能交付的 ROS2 子系统**：
`colcon build` → `ros2 run` / `ros2 launch`，参数可在线调，位姿走标准 TF 树。

> 算法本体在 `roboground` 包里（`src/roboground/`），本包**只做 ROS2 侧接线**。
> 这个边界是刻意的：494 个离线测试里绝大多数不需要 ROS2，
> 而 ROS2 侧保持薄才便于"换机器人只改接线"。

---

## 一、安装与构建

```bash
# 1) 装 ROS2 Humble（WSL 见 docs/WSL_ROS2_安装指南.md）
source /opt/ros/humble/setup.bash

# 2) 装算法本体（--no-deps：不要动 conda/系统里的 torch）
python3 -m pip install --no-deps -e /mnt/g/RoboGround

# 3) 构建本包
cd /mnt/g/RoboGround/ros2_ws
colcon build --packages-select roboground_ros
source install/setup.bash
```

一条命令全做完（含验证）：`bash scripts/wsl/40_build_ros2_pkg.sh`

**已知坑**：Ubuntu 22.04 自带的 setuptools 是 59.6，比 PEP 660（editable 安装）
要求的 64 还老，直接 `pip install -e` 会报
`build backend is missing the 'build_editable' hook`。
**不要**加 `--no-build-isolation` 绕开 —— 那只会让你用旧 setuptools 再撞一次墙；
让 pip 用隔离环境临时拉一个新版 setuptools 即可（脚本里就是这么做的）。

---

## 二、运行

```bash
# 只起感知节点（快档）
ros2 run roboground_ros perception

# 起感知 + 问答（慢档）
ros2 launch roboground_ros full.launch.py

# 没有真机 / 没有 TF 树时：顺带把完整 TF 树也起起来
ros2 launch roboground_ros full.launch.py static_tf:=true

# 回放 rosbag 时必须开仿真时间，否则 TF 按墙上时钟查、必然失败
ros2 launch roboground_ros full.launch.py use_sim_time:=true

# 只发 TF 树（调试位姿链路用）
ros2 launch roboground_ros tf_tree.launch.py
```

| 可执行 | 作用 | 话题 |
|---|---|---|
| `perception` | RGB-D + 位姿 → 3D 语义地图 | 订阅 `/camera/color/image_raw`、`/camera/depth/image_raw`、`/camera/color/camera_info`；发布 `/roboground/semantic_map` |
| `query` | 自然语言问题 → 结构化答案 | 订阅 `/roboground/semantic_map`、`/roboground/query`；发布 `/roboground/answer` |

---

## 三、参数

在线查看 / 修改：

```bash
ros2 param list /perception
ros2 param get  /perception sync_slop
ros2 param dump /perception
ros2 param set  /perception sync_slop 0.42      # 热参数，立刻生效
ros2 param set  /perception detector yolo       # 冷参数，会被拒绝并说明原因
```

优先级（与 nav2 相同的约定）：

```
内置默认  <  params_file（config/roboground.yaml）  <  命令行 -p key:=value
                                                    <  运行时 ros2 param set
```

### 热参数（改完下一帧就生效）

`sync_slop` · `publish_every_n` · `min_depth` · `max_depth` · `depth_scale` · `autosync_depth`

### 冷参数（只在节点构造时读一次）

`pose_source` · `pose_target_frame` · `pose_source_frame` · `optical_frame_correction` ·
`odom_topic` · `voxel_size` · `detector` · `segmenter` · `encoder` · 六个话题名 · `config_file`

改这些要重启节点（`ros2 run … --ros-args -p 名:=值`）。
**冷参数被 `ros2 param set` 时会明确拒绝并给出原因**，而不是静默忽略 ——
"配了不生效"是本项目最贵的一类坑。

### 三个必须记住的坑

1. **`optical_frame_correction` 与 `pose_source_frame` 必须自洽**：
   查 `*_optical_frame` 就设 `false`；只有查 `camera_link` 才设 `true`。
   设错会**重复旋转**（实测误差 1.697 m，而且不报任何错）。
2. **话题名只在构造时读**：ROS2 不支持运行时换订阅话题，改了话题名必须重启。
3. **`pose_source=tf` 但没有 TF 树** → 位姿查不到 → 帧被**丢弃**（不会伪造位姿）。
   节点启动时会等 `deploy.ros2.pose.ready_timeout_s`（默认 5 s）；
   仍等不到就明确告警。没有真机时加 `static_tf:=true`。

---

## 四、TF 树

期望的标准链（REP-105）：

```
map ──► odom ──► base_link ──► camera_link ──► camera_color_optical_frame
定位     里程计      URDF 外参        相机驱动
```

本包只负责验证这条链是否正确，不负责在真机上发布它：

| 边 | 真机上由谁发布 | 是否动态 |
|---|---|---|
| `map → odom` | 定位 / SLAM（AMCL、cartographer…） | 动态 |
| `odom → base_link` | 轮式 / 视觉里程计 | 动态 |
| `base_link → camera_link` | 相机外参标定（通常写进 URDF） | 静态 |
| `camera_link → camera_color_optical_frame` | 相机驱动（realsense2_camera 等） | 静态 |

`tf_tree.launch.py` 用**静态**发布器把这四条边搭出来，
用途只有一个：**在没有机器人时验证位姿链路本身是对的**。
它验证的是方向 / 四元数 / 单位 / 坐标系约定，**不是**"移动机器人建图"。

数值与推导都在 `roboground_ros/tf_spec.py`（零依赖，可离线验算）：

```bash
ros2 run tf2_tools view_frames        # 看整棵树
ros2 run tf2_ros tf2_echo map camera_color_optical_frame
```

⚠️ tf2 里 `frame_id=parent, child_frame_id=child` 的一条边存的是
**子系在父系中的位姿**（`p_parent = R·p_child + t`）。
所以 `camera_link → optical` 要填的是 **optical → camera_link** 的旋转
（`R_OPTICAL_TO_LINK`，四元数 `(-0.5, 0.5, -0.5, 0.5)`）；
填成转置是**另一个旋转**，会静默产生 1.697 m 误差。
`tests/test_ros2_package.py` 会把这张表的数值算出来和
`pose_source:=static` 的结果对比，锁死方向。

---

## 五、QoS 策略

| 数据类型 | reliability | durability | 理由 |
|---|---|---|---|
| 传感器（彩色/深度/内参） | `best_effort` | `volatile` | 丢几帧无所谓，绝不能因重传增加延迟（REP-2003） |
| 状态（地图 / 答案） | `reliable` | `transient_local` | **后**启动的规划/导航节点也要能拿到当前地图 |

地图如果用默认的 `volatile`，比感知节点晚启动的订阅者会一直等一张
**永远不会再发的**历史地图 —— 表现为"接不上"，且不报任何错。
`scripts/wsl/20_rosbag_e2e.py` 里有 A/B 对照把这条语义从行为上证出来
（晚加入 + `TRANSIENT_LOCAL` → 收到 1 条；`VOLATILE` → 0 条）。

---

## 六、验证

| 命令 | 验证内容 |
|---|---|
| `bash scripts/wsl/40_build_ros2_pkg.sh` | colcon 构建、ament 索引、launch 自检、参数服务与在线调参 |
| `bash scripts/wsl/60_verify_tf_tree.sh` | 真起 `tf_tree.launch.py`，链路与解析解逐位对比 + 走 TF 建图 |
| `python3 scripts/wsl/50_e2e_latency.py` | 延迟与吞吐（纯计算 / ROS 链路 / 承载上限，10 Hz 档**默认重复 3 次**） |
| `python3 scripts/wsl/20_rosbag_e2e.py` | rosbag2 录制 + 回放 + QoS A/B |
| `bash scripts/wsl/04_verify_ros2.sh --pose tf` | 三种位姿来源的端到端 |
| `powershell -File scripts\wsl\30_run_all_ros2.ps1` | **上面全部**，汇总成一张表 |
| `pytest` | 494 个离线测试（不需要 ROS2，含本包的结构性断言） |

> ⚠️ **性能类指标必须在"环境变化后"重跑**。
> 这不是套话：把"10 Hz 丢帧率 0.0%"写进文档之后，**电脑重启了一次**，
> 重跑统一入口直接 FAIL（冷启动那次丢帧 10%）。
> 详见 `docs/实现笔记.md` 4.5.6。

### 实测数字（本机 WSL / Ubuntu 22.04 / Python 3.10）

> 延迟数字是**多次测量**的范围（见下文"为什么给范围"），
> 其余为确定性结果（逐位可复现）。

| 项目 | 结果 |
|---|---|
| 三种位姿来源（static / tf / odometry）的地图 | **逐位一致**：table 0.534 m、chair 0.239 m、cup 0.053 m、461 体素 → 3 物体 |
| 完整 TF 树实测 vs 解析解 | 旋转最大误差 **0.000e+00**、平移 **0.000e+00** |
| rosbag2 录制回放 | 18 条消息 / 6 帧 / 2.28 MB，全部 PASS；QoS A/B：latched 1 vs volatile 0 |
| 单帧（320×240，3 个检测/帧） | 桥接 0.1~0.2 ms + 管线 **42~47 ms** ⇒ 上限 **21~24 Hz**；同步等待 ~22 ms |
| 单帧（640×480，3 个检测/帧） | 桥接 ~1 ms + 管线 **133~148 ms** ⇒ 上限 **6.8~7.5 Hz**（达不到 10 Hz） |
| 目标数伸缩（320×240） | N=3 → 22.8~25.6 Hz，N=6 → 14.5~16.1，N=9 → 11.7~12.9，N=12 → 8.6~9.5，N=18 → 5.2~5.7 |
| **10 Hz 丢帧率** | 热机 **0%**（5/5 次）；**冷启动实测 10%**（单帧 42 → 59.7 ms） |
| 单帧耗时构成（640×480） | 体素融合 64%，颜色直方图 `rgb_to_hsv` 22% |

#### 为什么延迟给的是范围而不是单值

同一台机器、同一份代码，跨 7 次测量：

| 机器状态 | ROS 链路单帧 | 10 Hz 丢帧率 |
|---|---|---|
| **冷启动（刚重启）** | **59.7 ms** | **10.0%** |
| 热机 ×5 | 40.1 / 40.2 / 40.8 / 41.9 / 42.5 ms | 0% ×5 |

**10 Hz 处于能力边缘**：纯计算上限 21~24 Hz，只有约 **2.2 倍余量**，
冷启动的缓存未命中与竞争就足以把它压下去。
所以正确的表述是「**32×240 下限 21 Hz 左右，10 Hz 勉强够但不是稳定达标**」，
而不是「10 Hz 丢帧率 0%」。
**要稳定跑 10 Hz，需要把单帧预算压到 ~20 ms 以内**（当前 42~47 ms）。

### 还没做的（诚实清单）

- **没有真机**：所有测试要么是合成场景，要么是 rosbag 回放。
  真机要做的只有两件事：把 `ros2 bag record` 换成录真机数据；
  把 `ros2 launch` 换成由上层（导航/调度）拉起。
- **相机是静止的**：`tf_tree.launch.py` 四条边全静态，
  所以还没验证"移动机器人 + 动态 TF"下的建图一致性。
- **单帧 42~47 ms 是纯 Python 实现的代价**：要做更高频率（或上 640×480 跑到 10 Hz），
  必须换掉逐帧点云/体素路径（向量化 / 降采样 / ONNX 或 C++ 后端）。
- **多机器人 / 多相机**：命名空间与多实例还没验证。
