# ROS2 集成说明

> **重要**：没有 ROS2 也能开发本项目。
> 本目录的 `bridge.py` 是**纯 Python 消息转换层**，不依赖 rclpy，有单元测试覆盖。
> 只有 `nodes.py` 里的节点类真正实例化时才需要 rclpy。

---

## 一、安装 ROS2（Windows / WSL2）

**不要用 pip 安装 ROS2。** 用官方发行版：

- **Windows**：安装 `ros2-humble` 官方包，并 source 其 setup 脚本
  ```powershell
  C:\dev\ros2_humble\local_setup.ps1
  ```
- **WSL2 / Ubuntu**（推荐，坑更少）：
  ```bash
  sudo apt install ros-humble-desktop
  source /opt/ros/humble/setup.bash
  ```

装完之后验证：
```bash
python -c "import rclpy; print('rclpy OK')"
```

本项目会自动探测：`bridge.ROS2_AVAILABLE` 为 `True` 时节点可用。

---

## 二、话题与消息约定

| 话题 | 方向 | 类型 | 内容 |
|---|---|---|---|
| `/camera/color/image_raw` | 订阅 | `sensor_msgs/Image` | RGB（`rgb8` / `bgr8`） |
| `/camera/depth/image_raw` | 订阅 | `sensor_msgs/Image` | 深度（`16UC1`，毫米） |
| `/camera/color/camera_info` | 订阅 | `sensor_msgs/CameraInfo` | 内参 K |
| `/roboground/semantic_map` | 发布 | `std_msgs/String`(JSON) | 语义地图摘要 + 物体列表 |
| `/roboground/query` | 订阅 | `std_msgs/String` | 自然语言问题 |
| `/roboground/answer` | 发布 | `std_msgs/String`(JSON) | 结构化回答 |

话题名可在两处改，**改完都要重启节点**（ROS2 不支持运行时换订阅话题）：

- `configs/*.yaml` 的 `deploy.ros2.topics`（库侧配置）
- **`ros2_ws/.../config/roboground.yaml`** 的 `*_topic` 参数（ROS2 参数，推荐）

> ⚠️ `nodes.py` 早期读的是 `deployment.ros2.topics`（**拼错的键**），
> 于是"在 YAML 里配了话题名"完全不生效。已修，并由
> `tests/test_config_paths.py` 锁死路径存在性。

### 为什么用 JSON 字符串而不是自定义 .msg

**优点**：
- 不需要 `colcon build` 编译消息包，跨机器/跨 ROS 发行版不会因为 msg 版本不一致而出问题；
- 地图/答案是"结构化但 schema 会演进"的数据，JSON 更灵活。

**代价（必须知道）**：
- 失去类型安全；
- 序列化开销比二进制 msg 大；
- 无法被 `ros2 topic echo` 之外的标准化工具直接解析。

**生产环境建议**：定义正式消息，例如
```
roboground_msgs/SemanticObject.msg     # label, center(geometry_msgs/Point), size, confidence
roboground_msgs/SemanticMap.msg        # header, objects[], voxel_size
roboground_msgs/SpatialQuery.msg       # header, query_text
roboground_msgs/SpatialAnswer.msg      # header, answer, targets[], relations[]
```

---

## 三、启动

> **推荐路径**：用 `ros2_ws/` 里的 ament 包（`colcon build` →
> `ros2 run` / `ros2 launch`），参数可在命令行与运行时调。
> 详见 **`ros2_ws/README.md`**。
> 下面 3.1/3.2 的手敲方式是"没构建包时的兜底"，仍然可用但不推荐交付。

### 3.1 感知节点（快档）

```bash
ros2 launch roboground_ros perception.launch.py            # 推荐
ros2 launch roboground_ros perception.launch.py static_tf:=true   # 无真机时
```

兜底方式（不构建包，直接 Python 起）：

```bash
python -c "
from roboground import load_config
from roboground.deployment.ros2.nodes import spin_perception
spin_perception(load_config())
"
```

它做的事：订阅 RGB-D + CameraInfo → 反投影 → 累积体素 → 每 10 帧发布一次地图。

> ⚠️ 这条兜底路径**不声明 ROS2 参数**，所以 `ros2 param set` 对它无效 ——
> 它只是为了"没有 colcon 环境时也能起"。要在线调参必须走 `ros2_ws/` 的包。

### 3.2 问答节点（慢档）

```bash
ros2 launch roboground_ros full.launch.py                  # 推荐（感知+问答）
ros2 launch roboground_ros full.launch.py use_vlm:=true    # 开 VLM
```

兜底方式：

```bash
python -c "
from roboground import load_config
from roboground.deployment.ros2.nodes import spin_query
spin_query(load_config())
"
```

它做的事：订阅问题 → 规则引擎/VLM 推理 → 发布结构化答案。
内部有独立工作线程，不会阻塞 ROS 回调。

### 3.3 提问与看结果

```bash
# 提问
ros2 topic pub --once /roboground/query std_msgs/String "{data: '杯子在哪'}"

# 看答案
ros2 topic echo /roboground/answer

# 看地图
ros2 topic echo /roboground/semantic_map

# 看参数 / 在线调参（只有走 ros2_ws 的包才有）
ros2 param list /perception
ros2 param set /perception sync_slop 0.42
```

---

## 四、无 ROS 环境下做什么

`bridge.py` 的全部函数都可以直接调用和测试：

```python
from roboground.deployment.ros2.bridge import (
    frame_to_dict, dict_to_frame, map_to_dict, answer_to_dict, to_json, from_json,
)

# RGBDFrame → 可 JSON 序列化的 dict（默认不含图像数据）
payload = frame_to_dict(frame)

# 带图像的双向转换
payload = frame_to_dict(frame, include_images=True)
frame2 = dict_to_frame(payload)

# 语义地图 → dict
payload = map_to_dict(semantic_map)

# 推理结果 → dict → JSON 字符串
text = to_json(answer_to_dict(result))
```

`tests/test_deployment.py` 里有 10 个相关测试，
覆盖：帧序列化往返、无图像时返回 None（明确失败）、numpy 类型处理、
JSON 解析失败的容错、无 rclpy 时节点模块仍可 import。

---

## 五、实机验证状态（2026-09-12 更新）

> ⚠️ 本节曾长期停留在**早期状态**，列着三条**已经解决**的限制，
> 反而把项目说得比实际差。现已按实测更新（原内容见本节末尾的"历史记录"）。

### 5.1 已验证通过 ✅

| 环节 | 状态 | 证据 |
|---|---|---|
| **相机位姿** | ✅ **已实现**（`tf.py`，711 行） | `TfPoseProvider` / `OdometryPoseProvider` / `StaticPoseProvider` / `TrajectoryPoseProvider`，见 `build_pose_provider` |
| **三路时间同步** | ✅ **已实现** | `nodes.py` 用 `message_filters.ApproximateTimeSynchronizer` 按时间戳配彩色+深度+内参 |
| **rclpy 端到端（RoboStack Windows）** | ✅ **PASSED** | 6 帧 / 0 位姿失败 |
| **三种位姿来源逐位一致** | ✅ **PASSED** | static / tf / odometry 给出**同一个** `table 0.534 / chair 0.239 / cup 0.053 m`、461 体素 → 3 物体 |
| **rosbag2 录制 + 回放** | ✅ 见第六节 | 18 条消息 / 6 帧 / 2.28 MB；含 QoS A/B 对照（latched 1 vs volatile 0） |
| **标准 ament 包** | ✅ `colcon build` 通过 | `ros2_ws/`：`package.xml` + `setup.py` + 3 个 launch + 参数 YAML；`ros2 run roboground_ros perception/query` |
| **参数化 + 在线调参** | ✅ **PASSED（19/19 检查）** | 节点自己声明参数；`ros2 param get` 显示真实值；`param set` 改热参数立刻生效；冷参数**明确拒绝并说明原因** |
| **完整 TF 树** | ✅ **PASSED** | `map→odom→base_link→camera_link→camera_color_optical_frame`，实测与解析解旋转/平移误差 **0.000e+00** |
| **端到端延迟与吞吐** | ✅ **已测** | 320×240：**42~47 ms/帧**（上限 **21~24 Hz**）；640×480：**133~148 ms/帧**（上限 6.8~7.5 Hz）；10 Hz 丢帧率：热机 **0%**（5/5 次）/ **冷启动 10%** |
| **QoS 策略** | ✅ 见第五节 | 传感器 `BEST_EFFORT`+`VOLATILE`，状态（地图/答案）`RELIABLE`+`TRANSIENT_LOCAL` |

> 三条**互相独立**的位姿来源给出逐位一致的地图，这一点比"跑通一条"有说服力得多 ——
> 它同时证明了 TF 约定解析、轴纠正、时间戳处理三件事都对。

一条命令跑完全部：`powershell -File scripts\wsl\30_run_all_ros2.ps1`（8 个环节，ALL PASS）。

### 5.2 仍未做 ❌（诚实清单）

1. **真机联调** —— 没有硬件。真机上还需要对接实际 `/tf` 树、
   话题命名空间与同步容差调参。**这是唯一真正缺的一环。**
2. **相机是静止的** —— `tf_tree.launch.py` 四条边全静态，
   所以"移动机器人 + 动态 TF"下的建图一致性还没验证过。
3. **单帧 42~47 ms 是纯 Python 实现的代价** —— 640×480 下 6.8~7.5 Hz 达不到 10 Hz；
   要做高频快档必须换掉逐帧点云/体素路径（向量化 / 降采样 / ONNX 或 C++）。
4. **未做不同网络条件下的压测** —— 局域网与 Wi-Fi 丢包场景没测过；
   现有延迟数字是**同进程回环**，属于**下界**。
5. **多机器人 / 多相机** —— 命名空间与多实例还没验证。

### 5.2b 已修掉的四个静默缺陷（2026-09-12）

这四个都是"不报错、单测全绿、只在真 ROS2 里跑一遍才暴露"的类型，
细节见 `docs/实现笔记.md` 第四点五节：

| 缺陷 | 症状 | 修法 |
|---|---|---|
| 参数接到不存在的 Config 路径 | `ros2 param set` 无声失败 | 补齐默认配置键 + `tests/test_config_paths.py` 锁死路径存在性 |
| 用别的名字的节点读参数 | 参数文件与真节点取值不一致，`param set` 对真节点无效 | 让真节点自己声明参数（`params.py`） |
| 启动竞态 | 首帧位姿查不到 → 静默丢帧（实测 6 帧只处理 5 帧） | `wait_ready()` 等到真能查到再收数据（修完 6/6 帧） |
| launch 里四元数拼进节点名 | 整棵 TF 树都起不来 | 节点名只用逻辑名（纯字符串） |

### 5.3 历史记录（已过期的旧表述，保留以说明演进）

> 以下三条**已经不再成立**，列在这里是因为它们曾经被写进文档、
> 影响过判断，值得留痕：
>
> - ~~"`PerceptionNode._pose` 目前留空，没有实现"~~ →
>   实际已实现 4 种位姿提供者并端到端验证；
> - ~~"未做时间同步，当前实现是缓存最新的一对"~~ →
>   实际已用 `ApproximateTimeSynchronizer`；
> - ~~"没有在真实 ROS2 + 机器人上端到端跑过"~~ →
>   真实 ROS2 已在**两条路径**上跑通，只有**真机**没跑。
>
> **教训**：交接文档的"已知限制"会随时间变成**假的自我贬低**。
> 它和"过期的漂亮数字"一样有害 —— 只是方向相反、更容易被忽略，
> 因为没人会去质疑一句"我这个还没做"。
