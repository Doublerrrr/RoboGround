# AGENTS.md — RoboGround 项目交接说明

> 给新工作区 / 新对话的完整上手文档。接手本项目前，先通读本文件。
> 项目：面向服务机器人的开放词汇 3D 场景理解与语言接地系统
> 位置：`G:\RoboGround`

---

## 一、项目是什么（一句话）

给定「RGB-D 视频流 + 一句自然语言」，输出目标物体在 3D 世界系中的**米制坐标**与空间关系。
这是服务机器人"看 → 懂 → 定位 → 规划"链路里的**感知层**任务，也是 VLM 在具身场景的落地形态。

**五个 Stage**（每个对应一类岗位能力）：

| Stage | 内容 | 状态 |
|---|---|---|
| S1 | 开放词汇 2D 感知（检测 / 分割 / 特征编码，可插拔） | ✅ 4 个后端 + 3 个离线降级，全部实现 |
| S2 | 3D 语义地图（实例关联 + 体素特征场 + 语言查询） | ✅ **核心，已完成并在真实数据上验证** |
| S3 | 空间推理（几何规则引擎 + VLM 融合） | ✅ 规则引擎完整；VLM 路径就绪待权重 |
| S4 | 部署（量化 / ONNX / 异步快慢分级 / ROS2） | ✅ 代码与测试完整；ROS2 未上真机 |
| S5 | 数据闭环（自动化标注 + 质量闸门） | ✅ 完整，含跨帧一致性检查 |

---

## 二、环境与运行

- **环境**：conda 环境 `lxr`（`G:\minigore\envs\lxr`），Python 3.11.14 + torch 2.5.1+cu124
- **GPU**：RTX 4060 8GB
- **⚠️ 不要用 base 环境**（Python 3.14，torch 装不上）
- **⚠️ 不要 `pip install torch`**（会覆盖 cu124 构建，GPU 失效）

```bash
conda activate lxr
cd G:\RoboGround
pip install -e ".[dev]"

python scripts/00_quickstart.py        # 全离线端到端，10 秒
pytest                                  # 564 个测试，约 12 秒
python scripts/01_check_env.py         # 环境自检
```

**数据路径**（`configs/*.yaml` 与代码里的默认值）：
- SUN RGB-D 原始：`G:\sunrgbd_raw\`（含 `SUNRGBD/` 与 `SUNRGBDtoolbox/`）
- 场景索引缓存：`G:\RoboGround\data\cache\sunrgbd_index.npz`（已建好，10335 场景 / 64783 GT 框）
- Embodied3D 的预处理 npz：`G:\Embodied3D\data\processed\`（本项目**不依赖**它，仅历史相关）
- 模型权重：`G:\RoboGround\weights\`（懒加载时自动下载）

---

## 三、目录结构

```
src/roboground/
├── config.py          配置：YAML → 嵌套 Config，支持点号访问 / 深度合并 / CLI / 环境变量覆盖
├── types.py           ★ 全项目数据契约：CameraPose / CameraIntrinsics / RGBDFrame /
│                        Detection2D / Observation / SemanticObject / SpatialRelation
├── cli.py             命令行入口
├── geometry/          ★★ 地基
│   ├── camera.py        像素网格、射线方向、刚体变换
│   ├── projection.py    深度→点、投影、反投影检测、幽灵点过滤、内参缩放
│   └── voxel.py         体素化、特征聚合（mean/max/conf_weighted）、增量 VoxelGrid、DBSCAN
├── perception/        Stage 1
│   ├── base.py          Detector / Segmenter / Encoder 抽象 + PerceptionPipeline
│   ├── registry.py      注册表 + 按名构建 + **自动降级**
│   ├── detectors/       stub（GT投影）/ grounding_dino / yolo
│   ├── segmenters/      box / sam
│   └── encoders/        color_hist（72维离线）/ dinov2 / clip（唯一支持文本）
├── mapping/           Stage 2 ★★★ 核心
│   ├── semantic_map.py  SemanticMap / SemanticObject / QueryResult + 持久化
│   ├── builder.py       MapBuilder：感知→反投影→**实例关联**→体素融合→物体构建
│   └── query.py         匹配器（词法/嵌入）+ QueryEngine + **双语别名表**
├── reasoning/         Stage 3
│   ├── spatial_relations.py  主导关系判定（主轴竞争 + 水平 adjacent→near）
│   ├── rule_engine.py        意图解析（词表召回，不切词）+ 7 类意图处理
│   └── vlm.py                VLMBrain（懒加载）+ HybridReasoner（规则保底+VLM增强）
├── data/
│   ├── sunrgbd.py       SUN RGB-D 加载器 + 场景索引缓存（含坐标标定说明）
│   ├── synthetic.py     解析式射线-盒渲染（含完美 GT，测试金标准）
│   ├── panorama.py      多视角 → 一张 360° 等距柱状全景（真实视角融合）
│   ├── pano_scene.py    2D-3D-S 采集点 → PanoScene（含 GT 3D 框）
│   ├── auto_label.py    Stage 5：四道质量闸门 + COCO 导出 + 跨帧一致性
│   ├── video/           ★ 单视频管线（镜头检测 / 抽帧 / 去重 / 质量）
│   │   ├── shot.py        镜头检测（HSV 直方图 + robust 阈值 + 峰值突出度判据）
│   │   ├── sampling.py    四种抽帧策略（uniform/thirds/keyframe/adaptive）
│   │   ├── filter.py      两级去重（dHash → 嵌入）+ 四道质量闸门
│   │   └── pipeline.py    process_video：镜头→抽帧→去重→质量 + 吞吐统计
│   └── corpus/          ★★ 数据集级多模态语料工程（大规模数据处理）
│       ├── schema.py      数据契约：VideoRecord / CaptionRecord
│       ├── sources.py     异构数据源适配（MSR-VTT / ActivityNet / 合成）
│       ├── caption.py     字幕清洗 + 三级去重（精确/词集/MinHash-LSH）+ 对齐打分
│       ├── runner.py      规模跑批 + 逐阶段漏斗 + 吞吐 + 断点续跑
│       ├── packing.py     温度配比 + token 预算 + JSONL 分片打包
│       ├── goldset.py     真实视频上的镜头检测**金种子**核验
│       └── datacard.py    数据卡（可审查的数据决策）
├── deployment/        Stage 4
│   ├── quantization.py  量化 + dtype 自动转换 + 延迟测量 + 对照实验
│   ├── export_onnx.py   ONNX 导出 + **数值对齐校验**
│   ├── pipeline.py      异步快慢分级流水线（丢帧而非阻塞）
│   └── ros2/            bridge（纯数据层，可测）+ nodes（防御式导入）+ tf.py
├── rl/                ★ 后训练：GRPO + 规则引擎可验证奖励（RLVR）
│   ├── reward.py        三层奖励（格式门控 / 塑形正确性 / 长度惩罚）
│   ├── grpo.py          组内优势、token/sequence IS、clip、KL
│   ├── torch_loss.py    torch 版 GRPO loss
│   └── trainer.py       训练循环 + FakePolicy（离线自检）
└── eval/
    ├── metrics.py       yaw 有向盒 3D IoU（SAT+多边形裁剪）、检测指标、定位误差
    └── benchmark.py     端到端基准 + **地图级定位**（主要质量指标）

scripts/    00_quickstart ~ 24_goldset_eval（每个都能独立跑）
            00 快速上手 / 01 环境自检 / 02 建索引 / 03 建图 / 04 查询 / 05 基准
            06 自动标注 / 07 ONNX+量化
            09 真实后端集成 / 10 嵌入阈值标定 / 11 编码器×池化消融
            12 空间QA数据生成 / 13 VLM LoRA 微调 + GT区域语义基准 / 14 VLM 推理评测
            15 SAM vs bbox 的 3D 定位消融 / 16 开放词汇查询评测
            17 视频管线评测 / 18 GRPO 训练 / 19 抽帧策略下游消融
            20 大规模多模态语料工程 / 21 镜头检测诊断图 / 22 镜头检测修复前后消融
            23 金种子参数扫描 / 24 金种子评测四阶段 / 25 文档数字审计（机器核对文档里的数）
            28~33 2D-3D-S 数据集拉取 / 核验 / 解压 / 字段与 GT 检查
            34~37 ★ 多视角全景融合链路（适配器核验 / 与官方全景对照 / 数据集统计 / 定位评测）
            _shot_legacy.py 修复前的镜头检测实现（只给 22/24 做消融复现用）
            fetch_corpus_data.py 语料下载器（HF 镜像 + 分块续传）
            wsl/ 01~05 安装脚本 + 13/42/43/44 诊断脚本
                 + 04_verify_ros2.sh（三种位姿端到端）/ ros2_e2e_test.py
                 + 20_rosbag_e2e.py（rosbag2 录制回放 + QoS A/B）
                 + 30_run_all_ros2.ps1（★ 统一入口，8 个环节一把跑完）
                 + 40_build_ros2_pkg.sh（colcon build + 参数服务验证）
                 + 50_e2e_latency.py（延迟与吞吐）/ 60_verify_tf_tree.sh + 61_e2e_tf_tree.py
            + 2 个 SUN RGB-D 坐标标定工具（calibrate_/diagnose_）
configs/    default / perception_openvocab（推荐 siglip）/ vlm_reasoning / sunrgbd
ros2_ws/    ★ 标准 ament 包（colcon build → ros2 run / ros2 launch）
              src/roboground_ros/{package.xml,setup.py,setup.cfg,resource/}
              src/roboground_ros/roboground_ros/{perception_node,query_node,launch_args,tf_spec}.py
              src/roboground_ros/{launch/*.py ×3, config/roboground.yaml}
              README.md（安装 / 运行 / 参数 / TF 树 / 验证 / 诚实清单）
tests/      564 个测试（540 默认 + 24 slow），全部通过
docs/       技术方案 / 运行手册 / 实现笔记 / 简历要点 / 面试QA
            真实后端集成报告 / WSL_ROS2_安装指南
            视频数据管线报告 / GRPO_RLVR报告 / 面试答题稿_VLM与RL
            大规模多模态数据工程报告 / 数据卡_MSRVTT
            项目导航_模块与文档索引（想查东西先看这份） / 学习路线_从零到面试
            项目完整清单与阅读顺序（★ 第一次看项目从这里开始）
            多视角数据核查报告（★ 一次自我纠错：多视角是重渲染，−52% 已作废；历史记录）
            多视角全景融合报告（★ 真实多视角 → 360° 全景，替换掉虚拟视角那条链路）
            2D3D-S数据集核验报告（2D-3D-S 的字段、位姿与 GT 约定核验）
```

---

## 二.5、★ WSL 环境已装好（2026-09-11 完成，无需再装）

**结论先说：WSL2 + Ubuntu 22.04 + ROS2 Humble 已经装好并验证通过。**
用户于 2026-09-11 10:30 重启后，`RebootPending` 清除、`wsl --version` 可用，
随后一次性装完 Ubuntu-22.04 与 ROS2 Humble。**当前不需要任何未完成的重启动作。**

> 历史：在此之前该功能因待重启标记（`Component Based Servicing\RebootPending`）
> 连续 3 轮无法推进，曾把目标标记为 blocked。重启后已全部解除。

环境细节、验证命令与 4 条 WSL 踩坑见 **二.6 方式 B**。

---

> ⚠️ **WSL 已不是必需品** —— ROS2 主体链路通过 **RoboStack**（conda 的 Windows 原生构建）
> 就能跑通，见下节方式 A。WSL 的作用是补 `tf2_ros` 那一项（**现已补完**）。

---

## 二.6、ROS2 真实链路已跑通（两条路径）

### 方式 A：RoboStack（Windows 原生，**推荐**，无需管理员/WSL/重启）

```powershell
powershell -ExecutionPolicy Bypass -File "G:\RoboGround\scripts\wsl\05_run_ros2_windows.ps1"
```

conda 环境 `ros2b`（`G:\minigore\envs\ros2b`，Python 3.12 + rclpy 3.3.21）。

**实测**：6 帧处理、0 位姿失败、物体定位误差 0.053~0.534 m、**PASSED**。

⚠️ **重建环境必须整组锁定** `python=3.12 + numpy>=2 + rclpy=3.3.21`。
混用 build 变体（`_13` vs `_14`）会报
`DLL load failed ... _rclpy_pybind11`（`ERROR_PROC_NOT_FOUND`，是 ABI 不匹配不是缺 DLL）。

**`tf2_ros` 无 win-64 构建** —— TF 数学已由 `tests/test_ros2_pose.py` 的 46 个测试
（加上 `tests/test_ros2_params.py` / `test_ros2_package.py` 的离线断言）覆盖，
只有 rclpy 集成需要 WSL。

### 方式 B：WSL2 + Ubuntu 22.04 + ROS2 Humble（★ **2026-09-11 已完成**）

**环境已装好并验证通过，不需要再装**。现状：

| 项 | 值 |
|---|---|
| 发行版 | Ubuntu 22.04.5 LTS (jammy)，WSL 版本 2，内核 6.18.33.2 |
| WSL 本体 | 2.7.13.0（`wsl --version` 可查） |
| ROS2 | **Humble**（`ros-humble-ros-base` + tf2 全套），`/opt/ros/humble` |
| 默认用户 | **root**（`wsl -d Ubuntu-22.04 -u root` 直接用） |
| apt 源 | Ubuntu→aliyun 镜像（`/etc/apt/sources.list.bak-*` 是备份）；ROS2 保持官方源 |
| GPU | **已透传**，`nvidia-smi` 在 WSL 里能看到 RTX 4060 |
| 资源 | 12 核 / 15.8 GB 内存 / 955 GB 可用 |

```powershell
# ★ 统一入口：8 个环节一把跑完（三种位姿 + rosbag + QoS + 构建 + 参数 + TF 树 + 延迟）
powershell -File "G:\RoboGround\scripts\wsl\30_run_all_ros2.ps1"

# 验证 tf2_ros 真实链路（走 TF 树，不是里程计）
wsl -d Ubuntu-22.04 -u root -- bash /mnt/g/RoboGround/scripts/wsl/04_verify_ros2.sh --pose tf

# 构建 ament 包 + 验证参数服务与在线调参（19/19 项）
wsl -d Ubuntu-22.04 -u root -- bash /mnt/g/RoboGround/scripts/wsl/40_build_ros2_pkg.sh

# 完整 TF 树端到端（真起 tf_tree.launch.py）
wsl -d Ubuntu-22.04 -u root -- bash /mnt/g/RoboGround/scripts/wsl/60_verify_tf_tree.sh

# 延迟与吞吐
wsl -d Ubuntu-22.04 -u root -- bash -lc "source /opt/ros/humble/setup.bash && python3 /mnt/g/RoboGround/scripts/wsl/50_e2e_latency.py"

# 只看 tf2 包是否齐备
wsl -d Ubuntu-22.04 -u root -- bash /mnt/g/RoboGround/scripts/wsl/12_check_tf2.sh

# TF 约定隔离诊断（A/B/C/D 四组对照）
wsl -d Ubuntu-22.04 -u root -- bash -lc "source /opt/ros/humble/setup.bash && python3 /mnt/g/RoboGround/scripts/wsl/13_diag_tf_pose.py"
```

**实测结果**：6 帧处理、0 位姿失败、table 0.534m / chair 0.239m / cup 0.053m，
**3/3 次稳定 PASS**，且与方式 A（里程计）**逐位一致**（461 体素 / 3 物体）。

**2026-09-12 新增**（打成标准 ament 包之后）：

| 环节 | 结果 |
|---|---|
| `colcon build` | ✅ 通过；`ros2 pkg list` / `executables` 正常 |
| 参数服务 | ✅ 19/19：`param get` 显示真实值、热参数 `set` 立刻生效、冷参数**拒绝并说明原因** |
| 完整 TF 树 | ✅ 五帧全可达；实测 vs 解析解误差 **0.000e+00**；走 TF 建图 6/6 帧、物体误差与 static **逐位一致** |
| 延迟 | 320×240 **42~47 ms/帧**（上限 **21~24 Hz**；10 Hz 热机 0% 丢帧、**冷启动 10%**）；640×480 **133~148 ms/帧**（上限 **6.8~7.5 Hz**） |
| 目标数伸缩 | 320×240：N=3→25.6 Hz，N=6→16.1，N=9→12.9，N=12→9.5，N=18→5.7 |
| 单帧耗时构成 | 体素融合 64%，`rgb_to_hsv` 22% |

#### WSL 环境的几条坑（重装时会遇到）

1. **`set -u` 会让 `source /opt/ros/humble/setup.bash` 直接失败** ——
   setup.bash 引用未定义的 `AMENT_TRACE_SETUP_FILES`。
   所有 WSL 脚本必须 `set +u`（`set -o pipefail` 可以留）。
   （已由 `tests/test_script_hygiene.py` 自动守这条。）
2. **宿主代理在 NAT 模式下不可达** —— 宿主配了 `127.0.0.1:7897`，
   WSL 会警告"未镜像到 WSL"。**不影响**：实测直连各源都通
   （aliyun 0.18s / packages.ros.org 0.53s / raw.githubusercontent 0.48s）。
   想用代理得在 `.wslconfig` 里开 `networkingMode=mirrored`。
3. **`.sh` 不能有 BOM、换行必须是 LF；`.ps1` 反过来必须有 BOM** ——
   `.sh` 有 BOM 会报 `$'\r': command not found` 或找不到解释器；
   `.ps1` 没 BOM 会被 PowerShell 5.1 按 GBK 解码 → 中文乱码甚至语法错误。
   两条都已由 `tests/test_script_hygiene.py` 守住
   （**注意：用编辑工具改完 `.ps1` 后 BOM 会被去掉，必须重新加回**）。
4. **`RGBDFrame` 的深度字段是 `depth_m` 不是 `depth`** ——
   写错会让 AttributeError 冒泡到 rclpy 的 spin 线程，
   表现为"0 帧处理"，把真正的问题掩盖掉（踩过）。
5. **Ubuntu 22.04 自带 setuptools 59.6 < PEP 660 要求的 64** ——
   `pip install -e` 会报 `build backend is missing the 'build_editable' hook`。
   **不要**加 `--no-build-isolation`（那只会用旧 setuptools 再撞一次）；
   让 pip 用隔离环境临时拉新版即可。
6. **`message_filters.registerCallback` 是「追加」不是「替换」** ——
   内部 `self.callbacks[len(...)] = (cb, args)`。
   想给同步回调打点而重新注册一个 wrapper，会让**每帧被处理两遍**
   （实测 12 次同步 → 24 帧），且数字看起来"挺合理"。
   正确做法：包实例上的公开方法（如 `node.process_frame = wrapper`）。
7. **`--params-file` 的顶层键就是节点名** —— 必须与运行时节点名完全一致；
   `/**` 通配键可用，但**按名字后缀匹配不行**。
   所以"用另一个名字的 bootstrap 节点读参数"是错的（会读到默认值）。

### 真实 rclpy 抓到的 bug（离线测试覆盖不到）

方式 A（RoboStack）抓到 2 个：

1. **`CameraInfo` 字段名大小写**：ROS2 的 Python 类是小写 `msg.k`，C++ 才是 `K`。
   按 C++ 写会让离线假对象测试全过、真机报 "缺少 K 矩阵"。已改为两种都收。
2. **`message_filters.Subscriber` 参数名**是 `qos_profile=` 不是 `qos=`。
   已修正 `nodes.py` 里 3 处。

方式 B（WSL 真 tf2）又抓到 2 个 —— 都只在**真 tf2_ros** 上才会暴露：

3. **★ `stamp` 类型不匹配（根因级）** —— 项目内部 `PoseProvider.get_pose(stamp)`
   的 stamp 是 **float 秒**（`nodes.py`: `stamp = _stamp_to_seconds(color_msg.header.stamp)`，
   `TrajectoryPoseProvider` 也是 `float(stamp)`），
   但 `tf2_ros.Buffer.lookup_transform()` 只接受 `rclpy.time.Time`。
   传 float 会抛异常，而 `build_pose_provider` 里配了
   `fallback=IdentityPoseProvider(warn=False)` —— **异常被静默吞掉、退回恒等位姿**。
   现象极具迷惑性：**处理了 6 帧却 0 体素，而 `pose_failures` 还是 0**。
   修：新增 `_stamp_to_ros_time()`；并把 identity 兜底**默认关闭**
   （需显式配 `deploy.ros2.pose.fallback_identity: true`），
   因为它违反了 `nodes.py` 自己写明的"宁可丢帧，也不要用恒等位姿凑数"。
4. **`optical_frame_correction` 被重复套用** —— `ros2_e2e_test.py` 查的是
   `camera_color_optical_frame`，却又设了 `correction=True`（注释和代码自相矛盾），
   轴纠正被应用两次。量化：查 optical frame 时 `False` → 误差 **0.000000 m**；
   `True` → **1.697 m**。

> 这两个 bug 潜伏很久的原因是：**RoboStack 没有 `tf2_ros`，测试自动退回 odometry，
> 把整条 TF 路径绕过去了**。装上 WSL 才现形 —— 这也是"只在一条路径上测"的风险。
>
> 已补**离线回归锁**（假 buffer + 假 rclpy 模块），不装 WSL 也能守住：
> `test_tf_provider_converts_float_stamp`、
> `test_build_pose_provider_tf_has_no_silent_identity_fallback`、
> `test_tf_provider_without_fallback_returns_none_on_failure`。

---

## 二.7、★ 实测结论汇总（可直接引用）

### VLM 空间问答微调（Qwen2-VL-2B + LoRA + 4bit）

| 指标 | 未微调基座 | 微调后 |
|---|---|---|
| 回答格式可解析率 | 8.3% | **100.0%** |
| 含米制单位率 | 4.2% | **100.0%** |
| 相对关系准确率（纯视觉） | 0%（答不出） | **100.0%** |
| 米制距离 MAE | — | **0.420 m**（41.7% ≤0.3m） |
| 3D 坐标误差 | 2.756 m | **1.346 m** |

**核心结论**：微调最大的价值是**让它按格式说话**，而不是让它学会度量。
3D 坐标 1.35m 的误差说明**单张 RGB 无法恢复绝对尺度** ——
这正是"语义归 VLM、度量归几何"的实证依据。

### ★ 三方对比：含规则引擎上界基线（2026-09-11 新增）

`python scripts/14_eval_vlm_spatial.py --compare --rule-baseline`
（`--rule-baseline-only` 可几秒钟只跑上界，不加载模型）

| 任务 | 指标 | base | finetuned | **rule·oracle** |
|---|---|---|---|---|
| 米制距离 | MAE | 不可解析 | 0.420 m | **0.000 m** |
| 米制距离 | ≤0.3m | 0% | 41.7% | **100%** |
| 物体定位 | 坐标误差 | 2.756 m | 1.346 m | **0.000 m** |
| 相对关系 | 准确率 | 不可解析 | **100%** | **100%** |
| 全局 | 可解析率/单位率 | 8.3%/4.2% | **100%/100%** | **100%/100%** |

→ **微调后 VLM 在语义与格式维度完全追平几何上界，度量维度差 1~2 个数量级。**
这个"追平一项、只差一项"的模式，恰恰证明**不是训练不足** ——
如果只是没训好，不可能语义追平、却只在度量上差两个数量级。
这就是"语义归 VLM、度量归几何"的**定量证据**（不是经验之谈）。

⚠️ **`rule·oracle` 是 oracle 上界，不是独立系统** —— GT 答案本来就是这套规则引擎
在 3D 地图上生成的，所以满分是**预期结果**。它的作用是当**天花板**做归因。

⚠️ **实现时的坑**：第一版按**问题文本**匹配答案 → 一个本该满分的 oracle
跑出"距离 MAE 0.178 m、坐标误差 1.001 m"。根因是同一句话在多个场景里重复出现
但答案不同（**键冲突**）。必须用 **(图像相对路径, 问题文本)** 做键。
**"上界"没拿到满分时，先怀疑自己的评测接线，而不是急着解释结果。**

### SAM vs bbox 的 3D 定位（120 个 GT 目标）

| 掩码 | 中心误差中位 | 3D IoU | 体积比(估/GT) |
|---|---|---|---|
| bbox | 1.324 m | 0.013 | **6.28×** |
| **SAM** | **0.555 m** | 0.053 | **2.33×** |

bbox 把 3D 包围盒撑到 GT 的 **6.28 倍**（背景点被反投影进来），SAM 压到 2.33 倍。

### 区域语义：两个口径必须分清

- 用**检测器输出的区域**评测 → Top-1 **0.54**
- 用 **GT 框 + GT 标签**评测 → Top-1 **0.76**（Top-3 = 1.00）

→ 说明全流水线的瓶颈在**检测**，不在编码器。

### 开放词汇查询：未见表达命中 + 未见类别拒识

6 场景 / 151 条未见表达（命名 111 + 描述 40）/ 72 条未见类别：

| 查询路径 | 命名类 | 描述类 | 拒识率 | 综合分 |
|---|---|---|---|---|
| 纯词法 | **100.0%** | 35.0% | **100.0%** | 0.828 |
| 纯嵌入（自监督标定） | 41.4% | 12.5% | 86.1% | **0.291** |
| 纯嵌入（硬编码常量 0.0005） | 31.5% | 12.5% | 87.5% | 0.232 |
| **hybrid（默认）** | **100.0%** | **37.5%** | **100.0%** | **0.834** |

→ 复现：`python scripts/16_eval_openvocab_query.py --encoder siglip`

**⚠️ 本轮修掉的三个真实缺陷（改动了默认值，别改回去）**：

1. **词法阈值 `0.05 → 0.5`**（`QueryEngine.min_score_lexical` 与
   `SemanticMap.query_min_score` 两处必须一致）。
   0.05 会让字符 bigram 那层噪声（`bicycle`/`bottle` 共享 `le` → 0.145 分）
   通过，实测 **27.8% 的未见类别查询返回错误物体**。改完拒识率 100%、
   命名类命中率不降，综合分 0.603 → 0.828。回归锁：
   `tests/test_query.py::test_lexical_threshold_rejects_ngram_noise`。
2. **嵌入阈值不能跨特征管线共用**。SigLIP 的 `suggested_pair_threshold = 0.02`
   是在**检测区域特征**上标的，而查询比对的是**融合后的地图物体特征**（尺度小约 20 倍）。
3. **同一个阈值在两种模式下角色不同**：hybrid 下它是"精度过滤器"（0.02 最优），
   纯嵌入下它是唯一的"接受阈值"（约 0.0005 量级）。
   用错角色会让纯嵌入的描述类命中率从 12.5% 掉到 2.5%。
   现在 `Encoder` 同时提供 `suggested_pair_threshold`（hybrid）与
   `suggested_standalone_threshold`（纯嵌入回退值），`QueryEngine` 按自身模式自动选。
   回归锁：`tests/test_query.py::test_embedding_threshold_depends_on_mode`。
   调阈值前先跑脚本 16 看扫描曲线，不要凭直觉改。

**★ 阈值现在是自监督标定的，不是硬编码常量（2026-09-11 新增）**

纯嵌入模式的阈值由 `QueryEngine._calibrate_standalone_threshold()` **就地算出来**：
地图每个物体都带着检测器给的标签，这就是一份自监督标定集 ——
物体对**自己**的标签应得高分、对**别的**标签应得低分。
据此在候选阈值上**最大化平衡准确率** (TPR+TNR) 即得阈值。

- 逐场景自适应（实测阈值范围 7.2e-05 ~ 2.4e-03）
- 无需任何 GT 标注，换数据集/编码器**不用重扫**
- 实测**优于**离线扫出来的硬编码常量：纯嵌入综合分 **0.232 → 0.291**
  （命名类命中 31.5% → 41.4%，拒识率几乎不变）
- 硬编码值（`suggested_standalone_threshold`）**降级为回退**，
  仅在标定不可行时使用：物体 <2 个 / 只有一个类别 / 编码器异常
- ⚠️ **标定必须走查询时完全相同的那条打分路径**（同一个 `EmbeddingMatcher.score`）。
  用底层 `pair_scores` 标、却用 matcher 校准后的分数查询，两者不在同一尺度上。
- 诊断：`engine.embedding_threshold_source` ∈ {`auto`, `encoder`, `explicit`}；
  `engine.calibration_debug` 给出正负样本数与分离度
- 回归锁：`test_standalone_threshold_is_self_calibrated`、
  `test_self_calibration_separates_own_label_and_rejects_absent`、
  `test_calibration_falls_back_when_only_one_label`、
  `test_hybrid_does_not_use_standalone_calibration`

**⚠️ 词法路径不要接这套标定（已实验否定，别再试）**

很自然的下一个想法是"同一套 (标签, 标签) 打分，词法阈值也该能自标定"。
**实测会变差**（脚本 16 第五节）：

| 路径 | 标定集分离度 BA（上界 2.0） | 标定阈值 | 综合分 |
|---|---|---|---|
| 词法 | **2.000**（完全可分） | 0.90~0.95 | 0.828 → **0.742** ↓ |
| 嵌入 | 1.33~1.50（有重叠） | 7e-05~2.4e-03 | 0.232 → **0.291** ↑ |

根因：词法标定集是「标签查自己」，别名表对自身必然 1.0、对别的≈0，
**正负样本完全可分**，优化器把阈值推到正样本分布最边缘（≈0.90）；
而真实查询（「杯子」「seat」）常只落在**子串那层 0.85**，全被误杀。

**判据：自监督标定只在「标定集难度 ≈ 部署难度」时才有效。
BA 接近 2.0 是「标定集太简单」的危险信号，不是成功信号。**
回归锁：`test_lexical_threshold_is_not_auto_calibrated`。

**⚠️ 还有一条关于评测脚本本身的教训**：第一版脚本在**纯嵌入模式**下排序时
误从样本里读了词法分数，于是"纯嵌入"其实混了词法信号，虚报成 78.4% / 25.0%。
修正视图后真实值是 31.5% / 12.5%。**评测脚本必须和被测路径共用同一套视图**，
而且这种错的方向**总是"看起来更好"** —— 数字好得反常时要先怀疑评测本身。

> 这几条都属于"测试全绿但行为是错的"：测试只覆盖了**召回**，
> 没有覆盖**拒识**。新增的 open-vocab 评测就是补这个盲区。

---

## 二.8、★ 真实开放词汇后端已跑通

**结论先说**：Grounding DINO + SAM + SigLIP 三个真实模型**已在真实 SUN RGB-D 上端到端跑通**，
显存峰值 1.89GB / 8.59GB。详细实测数据与 6 个集成问题的修复过程见
**`docs/真实后端集成报告.md`**。

### 环境（已装好，可直接用）

```bash
conda activate lxr
# 已安装：transformers 4.51.3, timm, einops, safetensors, onnx, onnxruntime, sentencepiece
set HTTPS_PROXY=http://127.0.0.1:7897     # HF 直连超时，必须走本地 Clash 代理
set HF_HOME=G:\RoboGround\weights\hf
```

### 权重（已下载 2.6GB，别重复下）

| 模型 | 位置 |
|---|---|
| Grounding DINO tiny / SAM vit-base / DINOv2 small | `weights/hf/hub/models--*`（cache 结构） |
| CLIP ViT-B/32 | `weights/clip-vit-base-patch32/`（**local_dir**，因为 Windows 符号链接权限问题） |
| SigLIP base-patch16-224 | `weights/siglip-base-patch16-224/`（**local_dir**） |

> ⚠️ **Windows 符号链接坑**：`snapshot_download` 默认用 symlink，会报
> `WinError 1314 客户端没有所需的特权`。**必须用 `local_dir=...` 参数**绕过。

### 两条最重要的实测结论（会改变你的技术选型）

1. **用 SigLIP 而不是 CLIP 做区域级开放词汇。**
   实测 CLIP ViT-B/32 的区域特征**分离度为负**（self_sim 0.252 < other_sim 0.255，
   即"和别的类别反而更像"），Top-1 只有 0.43；SigLIP 是 +0.003 / 0.54 / 中位排名 1.0。
   原因：CLIP 的 softmax 对比损失学的是相对排序，而 SigLIP 的 sigmoid 损失
   给每个图文对独立概率。

2. **阈值必须按编码器自适应，不能写死。**
   SigLIP 的 `logit_bias=-12.93` 把概率压得很低（命中也只有 0.01~0.05），
   CLIP 校准后的命中是 0.9 —— **差两个数量级**。
   所以 `Encoder.suggested_pair_threshold` 是个属性，config 里 `null` 表示自适应。

### 复现命令

```bash
python scripts/09_test_real_backends.py --steps all --encoder siglip   # 端到端
python scripts/10_calibrate_embedding.py --scenes 2                    # 阈值标定
python scripts/11_ablate_clip_pooling.py --scenes 2                    # 编码器消融
```

---

## 二.9、★ 镜头检测在真实视频上失效（2026-09-11 发现并修复）

**一句话**：默认阈值用 Otsu，而 Otsu 的前提（帧差分布双峰）在真实压缩视频上
**不成立**。10 条真实视频报出 **704 个镜头**（70/视频），人工核验真值是 10 个 ——
**精确率 3.0%**。修复后在人工核验的金种子上 **F1 0.059 → 0.947**。

### 根因（四层）

1. **阈值**：真实帧差是**重尾单峰**（中位到 p99 跨两个数量级，没有第二个峰）。
   Otsu 对单峰分布照样返回"把质量对半劈"的阈值（实测 video7010 算出 0.0000）。
   而它原有的守卫 `min(w0, w1) < 0.05` **恰好被单峰分布骗过** ——
   对半劈时两类权重都 ≈0.5。**守卫选错了判据**。
2. **渐变路径**：用"连续帧差的滑动平均"，而**噪声的滑动平均本身就是缓慢起伏的平台**，
   到处超阈。实测渐变单独贡献 102~143 个假边界。
3. **工程**：这段判定在 `shot.detect_shots` 和 `pipeline._detect_from_hists`
   里**各写了一份**，修一边漏一边。
4. **首镜头**：`min_shot_len` 把"片子一开始就是短镜头"判成噪声 ——
   video7014 第 2 帧的真硬切（帧差 0.4600）因此被静默丢弃。

### 修法

```python
# 默认阈值模式改为 robust
T = max( median + 6 × 1.4826 × MAD , p_quantile )   # 分位数随样本量自适应
```
- `median + k×MAD`：稳健，不会被少数切点自己抬高；
- 取 `p_quantile` 是因为 MAD **低估重尾**；
- **分位数必须随样本量自适应**：`np.quantile(d, 0.99)` 在 n=31 时就是最大值本身，
  会把真切点一起挡掉（实测召回 0.80 → 0.33）。参数是
  `robust_n_signal=4` / `robust_max_frac=0.10` ——
  ⚠️ **这两个参数必须真的接进 `_resolve_threshold`**，第一版只加进了 dataclass，
  于是参数扫描给出"扫了 4/6/8/12/16 结果全一样"的**假结论**；
- Otsu 保留，但加第二道守卫：`T_otsu < median + 3×1.4826×MAD` 判为"切在噪声里"→ 回退 robust；
- 渐变路径补上**峰值突出度门**（倍率 2.0，比硬切那路的 5.0 低，因为平滑会削峰）；
- 判定逻辑抽成唯一的 `decide_boundaries()`，两个入口共用；
- 首镜头豁免 `min_shot_len`（`exempt_edge_shots=True`），**但只对硬切生效** ——
  渐变判据的 `cum` 在两端有 `np.convolve(mode="same")` 边缘偏差。

### 消融（同一生产路径，10 条真实视频）

| 配置 | 总镜头 | 每视频 |
|---|---|---|
| 修复前（旧 Otsu + 渐变无门） | 704 | 70.4 |
| 只修阈值 | 87 | 8.7 |
| **修复后（robust + 渐变门 + 首镜头豁免）** | **36** | **3.6** |

⚠️ **不能用 `threshold_mode="otsu"` 复现"修复前"** —— 新守卫会让它在真实视频上
全部回退到 robust，两者结果一模一样（实测 36 vs 36）。
必须把旧实现整个 monkeypatch 回去：`scripts/22_shot_ablation.py`
（旧实现共享在 `scripts/_shot_legacy.py`）。

### 金种子：四阶段指标（6 条视频 / 2,244 帧 / 10 个真值边界）

| 阶段 | P | R | F1 | TP/FP/FN |
|---|---|---|---|---|
| ① 修复前 | 0.030 | 1.000 | 0.059 | 10 / 320 / 0 |
| ② 只修阈值 | 0.167 | 0.900 | 0.281 | 9 / 45 / 1 |
| ③ 修阈值+渐变门 | **1.000** | 0.800 | 0.889 | 8 / 0 / 2 |
| ④ **现行默认（+首镜头豁免）** | **1.000** | **0.900** | **0.947** | 9 / 0 / 1 |

复现：`python scripts/24_goldset_eval.py`（产物 `runs/goldset_eval.json`）

### ★★ 真值本身也会错：只核验"预测出来的边界"查不出漏检

**第一版核验协议有缺陷**：只对算法报出来的边界渲染密集条目视 ——
这只能查**误检**，查不出**漏检**。结果是**召回率的分母由算法自己决定**，
漏掉的真切点从没进过核验清单，于是"召回率 1.000"是**自我印证的假象**。

发现路径：`scripts/21_shot_diag.py` 画出帧差曲线，把局部峰与已标真值叠着看。
据此给 video7014 补了 2 个真值（`#2` 被 `min_shot_len` 挡、`#298` 被阈值
差 **0.0003** 挡），F1 从 1.000 诚实地降到 0.947。

> **必须主动去曲线上找"高但没超阈"的峰。**
> 这条已写进 `runs/goldset_msrvtt.json` 的 `_note`，是核验协议的固定一步。

### ★ 参数取舍：为最后 1 个漏检付出的代价（**故意不修**）

`n_signal=8` 能拿回 `#298`（R → 1.000），但代价是 **2 个误检**、F1 掉到 0.909；
`n_signal=32` 是 4 个误检、F1 0.833。**所以默认停在 4** —— 这是显式取舍，不是遗漏。
扫描：`python scripts/23_goldset_sweep.py` → `runs/goldset_sweep.json`。

### 为什么能潜伏这么久

1. **合成数据的帧差分布确实是双峰** —— Otsu 在合成视频上工作得很好；
2. **原有单测只覆盖召回** —— `test_shot_detection_finds_known_boundaries`
   断言 `recall >= 0.8`，而修复前召回率就是 **1.000**，永远测不出来；
3. 两个入口各写一份判定逻辑；
4. 真值只从预测结果里挑 → **召回率自我印证**。

> **在合成数据上"能用"不等于在真实数据上"正确"。
> 而评估维度选错（只测召回、不测拒识）会让错误看起来像成功；
> 评估**真值的来源**选错，会让错误看起来像完美。**

回归锁：`tests/test_video_pipeline.py` 的
`test_otsu_guard_rejects_split_inside_noise`、
`test_robust_threshold_separates_noise_from_cuts`、
`test_robust_quantile_is_sample_size_aware`、
`test_gradual_path_requires_prominence`、
`test_robust_n_signal_and_max_frac_are_wired_into_resolve_threshold`、
`test_robust_n_signal_trades_recall_for_precision`、
`test_first_shot_is_exempt_from_min_shot_len`、
`test_edge_exemption_only_applies_to_hard_cuts`、
`test_edge_exemption_adds_at_most_one_boundary`。

完整记录：`docs/大规模多模态数据工程报告.md` 第四节。

---

## 四、★ 关键技术决策与踩过的坑（接手务必知道）

这些是**实测校准**出来的，别按"想当然"重新踩一遍。

### 1. SUN RGB-D 的坐标系（最容易错，已用数据标定）

**结论**：GT 3D 框位于"y 为深度、z 为竖直"的重力对齐世界系；
`world → camera` 是 `R = Rtiltᵀ @ Pᵀ`，其中 `P = [[1,0,0],[0,0,1],[0,-1,0]]` 是固定轴置换。

- 在 OpenCV 相机系里直接解释 GT 框 → 投影后 z 全为负，有效率 **0%**；
- 用 `Pᵀ` → 有效率 89%；
- 加上 `Rtiltᵀ` → 角点闭环误差中位数从 **1.66m 降到 0.78m**。

已封装为 `CameraPose.from_sunrgbd()`。标定脚本：
`scripts/calibrate_sunrgbd_geometry.py`、`scripts/diagnose_sunrgbd_frame.py`。

### 2. `coeffs` 是**全边长**，不是半边长

依据：某场景 "bed" 的 `coeffs = (1.89, 2.30, 1.97)`
→ 当全边长是 1.9×2.3m 的床 ✔；当半边长是 3.8×4.6m 的床 ✘。
"nightstand" `coeffs=(0.58,0.55,0.90)` → 0.58m 床头柜 ✔ 同样印证。

⚠️ `G:\Embodied3D` 里写的 `size = 2.0 * coeffs` 会把框放大一倍。

### 3. SUN RGB-D 深度尺度是 `/10000`

不是 `/1000`。实测 depth PNG 数值 1.2万~4.5万，`depth_m = raw / 10000`。
写错一个数量级，整个点云尺度全错。

### 4. opencv 读不了 SUN RGB-D 的 16 位深度 PNG

`cv2.imread` 返回 `None`（静默失败）。**必须用 PIL**：
`np.asarray(Image.open(path))`。

### 5. 别名表反向索引必须**多对多**

"桌子" 既属于 `table`（餐桌）也属于 `desk`（书桌）。
用单值 dict 会让后写入的覆盖先写入的，导致"桌子"只能命中其中一个类别。
已改为 `Dict[str, set]`，并在 `_resolve_labels` 里**保留歧义**（返回全部达标标签，
按提及顺序 + 分数 + 地图标签顺序排序），而不是硬选一个。

### 6. 空间关系里"旁边(near)"的判断只能在**水平轴**做

"杯子放在桌上"的表面间隙是 **0**，如果对竖直方向也用"间隙小 = 旁边"，
"杯子在桌子上面"会被误判成"杯子在桌子旁边"。
所以规则是：**水平方向表面间隙 ≤ 15cm → near；竖直方向接触 → 仍是 above/below**。
（`tests/test_reasoning.py::test_relation_above_not_overridden_by_near` 是回归测试。）

### 7. 物体构建不能用空间聚类

DBSCAN 会把桌上和桌边的物体粘成一个。改用**实例关联**
（标签兼容 + 中心距离/IoU）。见"关键设计决策 2"。

### 8. GPU 计时必须 `torch.cuda.synchronize()`

不 sync 就把"提交内核"当成"算完"，测出的延迟**显著偏低**。
`eval/benchmark.py` 的 `_Stopwatch` 已自动处理。

### 9. FP16 模型必须配 FP16 输入

否则报 `mat1 and mat2 must have the same dtype`。
`deployment/quantization.py` 提供 `cast_inputs_to_model()` 自动转换；
`compare_quantization()` 已内建这个逻辑。

### 10. `project_points_to_image` 的 NaN 陷阱（已修）

向量化赋值会覆盖掉初始化时的 NaN，导致**相机后方的点**得到一个"看起来合法"的
镜像像素坐标。必须显式 `uv[~front] = np.nan`。已修复并有测试覆盖。

### 11. DataLoader 必须 `num_workers=0`

本机沙箱环境下多进程 DataLoader 会挂（Embodied3D 已踩过）。
所有配置默认都是 0。

### 12. 别为了"优雅"改回 `np.unique(axis=0)`

`geometry/voxel.py` 的 `_unique_rows()` 用 `void` 视图压行，
比 `np.unique(axis=0)` 快数倍 —— 这是每帧都跑的热路径。

### 13. `cfg.get(path, default)` 的路径写错**永远不报错**

这是本项目最贵的一类坑（已抓到 4 处，见 `docs/实现笔记.md` 4.5.1）：
`deployment.ros2.topics`（真实键是 `deploy.ros2.topics`）、
`deploy.ros2.min_depth`（真实键是 `geometry.min_depth`）、
`data.auto_label_min_points` / `reasoning.vlm.enabled`（当时配置里根本没有）。

**加任何 `cfg.get("...")` / `cfg.set("...")` 之后，跑 `pytest tests/test_config_paths.py`** ——
它会断言每个点号路径在 `DEFAULT_CONFIG` 里真实存在。

### 14. 参数化必须让**干活的节点自己声明参数**

`--params-file` 按**节点名**匹配。用另一个名字的节点去读参数会读到默认值，
而真节点读到文件里的值 —— 两者静默不一致，且 `ros2 param set` 对真节点无效。
规范位置：`src/roboground/deployment/ros2/params.py`（参数表 + 冷热分类 + 回调）。

### 15. 验证脚本/测量脚本本身也会骗人

已经踩过三次：ROS2 e2e 的"找最近 GT 不检查标签"（虚报 0.209 vs 真实 0.534）、
延迟脚本"忘了注入 GT 框"（目标数被放大 5 倍）、
"`registerCallback` 当成替换用"（每帧处理两遍）。
**共同特征：都让数字看起来更好或更"合理"，且不报错。**

所以：自建测量工具必须带**自检断言**（如 `len(perceive_ms) == frames_processed`），
并且尽量用**两条独立路径互相对账**。

---

## 五、当前状态（迁移时点）

- **S1~S5 代码全部完成**，564 个测试全部通过（540 默认 + 24 slow）。
- **几何层已在真实 SUN RGB-D 上验证**：
  - 场景索引：10335 场景 / 64783 GT 框
  - GT 框中心投影有效率 **90.5%**
  - 真实数据建图：172 ms/帧，14 个物体
- **基准数字**（可复现，见 `runs/b2.md` 与 `runs/bench_sunrgbd.json`；
  旧的 `runs/benchmark.md` 已不在磁盘上）：

  | 场景 | 地图级定位中位误差 |
  |---|---|
  | 合成（多视角） | **0.218 m**（100% ≤ 0.25m） |
  | 真实单帧 | **0.381 m**（5 场景实测；旧值 0.553 无产物可复现） |
  | 6 个**渲染**视角（非真实多视角） | **0.259 m**（45.7% ≤ 0.25m） |

  > ★ 第三行出自**已删除**的虚拟视角链路（`--source virtual` / `virtual_camera.py` / 脚本 26、27，
  > 点云重渲染造视角，不增加信息 —— 见 `docs/多视角数据核查报告.md`）：
  > 代码与产物 `runs/bench_virtual.json` **都已不在仓库里**，这一行只作历史记录、**不可复现**；
  > 现在的多视角路线是 **2D-3D-S 真实视角 → 一张 360° 全景**
  > （`src/roboground/data/panorama.py`，报告见 `docs/多视角全景融合报告.md`）。

- **曾列为"未完成"、现已完成**（2026-09-11 复核）：
  - ~~ROS2 节点未在真实 ROS2 环境跑过~~ → **两条路径都跑通了**：
    RoboStack Windows（里程计位姿）与 **WSL2 + ROS2 Humble（`tf2_ros` 真实 TF 树）**，
    都是 6 帧 / 0 位姿失败 / 物体误差 0.053~0.534 m，且两者**逐位一致**（见二.6）。
    过程中真实 rclpy 抓出 **4 个**离线测试覆盖不到的 bug。
  - ~~VLM 路径未实际加载权重跑过~~ → **Qwen2-VL-2B + LoRA 已完整微调并评测**
    （格式合规率 8.3% → 100%，且补了规则引擎上界基线，见二.7）。
  - ~~真实开放词汇后端未实际跑过~~ → **Grounding DINO + SAM + SigLIP 已端到端跑通**
    （见二.8 与 `docs/真实后端集成报告.md`）。
  - ~~Stage 3 的 VLM 微调没有训练脚本产物~~ → **`scripts/12~14` 已产出
    `runs/vlm_spatial_lora_full/`（两轮 adapter，37 MB/个）**。
  - ~~ROS2 部署"只是脚本，不是能交付的包"~~ → **已打成标准 ament 包**
    （`ros2_ws/`：`colcon build` → `ros2 run` / `ros2 launch`，3 个 launch、
    参数 YAML、完整 TF 树、参数可在线调）。新增 4 项实测：构建、参数服务、
    完整 TF 树（实测 vs 解析解 0.000e+00）、延迟与吞吐。见 `ros2_ws/README.md`。
    过程中又抓出 **4 个静默缺陷** + **1 个"测量工具骗人"**（见四.13~15 与
    `docs/实现笔记.md` 第四点五节）。

- **仍未完成**（都不阻塞，也没有外部依赖）：
  1. **自标定的"难度体检"**：目前靠人看 BA 判断标定是否可信（见二.7）。
     可做成自动判据 —— BA 接近上界时拒绝使用该阈值并回退。这是否定实验的直接产品。
  2. **扩大开放词汇评测**：当前 10 个类别全在别名表内，对词法路径天然有利，
     需引入别名表未覆盖的类别才能真实衡量嵌入路径的增量。
  3. **真机联调**：WSL 里的 TF 树是静态广播的合成场景；
     真机需要对接实际 `/tf` 树、话题命名与时间同步容差调参。需要硬件。
  4. **快档性能**：320×240 是 42 ms/帧（23.6 Hz）、640×480 是 139 ms/帧（7.2 Hz）——
     640×480 达不到 10 Hz。要提速必须换掉逐帧点云/体素路径
     （体素融合占 64%），这是已知工程缺口而不是隐藏项。

---

## 六、待办（接手后的下一步）

**没有任何一项被外部条件卡住** —— 之前唯一的阻塞（WSL 需要重启）已在
2026-09-11 解除并完成（见二.5 / 二.6）。以下都可直接开工：

1. **自标定的"难度体检"**（最高性价比）：目前靠人看 BA 判断自监督标定是否可信。
   可做成自动判据 —— BA 接近上界（≈2.0）时说明标定集比部署场景简单，
   应拒绝该阈值并回退。这是词法侧否定实验的直接产品（见二.7）。
2. **扩大开放词汇评测**：加入别名表**未覆盖**的类别，才能真正分离
   "词法表覆盖"与"嵌入泛化"的贡献。当前 10 类全在别名表内，对词法路径天然有利。
3. **ROS2 真机联调**：`deployment/ros2/nodes.py` 的 `PerceptionNode` 已能接真实 TF 树
   （WSL 上验证过），真机需要对接实际 `/tf` 树、话题命名与同步容差调参。需要硬件。
4. **3D 场景图**：用物体间关系图替代当前的两两关系列表。
5. **快档提速**（有明确入口）：`geometry/voxel.py` 的 `add_observation` 占单帧 64%。
   先试降采样 / 预计算体素索引 / 向量化，再看要不要 ONNX 化。
   量化起点：320×240 = 42~47 ms、640×480 = 133~148 ms（`scripts/wsl/50_e2e_latency.py --repeats 3`）。

---

## 七、给新对话的速查

- **第一次看这个项目** → `docs/项目完整清单与阅读顺序.md`（完整清单 + 按面试权重的阅读顺序）
- 想了解**技术原理** → `docs/技术方案报告.md`
- 想了解**怎么跑 / 怎么部署** → `docs/运行手册.md` + `ros2_ws/README.md`
- 想了解**踩坑细节** → `docs/实现笔记.md`（本文件第四节的展开版）
- 想了解**面试怎么讲** → `docs/面试QA.md` + `docs/简历要点.md`
- 想补 **RL / LoRA / 注意力 / vLLM 的知识** → `docs/面试答题稿_VLM与RL.md`
- 想了解**视频数据管线**（镜头检测 / 抽帧 / 去重 / 质量） → `docs/视频数据管线报告.md`
- 想了解**数据集级语料工程**（大规模多模态数据处理 / 漏斗 / 配比 / 金种子核验）
  → `docs/大规模多模态数据工程报告.md` ★ 面试"大规模多模态数据处理"就讲这份
- 想了解 **GRPO + 可验证奖励** → `docs/GRPO_RLVR报告.md`
- 想知道**还剩什么** → 本文件第五节 / 第六节
- 想**快速验证没坏** → `pytest && python scripts/00_quickstart.py`
- 想**验证部署层没坏** → `powershell -File "scripts\wsl\30_run_all_ros2.ps1"`（三种位姿 + rosbag 一键全跑）

### 常见操作

```bash
# 全量测试
pytest
pytest -m slow                      # 真实数据测试
pytest tests/test_geometry.py -v     # 单个模块

# 建图 + 查询
python scripts/03_build_map.py --source sunrgbd --save runs/map.npz
python scripts/04_query_demo.py runs/map.npz --auto

# 出数字
python scripts/05_benchmark.py --source sunrgbd --scenes 5 --out runs/bench_sunrgbd.json

# 多视角全景融合（需要 2D-3D-S 数据：G:\2d3ds\area_1\area_1）
python scripts/36_pano_dataset_stats.py
python scripts/35_validate_pano_vs_official.py --all --n 3
python scripts/37_eval_pano_grounding.py --rooms office_6 hallway_6 office_27

# 量化对照
python scripts/07_export_onnx.py --skip-onnx
```

### 修改代码时的注意事项

1. **几何层的约定不要改**（相机 OpenCV、世界 z 上、位姿 world→camera）；
   改了要同步改 `types.py` 的 docstring 与 `test_geometry.py`。
2. **新加感知后端**：在对应目录实现类 + `@register_xxx("名字")`，
   然后在 `perception/__init__.py` 的末尾 import 触发注册（否则 `available_*` 看不到）。
3. **新加空间关系**：同时更新 `RELATION_NAMES`（含中文名）与 `test_reasoning.py`。
4. **任何阈值改动**都要跑 `pytest tests/test_reasoning.py`，
   尤其 `test_relation_above_not_overridden_by_near`。
