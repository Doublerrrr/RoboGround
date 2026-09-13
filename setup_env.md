# 环境安装说明（Windows + RTX 4060 8GB）

> 本项目与 `G:\Embodied3D` 共用同一套 conda 环境，以下是已实测确认的信息。

## 已确认的本机环境

| 项 | 值 |
|---|---|
| GPU | **RTX 4060 8GB**（驱动 616.64） |
| conda 根 | `G:\minigore` |
| **目标环境** | **`lxr`**（`G:\minigore\envs\lxr`） |
| Python | 3.11.14 |
| torch | **2.5.1+cu124**（`torch.cuda.is_available() == True`） |
| torchvision / torchaudio | 0.20.1 / 2.5.1 |
| numpy | 2.4.1（偏新，注意兼容） |
| 磁盘 | `G:\` 736GB 空闲（数据/权重都放这里） |

## ⚠️ 三条硬规矩

1. **只用 `lxr` 环境**。base 是 Python 3.14，torch 不支持 3.14，装不上。
2. **绝不要 `pip install torch`**。会装成 CPU 版覆盖 cu124 构建，GPU 直接失效。
   `requirements.txt` 里已刻意不含 torch。
3. **DataLoader 必须 `num_workers=0`**。本机沙箱环境下多进程 DataLoader 会挂，
   `Embodied3D` 已踩过这个坑。本项目的 `data/` 模块默认就是 0。

## 安装步骤

```bash
conda activate lxr
cd G:\RoboGround

# 1) 核心依赖（大部分 lxr 已装，只会补齐缺的）
pip install -e .

# 2) 开发依赖（pytest 等）
pip install -e ".[dev]"

# 3) 校验环境
python scripts/01_check_env.py
# 或
python -m roboground.cli check
```

## 可选依赖（按需装，不装也能跑）

本项目所有重模型后端都是**懒加载 + 可降级**的：没装依赖/没下权重时，
会自动退回轻量后端（几何/颜色直方图），整条流水线仍可跑通。

```bash
# Stage 1 真实感知后端（Grounding DINO / SAM / DINOv2 / CLIP）
pip install -e ".[perception]"

# Stage 3 VLM 空间推理（Qwen2-VL + LoRA）
pip install -e ".[vlm]"

# Stage 4 量化/导出
pip install -e ".[export]"

# 可视化点云（可选，numpy 2.4 下建议 --no-deps 避免降级 numpy）
pip install --no-deps open3d
```

> **open3d 与 numpy 2.4 的兼容**：本机 numpy 是 2.4.1（偏新）。
> - 方案 A：不装 open3d —— 本项目自带 `utils/viz.py`（matplotlib 3D 散点），功能足够；
> - 方案 B：`pip install --no-deps open3d`，只装本体不碰 numpy。

## ROS2（Stage 4，可选）

ROS2 **不要用 pip 装**，用官方安装包（Windows 上推荐 `ros2-humble` 或 WSL2）。

本项目的 ROS2 节点采用**防御式导入**：没装 `rclpy` 时
`import roboground.deployment.ros2` 依然成功、单元测试照过，
只是不能真正启动节点。这样保证「无 ROS 环境也能开发和测试」。

## 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.5.1 True NVIDIA GeForce RTX 4060
```

## 数据放置建议

| 数据 | 大小 | 建议路径 | 状态 |
|---|---|---|---|
| SUN RGB-D 预处理 npz | ~8.6GB | `G:\Embodied3D\data\processed\` | ✅ 已有 10335 场景，**本项目直接复用** |
| SUN RGB-D 原始 | ~17GB | `G:\sunrgbd_raw\` | ✅ 已有 |
| ScanNet / ReferIt3D | ~30-40GB | `G:\RoboGround\data\scannet\` | ⏳ 需申请 |
| 模型权重 | 视后端 | `G:\RoboGround\weights\` | 懒加载时自动下载 |

> 好消息：**Stage 1/2 的开发与自测不依赖任何下载**。
> `G:\Embodied3D\data\processed\` 里的 SUN RGB-D npz 自带
> 点云 + RGB + 相机内参 + 3D box，足够跑通「单帧 → 语义地图 → 语言查询」全链路。
