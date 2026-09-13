# Stanford 2D-3D-S 数据集核验报告

> **目的**：在写适配器**之前**，把数据集的真实结构与字段含义**实测**清楚。
>
> 为什么必须先做这件事：本项目在 SUN RGB-D 上吃过一次亏 —— 照文档假设写代码，
> 真数据一到就崩，而且坐标系类的错误是**静默**的（地图重影、不报错）。
> 所以拿到数据的第一件事是核验，不是写代码。
>
> 核验日期：2026-09-13 ｜ 数据：`area_1_no_xyz.tar`（30.44 GB）｜
> 复现命令见本文末尾

---

## 一、获取与完整性

| 项 | 值 |
|---|---|
| 分发渠道 | **Redivis**（斯坦福 Doerr 可持续发展学院数据仓库，DOI `10.57761/gmhc-wx10`） |
| table 引用 | `sdss_data_repository.stanford_2d_3d_semantics_dataset_2d_3d_s:f304:v1_0.no_xyz:ct1f` |
| 访问模型 | 表元数据**公开可读**；**文件内容需授权**（`accessLevel: data` + token 需带 `data.data` scope） |
| 我们下载的 | `area_1_no_xyz.tar` — **30.44 GB** |
| **MD5 校验** | ✅ `21098fbe93b561e30e79197a95fa4fd2`（与官方 checksum 页一致） |
| 全量规模 | `no_xyz` 7 个 area 合计 **224.26 GB**（⚠️ 官方 README 写的 "110G" 已过时） |

> ⚠️ **原始斯坦福渠道已全部失效**：`3dsemantics.stanford.edu` / `buildingparser.stanford.edu`
> 返回 503，`cvgl.stanford.edu` 数据路径 404，原始 Google 表单 404。
> **Redivis 是当前唯一可用的分发渠道。**

---

## 二、★ 三处与官方 README **不符**的地方（实测纠正）

| # | 官方 README 说 | **实际是** | 影响 |
|---|---|---|---|
| 1 | `camera_rt_matrix` 是 **4×3** | **`(3, 4)`** = [R \| t] | 照 4×3 解析会**整个位姿错** |
| 2 | 文件命名 `camera_{uuid}__{room}_{i}_frame_{j}_domain__xxx`（**双**下划线） | `camera_{uuid}_{room}_{i}_frame_{j}_domain_{modality}.png`（**单**下划线） | 照模板匹配 → **一个文件都匹配不到** |
| 3 | 3D 点云文件 `Area_#_PointCloud.mat` | **`3d/pointcloud.mat`**（小写） | 找不到 GT 文件 |

另外：README 说 raw 深度是 HDR **JPG**，实测 raw 深度是 **PNG**（`{uuid}_d{pitch}_{yaw}.png`）。

> **教训**：官方文档的字段名/形状也会错。**核验不是形式主义** ——
> 上面第 1、2 条如果照文档写，错误都会是静默的。

---

## 三、★ 位姿约定（最关键的一项，已判定）

`data/pose/camera_*_domain_pose.json` 里同时给了相机光心位置与 RT 矩阵。
用一个**可判定的验算**就能确定方向：

```
设 M = camera_rt_matrix（实测 (3,4)）→ R = M[:, :3]，t = M[:, 3]
  · 若 p_cam = R·p_world + t （world→camera）→ 光心 C = −Rᵀ·t
  · 若 p_world = R·p_cam + t （camera→world）→ 光心 C = t
把两种 C 都算出来，跟 json 里的 camera_location 比：

实测（两个独立 json）：
  det(R) = +1.000000                      ← 合法旋转
  world→camera：C = [-17.8400, 20.2997, 1.3978]   误差 0.000001 m   ★★
  camera→world：C = [ -1.4260,  0.7255, -27.0136]  误差 38.207032 m
```

**结论：`p_cam = R · p_world + t`（world→camera）**，误差 **1.2 微米**，无可争议。

> **★ 这意味着：数据集约定与本项目 `CameraPose` 的约定完全一致。**
> 适配器可以**直接构造 `CameraPose(R, t)`，不需要任何约定转换** ——
> 而"转换写错"正是本项目踩过的那个静默坑（当时误差 1.697 m）。

pose json 的其余字段（实测）：

| 字段 | 含义 |
|---|---|
| `camera_k_matrix` | 3×3 内参，**逐帧不同**（1080×1080，cx=cy=540） |
| `field_of_view_rads` | 该帧的 FOV（实测 0.995 ~ 1.248 rad） |
| `camera_location` | 相机光心世界坐标 |
| `camera_rt_matrix` | **(3,4) = [R\|t]，world→camera** |
| `camera_uuid` / `point_uuid` | 采集点标识（32 位十六进制） |
| `frame_num` | 该采集点内的帧号 |
| `room` | 房间名（如 `office_6_1`） |

---

## 四、深度：位深、量纲、完整性

`data/depth/camera_*_domain_depth.png`（实测 3 张）：

| 项 | 实测值 |
|---|---|
| 尺寸 / 位深 | **1080×1080，uint16** |
| 缺失值占比 | **0.0%**（既没有 65535 也没有 0）→ **完全稠密** |
| 按 **1/512 m**（官方说明） | min 0.60 ~ max 3.09 m，中位 1.29 m ✅ **合理** |
| 按 1/1000 m（毫米） | min 0.31 ~ max 1.58 m ❌ **不合理**（办公室不可能这么近） |

**结论：`depth_m = raw / 512.0`，且不需要处理缺失值。**

> 与 SUN RGB-D 对比：SUN RGB-D 的深度要 `/10000` 且大量缺失；
> 2D-3D-S 的深度**全稠密**，这对建图是有利的（不必做空洞插值）。

---

## 五、GT：物体级标注（适配器的评测依据）

`3d/pointcloud.mat` 是 **MAT v7.3 / HDF5**（1.07 GB）。两个 MATLAB 存储坑：

1. **struct array 按字段存成引用数组**：`Area_1/Disjoint_Space` 是 Group，
   内含 `name`/`object`/`color`/`AlignmentAngle` 四个 **(44,1) 引用数组**；
   第 i 个空间的名字是 `h[h[...]['name'][i,0]]`。
2. **字符串是 MATLAB char 数组（uint16 码点）**，直接取 `[0]` 会得到
   **首字符的码点**（例如 'c' = 99），看起来像"名字是 99"这种莫名其妙的数字。

嵌套还有第二层：`object[i,0]` 解出来是个 **Group**（结构体数组），
它的每个字段（`name`/`Bbox`/`points`/`Voxels`…）又是 **(N,1) 引用数组**。

实测结果（area_1）：

| 项 | 值 |
|---|---|
| 空间（房间）数 | **44**（`conferenceRoom_1`、`copyRoom_1`、`hallway_1`…） |
| **物体总数** | **1,690** |
| 每空间物体数 | min 11 / 中位 35 / max 152 |
| 每物体的字段 | `name`、`Bbox`、`points`(3,N)、`RGB_color`、`global_name`、`Voxels`、`Voxel_Occupancy`、`Points_per_Voxel` |
| **`Bbox`** | **(6,) float64 = [Xmin Ymin Zmin Xmax Ymax Zmax]**，世界系 AABB |

**14 个类别**（S3DIS 13 类语义体系）：

```
clutter 775   wall 234    chair 155    bookcase 90   door 86
table   69    beam  61    column  57   ceiling 55    floor 44
window  29    board 27    sofa     6   stairs   2
```

> ✅ `Bbox` 与 `camera_rt_matrix` 在**同一世界坐标系**（已由位姿验算间接确认：
> 光心坐标的量级 [-17.8, 20.3, 1.4] 与 Bbox 的 [-20.5, 36.8, 0.9] 同量级同区域）。

---

## 六、「18 个视角」到底指什么（这是选它的核心理由）

官方描述：*"18 HDR RGB and Depth images（6 forward, 6 top, 6 bottom）per each of the 1,413 scan locations"*。

**实测确认**（`/raw` 目录，10830 个文件 ÷ 191 个采集点 = 56.7）：

| raw 内容 | 命名 | 每采集点 |
|---|---|---|
| RGB | `{uuid}_i{pitch}_{yaw}.jpg` | **18**（3 俯仰 × 6 方位） |
| 深度 | `{uuid}_d{pitch}_{yaw}.png` | **18** |
| 位姿 | `{uuid}_pose_{pitch}_{yaw}.txt` | **18** |
| 内参 | `{uuid}_intrinsics_{pitch}.txt` | **3**（每俯仰一个） |
| | **合计** | **57** ✅ 与 10830/191 吻合 |

**所以"相机原地转一圈（3 个传感器俯仰 × 6 个方位）= 18 个真实视角"是成立的** ——
这正是"多视角融合成一个全景"所需要的东西，且**是真实采集位姿，不是编造的视角**。

⚠️ 注意区分：`/data` 下的"常规"模态每采集点有约 **54** 帧（10,327 ÷ 191），
**多于** 18 —— 那是官方在更多方向上渲染的结果。**「18 视角」是 `/raw` 的属性。**

---

## 七、与下游衔接：官方全景就是「参考答案」

`/pano/` 下有 **191 个采集点**的等距柱状投影，每个采集点一套：

| 文件 | 用途 |
|---|---|
| `pano/rgb/..._frame_equirectangular_domain_rgb.png` | ★ **我们融合结果的参考答案**（RGB） |
| `pano/depth/..._domain_depth.png` | ★ 参考答案（深度） |
| `pano/semantic/..._domain_semantic.png` | 参考答案（逐像素实例标签） |
| `pano/pose/..._domain_pose.json` | 全景的位姿 |

**这是别的候选数据集给不了的**：我们的多视角融合可以**逐像素与官方全景比对**
（RGB 误差 / 深度误差 / 覆盖率），融合质量第一次有了客观指标，
而不是"看起来对"。

---

## 八、我们抽取了什么（30.44 GB → 13.80 GB）

| 抽取 | 大小 | 用途 |
|---|---|---|
| `data/rgb` | 9.3 GB | 主输入 |
| `data/depth` | 2.0 GB | 主输入 |
| `data/pose` | 8.9 MB | ★ 位姿 |
| `data/semantic` | 309 MB | 逐像素实例标签 |
| `pano/{rgb,depth,pose,semantic}` | 1.28 GB | ★ 融合参考答案 |
| `3d/pointcloud.mat` + `camera_to_room.json` | 1.07 GB | ★ 物体级 GT |
| **跳过** `pano/global_xyz` | **11.4 GB** | 逐像素世界坐标，本项目不需要 |
| **跳过** `data/normal` / `semantic_pretty` / 网格纹理 | 1.3 GB | 不需要 |

抽取脚本带**体积核算**（先 dry-run 看清要抽多少再动手）。

---

## 九、待你拍板的一件事

**主输入用 `/raw` 还是 `/data`？**

| | `/raw` | `/data` |
|---|---|---|
| 像素来源 | **真实传感器采集** | 官方在**同一批真实位姿**上渲染 |
| 深度 | PNG（实测，非 HDR JPG） | 16-bit PNG，**全稠密** |
| 逐像素语义 | ❌ 无 | ✅ 有 |
| 格式规整度 | 需处理 HDR/色调映射（RGB 是 HDR JPG） | 规整 |

**我的建议**：主链路用 **`/data`**（格式规整、深度稠密、带语义标签），
并在文档里**明确标注**"视角是真实采集位姿、像素来自官方重建渲染"；
再抽几个采集点做 **`/raw` vs `/data`** 对照，把"传感器 vs 重建"的差异**量化**出来。

---

## 十、复现清单

```bash
# 1) 列出 Redivis 上的文件与大小（需要带 data.data scope 的 token）
set REDIVIS_API_TOKEN=xxxxx
python scripts/28_fetch_2d3ds.py --list

# 2) 下载 + MD5 校验（30.44 GB，约 45 分钟 @ 15 MB/s）
python scripts/28_fetch_2d3ds.py --download area_1_no_xyz.tar --out G:\2d3ds

# 3) 核验 tar 结构（文件数/目录体积/命名约定）
python scripts/29_verify_2d3ds.py --tar G:\2d3ds\area_1_no_xyz.tar

# 4) 只看真实文件名样本（写适配器前必看）
python scripts/30_inspect_2d3ds_names.py --tar G:\2d3ds\area_1_no_xyz.tar

# 5) 选择性抽取（带体积核算，先 dry-run）
python scripts/31_extract_2d3ds.py --tar G:\2d3ds\area_1_no_xyz.tar --out G:\2d3ds\area_1 --dry-run
python scripts/31_extract_2d3ds.py --tar G:\2d3ds\area_1_no_xyz.tar --out G:\2d3ds\area_1

# 6) ★ 探明位姿方向 / 深度换算（决定性验算）
python scripts/32_probe_2d3ds_fields.py --dir G:\2d3ds\area_1\area_1

# 7) ★ 解出物体级 GT（MAT v7.3 / HDF5，含两个 MATLAB 存储坑的处理）
python scripts/33_probe_2d3ds_gt.py --mat G:\2d3ds\area_1\area_1\3d\pointcloud.mat
```

---

## 十一、方法论收获

1. **官方文档的字段名和形状也会错**（4×3 vs 3×4、双下划线 vs 单下划线、文件名大小写）。
   核验脚本的价值不在于"确认文档对"，而在于**发现文档错**。
2. **坐标系约定要用"可判定的验算"来定，不能靠推理。**
   "−Rᵀt 与 camera_location 差 1.2 µm、反向差 38 m" —— 这种对比是**判决性**的，
   比读十遍文档都可靠。
3. **量纲同样要验算，不能只看文档。** 深度按 1/512 得到 0.6~3.1 m（合理），
   按 1/1000 得到 0.3~1.6 m（不合理）—— **一算就知道哪个对**。
4. **MATLAB v7.3 的 struct array 要按"按字段存引用数组"来读**，
   而且字符串是 char 码点数组 —— 不知道这两点会读出一堆莫名其妙的数字。
5. **先看清楚再下载/抽取**：`pano/global_xyz` 一个目录就 11.4 GB，
   核算之后 30.44 GB 只需抽 13.80 GB。
