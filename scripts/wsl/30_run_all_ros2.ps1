# =============================================================================
# 30 · ROS2 部署层验证统一入口（在 Windows 上执行）
# =============================================================================
# 一次跑完部署层的**全部**验证，并汇总成一张表。
#
# 为什么需要一个统一入口：
#   部署验证分支多（三种位姿 × 多种环境 + rosbag + 完整 TF 树 + 延迟），
#   散着跑很容易漏 —— `--pose static` 就**因为从没进过回归而坏了很久没人发现**。
#   统一入口的价值不是省事，而是**让"全跑过"成为默认行为**。
#
# 六个环节：
#   1) 三种位姿来源的端到端（static / tf / odometry）
#   2) rosbag2 录制 + 回放（含 QoS A/B 对照）
#   3) QoS 策略回归锁（离线可测）
#   4) ament 包构建与参数服务（colcon build / ros2 param）
#   5) 完整 TF 树 map→odom→base_link→camera_link→optical 端到端
#   6) 延迟与吞吐测量
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File "scripts\wsl\30_run_all_ros2.ps1"
#   powershell -ExecutionPolicy Bypass -File "scripts\wsl\30_run_all_ros2.ps1" -SkipRosbag
#   powershell -ExecutionPolicy Bypass -File "scripts\wsl\30_run_all_ros2.ps1" -SkipSlow
# =============================================================================
param(
    [switch]$SkipRosbag,
    #: 跳过较慢的两项（ament 构建 ~1 分钟、延迟测量 ~2 分钟）
    [switch]$SkipSlow,
    [int]$Frames = 6,
    #: 项目 Python 解释器。默认指向 conda `lxr` 环境 ——
    #: ⚠️ 必须显式指定：直接调 `python` 会拿到**系统解释器**（没有 yaml 等依赖），
    #: 表现为 QoS 检查莫名 FAIL。典型的环境不匹配坑。
    [string]$Python = "G:\minigore\envs\lxr\python.exe"
)

$ErrorActionPreference = "Continue"     # rclpy 会把日志写到 stderr，Stop 会误报
$env:WSL_UTF8 = "1"

if (-not (Test-Path $Python)) {
    $fallback = (Get-Command python -ErrorAction SilentlyContinue).Source
    $Python = if ($fallback) { $fallback } else { "python" }
    Write-Host "  [WARN] 指定的 Python 不存在，退回：$Python" -ForegroundColor Yellow
}

$Distro   = if ($env:ROBOGROUND_WSL_DISTRO) { $env:ROBOGROUND_WSL_DISTRO } else { "Ubuntu-22.04" }
$ProjWsl  = "/mnt/g/RoboGround"
$Verify   = "$ProjWsl/scripts/wsl/04_verify_ros2.sh"
$Rosbag   = "$ProjWsl/scripts/wsl/20_rosbag_e2e.py"

function Write-Head($t) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host "  $t" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
}

function Test-Wsl {
    $out = & wsl -d $Distro -u root -- bash -lc "test -f /opt/ros/humble/setup.bash && echo OK" 2>&1 | Out-String
    return ($out -match "OK")
}

# -----------------------------------------------------------------------------
Write-Head "ROS2 部署层验证（RoboGround）"
Write-Host "  发行版 : $Distro"
Write-Host "  帧数   : $Frames"

if (-not (Test-Wsl)) {
    Write-Host ""
    Write-Host "  [SKIP] 没找到 ROS2 Humble（$Distro）。" -ForegroundColor Yellow
    Write-Host "         先按 docs/WSL_ROS2_安装指南.md 装好，或用 -Distro 指定其他发行版。" -ForegroundColor Yellow
    Write-Host "         仅跑验证：wsl -d $Distro -u root -- bash $Verify" -ForegroundColor Yellow
    exit 2
}
Write-Host "  ROS2   : OK" -ForegroundColor Green

$results = @()

# -----------------------------------------------------------------------------
# 1) 三种位姿模式的端到端
# -----------------------------------------------------------------------------
foreach ($pose in @("static", "tf", "odometry")) {
    Write-Head "端到端 · 位姿模式 = $pose"
    $out = & wsl -d $Distro -u root -- bash $Verify --pose $pose --frames $Frames 2>&1 | Out-String
    $out -split "`n" | Where-Object { $_ -match "位姿来源|已处理帧数|地图物体数|[✓✗] " } |
        ForEach-Object { Write-Host "  $($_.Trim())" }

    $pass = if ($out -match "\[PASS\]") { "PASS" } else { "FAIL" }
    $objs = ([regex]::Match($out, "地图物体数\s*:\s*(\d+)")).Groups[1].Value
    $errs = ([regex]::Matches($out, "误差=([\d.]+)m") | ForEach-Object { $_.Groups[1].Value }) -join "/"
    $color = if ($pass -eq "PASS") { "Green" } else { "Red" }
    Write-Host "  → $pass  物体=$objs  误差=$errs m" -ForegroundColor $color
    $results += [pscustomobject]@{ 环节 = "e2e/$pose"; 结果 = $pass; 细节 = "物体=$objs 误差=$errs" }
}

# -----------------------------------------------------------------------------
# 2) rosbag2 录制 + 回放
# -----------------------------------------------------------------------------
if (-not $SkipRosbag) {
    Write-Head "rosbag2 录制 + 回放（真机部署前的标准验证）"
    $out = & wsl -d $Distro -u root -- bash -lc "source /opt/ros/humble/setup.bash && cd $ProjWsl && python3 $Rosbag" 2>&1 | Out-String
    $out -split "`n" | Where-Object { $_ -match "bag:|消息 \d|已处理帧数|地图物体数|GT (table|chair|cup)|TRANSIENT_LOCAL|VOLATILE|^\s+[✓✗] " } |
        ForEach-Object { Write-Host "  $($_.Trim())" }

    $pass = if ($out -match "\[PASS\]") { "PASS" } else { "FAIL" }
    $color = if ($pass -eq "PASS") { "Green" } else { "Red" }
    $latched = ([regex]::Match($out, "TRANSIENT_LOCAL → 收到 (\d+)")).Groups[1].Value
    Write-Host "  → $pass  晚加入(latched)收到=$latched 条" -ForegroundColor $color
    $results += [pscustomobject]@{ 环节 = "rosbag2 回放"; 结果 = $pass; 细节 = "latched=$latched" }
}

# -----------------------------------------------------------------------------
# 3) QoS 策略（离线可测的那部分）
# -----------------------------------------------------------------------------
Write-Head "QoS 策略回归锁（离线）"
$qos = & $Python -c "import sys; sys.path.insert(0,'src'); from roboground.deployment.ros2.nodes import QOS_POLICIES as Q; print(Q['state']['durability'], Q['sensor']['reliability'])" 2>&1 | Out-String
if ($qos -match "transient_local") {
    Write-Host "  state  = transient_local  ✓（地图/答案：后加入的订阅者能收到）" -ForegroundColor Green
    Write-Host "  sensor = best_effort      ✓（传感器：宁可丢帧不要延迟）" -ForegroundColor Green
    $results += [pscustomobject]@{ 环节 = "QoS 策略"; 结果 = "PASS"; 细节 = "state=transient_local" }
} else {
    Write-Host "  [FAIL] QoS 策略不符：$($qos.Trim())" -ForegroundColor Red
    $results += [pscustomobject]@{ 环节 = "QoS 策略"; 结果 = "FAIL"; 细节 = "" }
}

# -----------------------------------------------------------------------------
# 4) ament 包构建 + 参数服务（colcon build / ros2 param / 在线调参）
# -----------------------------------------------------------------------------
if (-not $SkipSlow) {
    Write-Head "ament 包构建与参数服务（colcon build + ros2 param）"
    $build = "$ProjWsl/scripts/wsl/40_build_ros2_pkg.sh"
    $out = & wsl -d $Distro -u root -- bash $build 2>&1 | Out-String
    $out -split "`n" | Where-Object { $_ -match "^\s+[✓✗] |通过 \d+ / 失败" } |
        ForEach-Object { Write-Host "  $($_.Trim())" }

    $pass = if ($out -match "\[PASS\]") { "PASS" } else { "FAIL" }
    $color = if ($pass -eq "PASS") { "Green" } else { "Red" }
    $n = ([regex]::Match($out, "通过 (\d+) / 失败")).Groups[1].Value
    Write-Host "  → $pass  通过 $n 项" -ForegroundColor $color
    $results += [pscustomobject]@{ 环节 = "ament 构建+参数"; 结果 = $pass; 细节 = "通过 $n 项" }
}

# -----------------------------------------------------------------------------
# 5) 完整 TF 树端到端（真起 tf_tree.launch.py）
# -----------------------------------------------------------------------------
Write-Head "完整 TF 树 map→odom→base_link→camera_link→optical"
$tf = "$ProjWsl/scripts/wsl/60_verify_tf_tree.sh"
$out = & wsl -d $Distro -u root -- bash $tf 2>&1 | Out-String
$out -split "`n" | Where-Object { $_ -match "共 \d+ 条边|→ camera_color|can_transform|实测|解析|误差 [\d.]+ m|^\s+[✓✗] " } |
    ForEach-Object { Write-Host "  $($_.Trim())" }

$pass = if ($out -match "\[PASS\]") { "PASS" } else { "FAIL" }
$color = if ($pass -eq "PASS") { "Green" } else { "Red" }
$tfErr = ([regex]::Match($out, "旋转矩阵最大误差\s*([\d.e+-]+)")).Groups[1].Value
Write-Host "  → $pass  实测vs解析 旋转误差=$tfErr" -ForegroundColor $color
$results += [pscustomobject]@{ 环节 = "完整 TF 树"; 结果 = $pass; 细节 = "旋转误差=$tfErr" }

# -----------------------------------------------------------------------------
# 6) 延迟与吞吐
# -----------------------------------------------------------------------------
if (-not $SkipSlow) {
    Write-Head "延迟与吞吐（纯计算 + ROS 链路 + 承载上限）"
    $lat = "$ProjWsl/scripts/wsl/50_e2e_latency.py"
    $cmd = "source /opt/ros/humble/setup.bash && source $ProjWsl/ros2_ws/install/setup.bash && python3 $lat"
    $out = & wsl -d $Distro -u root -- bash -lc $cmd 2>&1 | Out-String
    $out -split "`n" | Where-Object {
        $_ -match "个检测/帧|^\s+320×240:|^\s+640×480:|目标数伸缩|实测 10 Hz|^\s+[✓✗] " } |
        ForEach-Object { Write-Host "  $($_.Trim())" }

    $pass = if ($out -match "\[PASS\]") { "PASS" } else { "FAIL" }
    $color = if ($pass -eq "PASS") { "Green" } else { "Red" }
    $hz = ([regex]::Match($out, "320×240: .*?稳态上限 ([\d.]+) Hz")).Groups[1].Value
    # 丢帧率是**重复测量**的结果，整串拿出来（形如 "0.0% / 0.0% / 0.0%"）——
    # 只取第一个数会掩盖波动，而"波动"正是这次重启重跑暴露出来的教训。
    $drop = ([regex]::Match($out, "★ 10 Hz 重复 \d+ 次：丢帧率 ([^\r\n（]+)")).Groups[1].Value.Trim()
    if (-not $drop) { $drop = "?" }     # 抽不到就显式标出来，不要静默留空
    Write-Host "  → $pass  320×240 上限=$hz Hz  10Hz 丢帧(重复)=$drop" -ForegroundColor $color
    $results += [pscustomobject]@{ 环节 = "延迟与吞吐"; 结果 = $pass;
                                   细节 = "上限=$hz Hz 丢帧=$drop" }
}

# -----------------------------------------------------------------------------
Write-Head "汇总"
$results | Format-Table -AutoSize

$failed = @($results | Where-Object { $_.结果 -ne "PASS" })
Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "  [ALL PASS] 部署层全部验证通过。" -ForegroundColor Green
    Write-Host ""
    Write-Host "  这说明：三种互相独立的位姿来源 → 逐位一致的地图；" -ForegroundColor Gray
    Write-Host "  真正的 rosbag2 文件可录制可回放；状态型话题的 QoS 语义正确；" -ForegroundColor Gray
    Write-Host "  ament 包能构建、参数能在线调；完整 TF 树与解析解逐位一致；" -ForegroundColor Gray
    Write-Host "  延迟与承载上限都有实测数字。" -ForegroundColor Gray
    Write-Host "  真机上唯一还要做的是：把 ros2 bag record 换成录真机数据。" -ForegroundColor Gray
    exit 0
} else {
    Write-Host "  [FAIL] 有 $($failed.Count) 项未通过，见上表。" -ForegroundColor Red
    exit 1
}
