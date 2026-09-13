# =============================================================================
# 02 · 首次进入 Ubuntu 并自动完成初始化（不需要管理员权限）
# =============================================================================
# 用法：
#   powershell -ExecutionPolicy Bypass -File "G:\RoboGround\scripts\wsl\02_enter_ubuntu.ps1"
#
# 这个脚本做什么：
#   1. 确认 WSL 与发行版就绪
#   2. 以 root 身份（绕过首次 OOBE 的交互式建用户）创建普通用户 + 免密 sudo
#   3. 写入 /etc/wsl.conf：默认用户 + 开启 systemd（ROS2 更省心）
#   4. 关掉发行版让配置生效
#   5. 自动执行 03_provision_ubuntu.sh 完成 ROS2 安装
#
# 若第 2 步失败（说明该 WSL 版本强制 OOBE），脚本会明确告诉你怎么手动建用户。
# =============================================================================

[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu-22.04",
    [string]$LinuxUser = "lxr",
    [switch]$SkipProvision
)

$ErrorActionPreference = "Continue"

function Write-Step([string]$m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Ok([string]$m)   { Write-Host "[OK]   $m" -ForegroundColor Green }
function Write-Warn2([string]$m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Write-Err2([string]$m) { Write-Host "[FAIL] $m" -ForegroundColor Red }

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # G:\RoboGround

# -----------------------------------------------------------------------------
# 1) 检查 WSL / 发行版
# -----------------------------------------------------------------------------
Write-Step "1/5 检查 WSL 状态"
try {
    $list = & wsl.exe -l -v 2>&1
    $list | ForEach-Object { Write-Host "  $_" }
} catch {
    Write-Err2 "WSL 尚不可用。请先以管理员身份运行 01_install_wsl.ps1，必要时重启电脑。"
    exit 1
}

$names = (& wsl.exe -l -q 2>$null) | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ }
if (-not ($names -contains $Distro)) {
    Write-Err2 "没有找到发行版 '$Distro'（现有：$($names -join ', ')）"
    Write-Host "  请先以管理员身份运行 01_install_wsl.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Ok "发行版 '$Distro' 存在"

# -----------------------------------------------------------------------------
# 2) 用 root 绕过 OOBE，创建普通用户
# -----------------------------------------------------------------------------
Write-Step "2/5 初始化 Linux 用户（以 root 执行，绕过交互式 OOBE）"

$bootstrap = @"
set -e
if id -u $LinuxUser >/dev/null 2>&1; then
  echo "USER_EXISTS:$LinuxUser"
else
  useradd -m -s /bin/bash -G sudo $LinuxUser 2>/dev/null || adduser --disabled-password --gecos "" --ingroup sudo $LinuxUser
  echo "USER_CREATED:$LinuxUser"
fi
# 免密 sudo：仅为自动化便利，**开发机可用，生产环境请去掉**
echo '$LinuxUser ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-$LinuxUser
chmod 440 /etc/sudoers.d/90-$LinuxUser
# 默认用户 + systemd（ROS2 需要 systemd 才能用 ros2 daemon / systemd 服务）
cat > /etc/wsl.conf <<EOF
[user]
default=$LinuxUser
[boot]
systemd=true
[interop]
enabled=true
appendWindowsPath=true
EOF
echo "WSLCONF_WRITTEN"
"@

$bootstrapPath = Join-Path $env:TEMP "roboground_bootstrap.sh"
# 用 LF 换行写文件 —— WSL 里 CRLF 会让 set -e 之类的脚本报错
$bootstrap -replace "`r`n", "`n" | Set-Content -Path $bootstrapPath -Encoding utf8NoBOM -NoNewline

$wslTmp = "/mnt/" + ($bootstrapPath.Substring(0,1).ToLower()) + ($bootstrapPath.Substring(2).Replace('\','/'))
Write-Host "  引导脚本：$bootstrapPath"
Write-Host "  WSL 内路径：$wslTmp"

$out = & wsl.exe -d $Distro -u root -- bash $wslTmp 2>&1
$out | ForEach-Object { Write-Host "  $_" }
$joined = ($out -join "`n")

if ($joined -match "USER_EXISTS|USER_CREATED") {
    Write-Ok "Linux 用户 '$LinuxUser' 就绪，已写入 /etc/wsl.conf"
} else {
    Write-Err2 "root 初始化失败。该 WSL 版本可能强制首次交互式创建用户。"
    Write-Host ""
    Write-Host "  请手动完成一次（约 30 秒）：" -ForegroundColor Yellow
    Write-Host "    1. 运行： wsl -d $Distro" -ForegroundColor White
    Write-Host "    2. 按提示输入用户名（建议 $LinuxUser）和密码" -ForegroundColor White
    Write-Host "    3. 退出后重新运行本脚本" -ForegroundColor White
    exit 1
}

# -----------------------------------------------------------------------------
# 3) 重启发行版让 wsl.conf 生效
# -----------------------------------------------------------------------------
Write-Step "3/5 重启发行版以应用 systemd 与默认用户"
& wsl.exe --terminate $Distro 2>&1 | Out-Null
Start-Sleep -Seconds 2
$whoami = (& wsl.exe -d $Distro -- whoami 2>&1) -join ""
Write-Ok "默认用户 = $($whoami.Trim())"

$systemd = (& wsl.exe -d $Distro -- bash -lc "ps -p 1 -o comm= 2>/dev/null || echo none" 2>&1) -join ""
Write-Host "  PID 1 = $($systemd.Trim())  $(if ($systemd -match 'systemd') { '(systemd 已启用 ✓)' } else { '(未启用 systemd，ROS2 基础功能仍可用)' })"
# 需要再 terminate 一次让 systemd 真正接管
if ($systemd -notmatch 'systemd') {
    & wsl.exe --terminate $Distro 2>&1 | Out-Null
    Start-Sleep -Seconds 2
}

# -----------------------------------------------------------------------------
# 4) 准备项目访问路径
# -----------------------------------------------------------------------------
Write-Step "4/5 检查 Windows 侧项目能否在 WSL 中访问"
$wslRoot = "/mnt/" + ($Root.Substring(0,1).ToLower()) + ($Root.Substring(2).Replace('\','/'))
$check = (& wsl.exe -d $Distro -- bash -lc "ls -d '$wslRoot' 2>/dev/null && ls '$wslRoot' | head -5" 2>&1) -join "`n"
Write-Host "  WSL 内项目路径：$wslRoot"
$check | ForEach-Object { Write-Host "  $_" }
if ($check -match "src|scripts|configs") {
    Write-Ok "项目可在 WSL 中访问"
} else {
    Write-Warn2 "未能列出项目内容；若 /mnt/g 未挂载，请检查 WSL 的 automount 设置"
}

# -----------------------------------------------------------------------------
# 5) 执行 Linux 侧 provisioning
# -----------------------------------------------------------------------------
Write-Step "5/5 安装 ROS2 Humble 与 Python 环境"
if ($SkipProvision) {
    Write-Host "  已跳过（-SkipProvision）"
} else {
    $prov = Join-Path $PSScriptRoot "03_provision_ubuntu.sh"
    if (-not (Test-Path $prov)) {
        Write-Err2 "找不到 $prov"
        exit 1
    }
    $provWsl = "/mnt/" + ($prov.Substring(0,1).ToLower()) + ($prov.Substring(2).Replace('\','/'))
    # 用 bash -lc 保证是登录 shell（能读到 /etc/profile.d 里的 ROS 环境）
    & wsl.exe -d $Distro -u $LinuxUser -- bash -lc "bash '$provWsl'" 2>&1 | ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "ROS2 provisioning 完成"
    } else {
        Write-Warn2 "provisioning 返回码 $LASTEXITCODE，请检查上面的输出"
    }
}

Write-Host ""
Write-Host "  下一步（在 WSL 里跑真实的 ROS2 端到端测试）：" -ForegroundColor Yellow
Write-Host "    wsl -d $Distro -- bash -lc `"bash $wslRoot/scripts/wsl/04_verify_ros2.sh`"" -ForegroundColor White
Write-Host ""
Write-Host "  或者直接进 WSL 交互操作：" -ForegroundColor DarkGray
Write-Host "    wsl -d $Distro" -ForegroundColor DarkGray
