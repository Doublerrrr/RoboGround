# =============================================================================
# 01 · 在 Windows 上启用 WSL2 并安装 Ubuntu 22.04（**需要管理员权限**）
# =============================================================================
# 用法（二选一）：
#   A) 右键本文件 → "使用 PowerShell 运行"（若未提权，脚本会自己请求 UAC）
#   B) 以管理员身份打开 PowerShell，然后：
#        powershell -ExecutionPolicy Bypass -File "G:\RoboGround\scripts\wsl\01_install_wsl.ps1"
#
# 这个脚本做什么：
#   1. 自检并自动请求管理员权限
#   2. 启用两个必需的 Windows 可选功能：
#        - Microsoft-Windows-Subsystem-Linux
#        - VirtualMachinePlatform
#   3. 把 WSL 默认版本设为 2，并更新 WSL 内核
#   4. 安装 Ubuntu-22.04（ROS2 Humble 的官方目标发行版）
#   5. 检测是否需要重启，并给出明确的下一步提示
#
# 幂等：重复运行是安全的，已启用的功能会被跳过。
# =============================================================================

[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu-22.04",
    [switch]$SkipLaunch
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Warn2([string]$msg){ Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err2([string]$msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red }

# -----------------------------------------------------------------------------
# 0) 自提权
# -----------------------------------------------------------------------------
$identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$isAdmin   = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Warn2 "当前不是管理员，正在请求提权（会弹出 UAC 对话框，请点『是』）..."
    try {
        $argList = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", "`"$PSCommandPath`"",
            "-Distro", $Distro
        )
        if ($SkipLaunch) { $argList += "-SkipLaunch" }
        Start-Process -FilePath "powershell.exe" -ArgumentList $argList -Verb RunAs -Wait
        Write-Host "`n提权进程已结束。若上面的输出看不到，请重新以管理员身份运行本脚本查看结果。" -ForegroundColor Yellow
    } catch {
        Write-Err2 "提权被拒绝或失败：$($_.Exception.Message)"
        Write-Host "请手动操作：右键『开始菜单』→『终端(管理员)』，然后运行：" -ForegroundColor Yellow
        Write-Host "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -ForegroundColor White
    }
    exit 0
}

Write-Ok "已获得管理员权限（$($identity.Name)）"

# -----------------------------------------------------------------------------
# 1) 前置检查：CPU 虚拟化
# -----------------------------------------------------------------------------
Write-Step "1/5 检查 CPU 虚拟化支持"
$cs = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
Write-Host "  CPU: $($cpu.Name)"
if (-not $cpu.VirtualizationFirmwareEnabled) {
    Write-Err2 "BIOS 中未启用虚拟化（SVM/VT-x）。"
    Write-Host "  请在开机时进 BIOS，开启 SVM Mode（AMD）/ Intel Virtualization Technology，然后重跑本脚本。"
    exit 1
}
Write-Ok "CPU 虚拟化已启用（SLAT: $($cpu.SecondLevelAddressTranslationExtensions)）"

# -----------------------------------------------------------------------------
# 2) 启用 Windows 可选功能
# -----------------------------------------------------------------------------
Write-Step "2/5 启用 WSL 所需的 Windows 功能"
$features = @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")
$needReboot = $false
$restartNeeded = $false

foreach ($f in $features) {
    $state = (Get-WindowsOptionalFeature -Online -FeatureName $f).State
    if ($state -eq "Enabled") {
        Write-Ok "$f 已启用"
        continue
    }
    Write-Host "  正在启用 $f ..."
    $res = Enable-WindowsOptionalFeature -Online -FeatureName $f -All -NoRestart
    if ($res.RestartNeeded) { $needReboot = $true }
    Write-Ok "$f 已启用（RestartNeeded=$($res.RestartNeeded)）"
}

# -----------------------------------------------------------------------------
# 3) WSL 内核与默认版本
# -----------------------------------------------------------------------------
Write-Step "3/5 更新 WSL 内核并设置默认版本为 2"
try {
    # 有些环境没有 Store 版 WSL，--update 会失败但不影响后续（inbox 版可用）
    & wsl.exe --update --web-download 2>&1 | ForEach-Object { Write-Host "  $_" }
    Write-Ok "WSL 内核已更新"
} catch {
    Write-Warn2 "wsl --update 失败（$($_.Exception.Message)），继续尝试使用已有内核"
}
try {
    & wsl.exe --set-default-version 2 2>&1 | ForEach-Object { Write-Host "  $_" }
    Write-Ok "默认 WSL 版本 = 2"
} catch {
    Write-Warn2 "设置默认版本失败，可能需要在重启后再执行"
}

# -----------------------------------------------------------------------------
# 4) 安装发行版
# -----------------------------------------------------------------------------
Write-Step "4/5 安装 $Distro"
$installed = (& wsl.exe -l -q 2>$null) | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($installed -contains $Distro) {
    Write-Ok "$Distro 已安装，跳过"
} else {
    Write-Host "  正在安装（首次会下载约 500MB，请耐心等待）..."
    # --no-launch：避免脚本卡在交互式"创建用户"环节；用户随后手动创建
    & wsl.exe --install -d $Distro --no-launch 2>&1 | ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Warn2 "wsl --install 返回码 $LASTEXITCODE；可能是需要先重启。"
        $needReboot = $true
    } else {
        Write-Ok "$Distro 安装完成"
    }
}

# -----------------------------------------------------------------------------
# 5) 结果与下一步
# -----------------------------------------------------------------------------
Write-Step "5/5 汇总"
if ($needReboot) {
    Write-Warn2 "系统需要**重启**才能让 WSL 生效。"
    Write-Host ""
    Write-Host "  下一步：" -ForegroundColor Yellow
    Write-Host "    1. 重启电脑" -ForegroundColor White
    Write-Host "    2. 重启后再跑一次本脚本（会跳过已完成的步骤）" -ForegroundColor White
    Write-Host "    3. 然后运行： powershell -File `"G:\RoboGround\scripts\wsl\02_enter_ubuntu.ps1`"" -ForegroundColor White
} else {
    Write-Ok "WSL2 + $Distro 已就绪，无需重启。"
    Write-Host ""
    Write-Host "  下一步：" -ForegroundColor Yellow
    Write-Host "    powershell -File `"G:\RoboGround\scripts\wsl\02_enter_ubuntu.ps1`"" -ForegroundColor White
}

Write-Host ""
Write-Host "  查看状态： wsl -l -v" -ForegroundColor DarkGray
