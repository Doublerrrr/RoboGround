# =============================================================================
# 05 - Run the RoboGround ROS2 end-to-end test on Windows (RoboStack, no WSL)
# =============================================================================
# NOTE: This file is intentionally ASCII-only. Windows PowerShell 5.1 reads
#       .ps1 files as GBK when there is no UTF-8 BOM, which corrupts non-ASCII
#       characters and breaks the parser. ASCII avoids the issue entirely.
#
# Usage (double-click the companion .bat, or run):
#   powershell -ExecutionPolicy Bypass -File "G:\RoboGround\scripts\wsl\05_run_ros2_windows.ps1"
#   powershell -ExecutionPolicy Bypass -File "...\05_run_ros2_windows.ps1" -Frames 8 -Detailed
#
# Why RoboStack instead of WSL:
#   RoboStack publishes native win-64 ROS2 builds via conda, so rclpy works
#   WITHOUT administrator rights, WITHOUT WSL and WITHOUT a reboot.
#   Limitation: `tf2_ros` has no win-64 build, so pose comes from /odom here.
#   The tf2 path is covered by the offline unit tests + the WSL recipe.
# =============================================================================

[CmdletBinding()]
param(
    [string]$EnvName = "ros2b",
    [int]$Frames = 6,
    [ValidateSet("auto", "tf", "odometry", "static")]
    [string]$Pose = "auto",
    [switch]$Detailed
)

$ErrorActionPreference = "Continue"   # rclpy logs to stderr; do not treat that as fatal

function Info($m) { Write-Host "[..]   $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[OK]   $m" -ForegroundColor Green }
function Fail($m) { Write-Host "[FAIL] $m" -ForegroundColor Red }

$Root    = "G:\RoboGround"
$EnvDir  = "G:\minigore\envs\$EnvName"
$Python  = Join-Path $EnvDir "python.exe"
$Test    = Join-Path $Root "scripts\wsl\ros2_e2e_test.py"

Write-Host "=" * 72
Write-Host "RoboGround - ROS2 end-to-end test (Windows / RoboStack)"
Write-Host "=" * 72

if (-not (Test-Path $Python)) {
    Fail "conda env '$EnvName' not found at $EnvDir"
    Write-Host ""
    Write-Host "Create it with (no admin needed, ~90 seconds):" -ForegroundColor Yellow
    Write-Host "  & 'G:\minigore\Scripts\conda.exe' create -n $EnvName -y ``" -ForegroundColor White
    Write-Host "      --override-channels ``" -ForegroundColor White
    Write-Host "      -c https://conda.anaconda.org/robostack-staging -c conda-forge ``" -ForegroundColor White
    Write-Host "      python=3.12 ros-humble-rclpy=3.3.21 ros-humble-message-filters ``" -ForegroundColor White
    Write-Host "      ros-humble-sensor-msgs ros-humble-std-msgs ros-humble-geometry-msgs ``" -ForegroundColor White
    Write-Host "      ros-humble-nav-msgs 'numpy>=2' pyyaml pillow" -ForegroundColor White
    Write-Host ""
    Write-Host "IMPORTANT: pin python=3.12 + numpy>=2 + rclpy=3.3.21 together." -ForegroundColor Yellow
    Write-Host "           Mixing build variants (e.g. _13 vs _14) causes" -ForegroundColor Yellow
    Write-Host "           'DLL load failed ... _rclpy_pybind11' (ERROR_PROC_NOT_FOUND)." -ForegroundColor Yellow
    exit 2
}
Ok "conda env found: $EnvDir"

if (-not (Test-Path $Test)) { Fail "test script not found: $Test"; exit 2 }

# --- RoboStack needs the env's DLL directories on PATH, otherwise rclpy's
#     pybind11 extension fails to load its dependencies. ---
$env:PATH = "$EnvDir;$EnvDir\Library\bin;$EnvDir\Library\lib;$EnvDir\Scripts;$EnvDir\bin;$env:PATH"
$env:ROS_DISTRO        = "humble"
$env:AMENT_PREFIX_PATH = $EnvDir
# Keep ROS traffic off the default domain so it cannot collide with other ROS
# nodes on the same network (e.g. a lab machine).
if (-not $env:ROS_DOMAIN_ID) { $env:ROS_DOMAIN_ID = "42" }
$env:PYTHONWARNINGS    = "ignore"      # silence transformers/fast-processor chatter
$env:PYTHONIOENCODING  = "utf-8"

Info "env       : $EnvName"
Info "ROS_DISTRO: $env:ROS_DISTRO"
Info "pose mode : $Pose"
Info "frames    : $Frames"
Write-Host ""

$args = @($Test, "--pose", $Pose, "--frames", "$Frames")
if ($Detailed) { $args += "--verbose" }

# rclpy writes INFO logs to stderr; PowerShell turns those into NativeCommandError
# noise. Merge the streams so the output stays readable.
& $Python @args 2>&1 | ForEach-Object { Write-Host $_ }
$rc = $LASTEXITCODE

Write-Host ""
if ($rc -eq 0) {
    Ok "ROS2 end-to-end test PASSED"
} else {
    Fail "ROS2 end-to-end test FAILED (exit=$rc)"
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1) DLL load failed for _rclpy_pybind11 -> recreate the env with the pinned set above"
    Write-Host "  2) no frames processed               -> check /camera/* topics are published"
    Write-Host "  3) pose failures > 0                  -> odom stamp too old (max_age_s) or topic mismatch"
}
exit $rc
