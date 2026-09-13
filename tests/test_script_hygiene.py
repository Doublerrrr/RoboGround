# -*- coding: utf-8 -*-
"""脚本卫生：编码、行尾、BOM —— 那些"本地能跑、别人一跑就崩"的坑。

为什么值得单独一个测试文件
========================
这类问题的共同点是：**在写脚本的这台机器上永远复现不了**，
而换一个人/换一个 shell 就立刻炸，而且报错信息通常指向别处。

本项目的真实案例：
  · `.ps1` 不带 UTF-8 BOM → Windows PowerShell 5.1 按 **GBK** 解码，
    中文注释/字符串变成乱码；更糟的是乱码里可能出现引号字节，
    直接导致语法错误（报错行号还是错的）。
  · `.sh` 带 BOM 或在 WSL 里用 CRLF → `bash: $'\\r': command not found`，
    或者 shebang 变成 `#!/usr/bin/env bash\\r` 而找不到解释器。
  · `.sh` 里 `set -u` 后 `source /opt/ros/humble/setup.bash` →
    `AMENT_TRACE_SETUP_FILES: unbound variable` 直接退出。

这些都能用"读字节"来断言，所以放进 `pytest` 比写在文档里靠谱。
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
UTF8_BOM = b"\xef\xbb\xbf"


def _scripts(suffix: str):
    if not SCRIPTS_DIR.exists():
        return []
    return sorted(p for p in SCRIPTS_DIR.rglob(f"*{suffix}") if p.is_file())


# ==========================================================================
# PowerShell
# ==========================================================================
def test_ps1_files_have_utf8_bom():
    """`.ps1` **必须**带 UTF-8 BOM。

    Windows PowerShell 5.1 对无 BOM 的文件按系统代码页（中文系统 = GBK）解码。
    本项目所有脚本都带中文注释和中文输出，没有 BOM 就会乱码甚至语法错误。
    """
    files = _scripts(".ps1")
    if not files:
        pytest.skip("没有 .ps1 脚本")
    bad = [str(p.relative_to(ROOT)) for p in files
           if not p.read_bytes().startswith(UTF8_BOM)]
    assert not bad, (
        "以下 .ps1 缺少 UTF-8 BOM（PowerShell 5.1 会按 GBK 解码 → 乱码/语法错误）：\n"
        + "\n".join(f"  {b}" for b in bad)
        + "\n修法（PowerShell）：\n"
          '  $p="路径"; $t=[IO.File]::ReadAllText($p,[Text.Encoding]::UTF8); '
          '[IO.File]::WriteAllText($p,$t,(New-Object Text.UTF8Encoding($true)))'
    )


def test_ps1_files_are_valid_utf8():
    """`.ps1` 的内容必须是合法 UTF-8（乱码写进去就再也救不回来）。"""
    for p in _scripts(".ps1"):
        raw = p.read_bytes()
        if raw.startswith(UTF8_BOM):
            raw = raw[len(UTF8_BOM):]
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            pytest.fail(f"{p.relative_to(ROOT)} 不是合法 UTF-8：{exc}")


# ==========================================================================
# Shell
# ==========================================================================
def test_sh_files_have_no_bom():
    """`.sh` **不能**带 BOM —— shebang 前面多三个字节会让内核找不到解释器。"""
    files = _scripts(".sh")
    if not files:
        pytest.skip("没有 .sh 脚本")
    bad = [str(p.relative_to(ROOT)) for p in files
           if p.read_bytes().startswith(UTF8_BOM)]
    assert not bad, (
        "以下 .sh 带了 UTF-8 BOM（应为无 BOM）：\n"
        + "\n".join(f"  {b}" for b in bad)
    )


def test_sh_files_use_lf_line_endings():
    """`.sh` 必须是 LF 行尾。

    在 WSL 里跑 CRLF 的脚本，`#!/usr/bin/env bash\\r` 会让内核报
    "bad interpreter: No such file or directory"，
    或者报一堆 `$'\\r': command not found` —— 都是很难一眼看出的错。
    """
    files = _scripts(".sh")
    if not files:
        pytest.skip("没有 .sh 脚本")
    bad = {}
    for p in files:
        text = p.read_bytes().decode("utf-8", errors="replace")
        n = text.count("\r\n")
        if n:
            bad[str(p.relative_to(ROOT))] = n
    assert not bad, (
        "以下 .sh 含 CRLF 行尾（会报 $'\\r': command not found）：\n"
        + "\n".join(f"  {k}: {v} 处" for k, v in sorted(bad.items()))
    )


def test_sh_scripts_start_with_shebang():
    """每个 `.sh` 都必须以 shebang 开头，否则 `bash 脚本` 之外都跑不起来。"""
    for p in _scripts(".sh"):
        first = p.read_bytes().split(b"\n", 1)[0]
        assert first.startswith(b"#!"), f"{p.relative_to(ROOT)} 缺 shebang"


def test_wsl_scripts_avoid_set_u_before_sourcing_ros():
    """★ `.sh` 里 `source /opt/ros/humble/setup.bash` **之前**不能处于 `set -u`。

    ROS2 的 `setup.bash` 会引用未定义变量 `AMENT_TRACE_SETUP_FILES`，
    在 `set -u` 下直接以 "unbound variable" 退出 ——
    表现为"脚本什么都没干就失败了"，而且报错来自被 source 的文件，很容易误判。

    允许的写法（本项目的约定）：
        set +u
        source /opt/ros/humble/setup.bash
        set -u
    """
    offenders = []
    for p in _scripts(".sh"):
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        strict = False
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if s in ("set -u", "set -euo pipefail", "set -eo pipefail", "set -eu"):
                strict = True
            elif s.startswith("set +u"):
                strict = False
            elif strict and _is_source_command(s):
                offenders.append(f"{p.relative_to(ROOT)}:{i}")
    assert not offenders, (
        "以下位置在 `set -u`（或 -eu）生效时 source ROS 的 setup.bash，"
        "会因 AMENT_TRACE_SETUP_FILES 未定义而退出；请先 `set +u`：\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


def _is_source_command(stripped: str) -> bool:
    """判断这一行是不是**真的在 source** 某个 setup.bash。

    ⚠️ 必须区分"赋值"和"执行"：
        SOURCE_LINE="source /opt/ros/humble/setup.bash"   ← 只是字符串，安全
        source /opt/ros/humble/setup.bash                  ← 真的执行，set -u 下会炸
    用 `"source " in line` 去 grep 会把前者误报成后者（本项目第一次写这个检查
    就误报了 `03_provision_ubuntu.sh`，所以这里按"行首是不是命令"判断）。
    """
    if "setup.bash" not in stripped and "setup.zsh" not in stripped:
        return False
    for prefix in ("source ", ". ", "sudo source ", "source\t"):
        if stripped.startswith(prefix):
            return True
    return False


# ==========================================================================
# Python 脚本
# ==========================================================================
def test_python_scripts_have_no_bom_or_crlf():
    """`.py` 脚本不要 BOM、不要 CRLF（BOM 会让某些工具把首行当语法错）。"""
    bad = []
    for p in _scripts(".py"):
        raw = p.read_bytes()
        if raw.startswith(UTF8_BOM):
            bad.append(f"{p.relative_to(ROOT)}: 有 BOM")
        if b"\r\n" in raw:
            bad.append(f"{p.relative_to(ROOT)}: 有 CRLF")
    assert not bad, "以下 Python 脚本编码不规范：\n" + "\n".join(f"  {b}" for b in bad)


def test_unified_runners_mention_current_scripts():
    """统一入口脚本必须引用所有该跑的环节（避免"加了脚本忘了接线"）。

    这是本项目反复出现的坑型：能力加了、但没接进回归入口，
    于是"验证过了"只覆盖了一部分，而且没人知道漏了什么。
    """
    runner = SCRIPTS_DIR / "wsl" / "30_run_all_ros2.ps1"
    if not runner.exists():
        pytest.skip("没有统一入口脚本")
    text = runner.read_text(encoding="utf-8", errors="replace")
    expected = [
        "04_verify_ros2.sh",        # 三种位姿端到端
        "20_rosbag_e2e.py",         # rosbag2 录制回放
        "40_build_ros2_pkg.sh",     # ament 构建
        "60_verify_tf_tree.sh",     # 完整 TF 树
        "50_e2e_latency.py",        # 延迟与吞吐
    ]
    missing = [e for e in expected if e not in text]
    assert not missing, (
        f"统一入口没引用这些验证脚本（加了没接线）：{missing}"
    )
