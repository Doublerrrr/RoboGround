# -*- coding: utf-8 -*-
"""配置路径的**结构性**测试：代码读/写的每个点号路径都必须真实存在。

为什么值得单独一个测试文件
========================
这个项目已经两次栽在同一类 bug 上，而且两次都**完全静默**：

1. `nodes.py` 读 `deployment.ros2.topics`，而真实键是 `deploy.ros2.topics`
   → 用户在任何 YAML 里配话题名都**不生效**，节点一路用硬编码默认值。
2. ROS2 参数表把 `min_depth` 接到 `deploy.ros2.min_depth`，而节点读的是
   `geometry.min_depth` → `ros2 param set /perception min_depth 0.3` 无声失败。

根源是 `cfg.get(path, default)` 的语义：**路径写错永远不报错**，只会悄悄用
default。这类 bug 单元测试抓不到（行为"正常"），只有把"路径存在性"本身
当成断言才能抓住。

覆盖范围
--------
· `src/**` 里所有 `cfg.get("...")` —— 生产代码读的键必须存在（否则等于没配）
· `src/**` 里所有 `cfg.set("...")` —— 生产代码写的键必须存在（否则写了没人读）
· `scripts/**` 里的 `cfg.set("...")` —— 验证脚本设的键必须存在
   （否则验证脚本"证明"的配置其实没被验证到）
· `ros2_ws/.../PARAM_MAP` 的每个 Config 路径 —— ROS2 参数必须接得上

不计入的少数情况
----------------
`ALLOWLIST_READ` 里列的是**有内联兜底值**的可选覆盖项（例如评测脚本允许
在 YAML 里额外给 SigLIP 的 null_texts，不给就用脚本内的默认列表）。
它们"不存在"是设计意图，不是 bug，因此显式登记而不是悄悄放过。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from roboground.config import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
ROS2_PKG = ROOT / "ros2_ws" / "src" / "roboground_ros"

READ_RE = re.compile(r"""\bcfg\.get\(\s*["']([A-Za-z_][\w.]*)["']""")
SET_RE = re.compile(r"""\bcfg\.set\(\s*["']([A-Za-z_][\w.]*)["']""")

#: 允许"不存在"的**读**路径 → 理由（必须写清楚，否则就是在掩盖 bug）
ALLOWLIST_READ = {
    "perception.encoder_kwargs.null_texts": (
        "scripts/13 允许在 YAML 里覆盖 SigLIP 的零样本文本；"
        "不给时用脚本内联的 6 条默认文本，属于有意的可选覆盖"
    ),
}

_MISSING = object()


def _resolve(path: str):
    node = DEFAULT_CONFIG
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _scan(base: Path, pattern: re.Pattern) -> dict:
    """返回 {点号路径: [文件:行, ...]}。"""
    hits: dict[str, list[str]] = {}
    for py in sorted(base.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        for m in pattern.finditer(text):
            line = text[: m.start()].count("\n") + 1
            hits.setdefault(m.group(1), []).append(
                f"{py.relative_to(ROOT).as_posix()}:{line}"
            )
    return hits


def _missing(hits: dict) -> dict:
    return {k: v for k, v in hits.items() if _resolve(k) is _MISSING}


def _fmt(misses: dict) -> str:
    return "\n".join(f"  {k}\n      ↳ " + "\n      ↳ ".join(v) for k, v in sorted(misses.items()))


# ==========================================================================
# 生产代码
# ==========================================================================
def test_src_reads_only_existing_config_paths():
    """生产代码读的配置路径必须存在于 DEFAULT_CONFIG。

    反例（历史 bug）：`deployment.ros2.topics` —— 话题名配了不生效。
    """
    hits = _scan(SRC, READ_RE)
    assert len(hits) > 30, f"扫描到的配置路径太少（{len(hits)}），正则可能失效了"
    misses = _missing(hits)
    assert not misses, (
        "生产代码读了不存在的配置路径（会静默走 fallback 默认值）：\n"
        + _fmt(misses)
        + "\n修法：要么在 DEFAULT_CONFIG 里补上这个键，要么改正路径。"
    )


def test_src_writes_only_existing_config_paths():
    """生产代码写的配置路径必须存在，否则"写了没人读"。"""
    misses = _missing(_scan(SRC, SET_RE))
    assert not misses, "生产代码写了不存在的配置路径：\n" + _fmt(misses)


def test_scripts_set_only_existing_config_paths():
    """验证脚本 `cfg.set` 的路径必须存在。

    否则脚本会"验证"一个没人读的键 —— 结论看着有、其实没生效。
    """
    misses = _missing(_scan(SCRIPTS, SET_RE))
    assert not misses, "验证脚本设了不存在的配置路径：\n" + _fmt(misses)


def test_scripts_reads_are_known():
    """脚本里读的路径要么存在，要么在 allowlist 里显式登记。"""
    hits = _scan(SCRIPTS, READ_RE)
    misses = _missing(hits)
    unexpected = {k: v for k, v in misses.items() if k not in ALLOWLIST_READ}
    assert not unexpected, (
        "脚本读了不存在的配置路径，且未登记理由：\n" + _fmt(unexpected)
    )
    stale = set(ALLOWLIST_READ) - set(misses)
    assert not stale, f"allowlist 里有已经不需要的条目，请删掉：{sorted(stale)}"


# ==========================================================================
# ROS2 参数接线
# ==========================================================================
def test_ros2_param_map_targets_exist():
    """ROS2 参数表的每个目标路径都必须存在于 DEFAULT_CONFIG。

    反例（历史 bug）：`min_depth` → `deploy.ros2.min_depth`，
    而节点读 `geometry.min_depth` → `ros2 param set` 静默失效。

    参数表的**规范位置**在库里（`roboground.deployment.ros2.params`）：
    声明参数的节点必须是干活的节点，所以表也跟着节点走。
    `ros2_ws` 那个 ament 包只再导出一次方便外部脚本取用。
    """
    from roboground.deployment.ros2.params import PARAM_MAP

    misses = {
        name: [path] for name, path in PARAM_MAP.items()
        if path is not None and _resolve(path) is _MISSING
    }
    assert not misses, (
        "ROS2 参数接到了一个没人读的 Config 路径：\n" + _fmt(misses)
    )


def test_ros2_ws_reexports_the_library_param_map():
    """`ros2_ws` 里的再导出必须是**同一张表**，不能各写一份。

    两份表一旦分叉，就会出现"改了库里的表、ament 包还按旧表声明参数"。
    """
    import importlib.util

    mod_path = ROS2_PKG / "roboground_ros" / "perception_node.py"
    if not mod_path.exists():
        pytest.skip(f"没有 ros2_ws 包：{mod_path}")
    spec = importlib.util.spec_from_file_location("_rg_pnode_reexport", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from roboground.deployment.ros2.params import HOT_PARAMS, PARAM_MAP

    assert mod.PARAM_MAP is PARAM_MAP, \
        "ros2_ws 里的 PARAM_MAP 不是库里的那张表（应 `from ... import`）"
    assert tuple(mod.HOT_PARAMS) == tuple(HOT_PARAMS)
