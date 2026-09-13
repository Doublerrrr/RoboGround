# -*- coding: utf-8 -*-
"""ROS2 参数接线与在线调参逻辑的离线测试（**不需要 ROS2**）。

被测对象是 `roboground.deployment.ros2.params`。它刻意只依赖宿主对象的 4 个方法
（`get_name` / `declare_parameter` / `get_parameter` / `add_on_set_parameters_callback`），
所以下面这个 40 行的假节点就能把整条链路测掉 —— 这正是"把策略提成可离线断言的
形态"的收益：真机上最容易错的那几步，在 `pytest` 里就能红。

要守住的四件事
============
1. **不静默覆盖配置**：声明默认值必须是配置里的真实值，
   而不是某个写死的常量（写死会把 YAML 里配的值盖掉）。
2. **覆盖值要回灌**：`-p sync_slop:=0.42` / 参数文件里的值必须真的进到
   `Config`，否则会出现"`ros2 param get` 显示 0.42、算法按 0.05 跑"。
3. **热参数真能在线改**：`ros2 param set` 之后 `Config` 与节点的运行期字段都变。
4. **冷参数被显式拒绝**：返回 `successful=False` 且原因里点名是哪个参数，
   而不是静默忽略（"配了不生效"是本项目最贵的坑）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from roboground.config import DEFAULT_CONFIG, Config
from roboground.deployment.ros2 import params as P


class _FakeParameter:
    """`rclpy.parameter.Parameter` 的替身（只有 name/value 两个字段被用到）。"""

    def __init__(self, name: str, value: Any = None):
        self.name = name
        self.value = value


class FakeParamNode:
    """最小可用的 rclpy 节点替身，只实现 `declare_params` 用到的方法。"""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None, name: str = "perception"):
        #: 模拟 `-p` / `--params-file`：在 declare 时就给出覆盖值
        self._overrides = dict(overrides or {})
        self._params: Dict[str, Any] = {}
        self._callback = None
        self._name = name
        self.refresh_calls = 0

    # ---- rclpy 节点接口（子集）----
    def get_name(self) -> str:
        return self._name

    def declare_parameter(self, name: str, value: Any = None):
        if name in self._params:
            raise RuntimeError(f"参数 {name} 重复声明")
        self._params[name] = self._overrides.get(name, value)
        return self._params[name]

    def get_parameter(self, name: str):
        if name not in self._params:
            raise RuntimeError(f"参数 {name} 未声明")
        return _Param(self._params[name])

    def set_parameters(self, values: Dict[str, Any]) -> List[Any]:
        """模拟 `ros2 param set`：构造参数列表 → 交给回调。

        注意成功/失败都要更新**本节点可见的参数值**（真 rclpy 也是这个行为：
        整批失败时不落盘，成功时落盘）。
        """
        ps = [_FakeParameter(n, v) for n, v in values.items()]
        result = self._callback(ps) if self._callback else None
        if result is None or result.successful:
            for p in ps:
                self._params[p.name] = p.value
        return [result]

    def add_on_set_parameters_callback(self, cb):
        self._callback = cb

    # ---- 供测试断言 ----
    def refresh_params(self) -> None:
        self.refresh_calls += 1


class _Param:
    def __init__(self, value):
        self.value = value


@pytest.fixture
def declared():
    """声明全部参数后的 `(node, cfg, applied)`。"""
    node = FakeParamNode()
    cfg, applied = P.declare_params(node)
    return node, cfg, applied


# ==========================================================================
# 1) 接线表本身
# ==========================================================================
def test_param_map_paths_exist_in_default_config():
    """每个非 None 的目标路径都必须在 DEFAULT_CONFIG 里真实存在。"""
    missing = []
    for name, path in P.PARAM_MAP.items():
        if path is None:
            continue
        node = DEFAULT_CONFIG
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.append((name, path))
                break
            node = node[part]
    assert not missing, f"参数接到了不存在的 Config 路径：{missing}"


def test_hot_and_cold_params_partition_the_table():
    """热 + 冷必须**恰好**覆盖整张表，不重不漏。

    漏掉某个参数 → 它既不在热也不在冷 → 回调会把 plan 归为"未处理"，
    行为就变成静默忽略了（正是要避免的）。
    """
    mapped = {n for n, p in P.PARAM_MAP.items() if p is not None}
    hot, cold = set(P.HOT_PARAMS), set(P.COLD_PARAMS)
    assert hot & cold == set(), f"热冷重叠：{hot & cold}"
    assert hot | cold == mapped, (
        f"热+冷没有覆盖整张表：\n  缺：{sorted(mapped - (hot | cold))}\n"
        f"  多：{sorted((hot | cold) - mapped)}"
    )
    assert "config_file" not in hot | cold, "config_file 不进 Config，不该出现在热冷表里"
    # config_file 在 PARAM_MAP 里但 path=None，必须明确归入"需重启"而不是被漏掉
    assert P.plan_param_update(["config_file"])["restart"] == ["config_file"], \
        "config_file 必须明确归入 restart（运行时改它不可能生效）"


@pytest.mark.parametrize("name,expected", [
    ("sync_slop", "apply"),
    ("publish_every_n", "apply"),
    ("min_depth", "apply"),
    ("max_depth", "apply"),
    ("depth_scale", "apply"),
    ("autosync_depth", "apply"),
    ("detector", "restart"),
    ("encoder", "restart"),
    ("pose_source", "restart"),
    ("color_image_topic", "restart"),
    ("voxel_size", "restart"),
    ("不存在的参数", "unknown"),
])
def test_plan_param_update_classifies(name, expected):
    """热/冷/未知分类必须正确（分类错了就会出现"以为改了其实没改"）。"""
    plan = P.plan_param_update([name])
    assert plan[expected] == [name], f"{name} 应归入 {expected}，实际 {plan}"


# ==========================================================================
# 2) 声明默认值 = 配置真实值（不静默覆盖 YAML）
# ==========================================================================
def test_declared_defaults_equal_config_values(declared):
    """★ 声明出来的默认值必须**逐项等于**配置里的真实值。

    这是"不静默覆盖 YAML"的核心保证：
      · 若声明成写死的常量（曾经是 `sync_slop=0.1`），用户在
        `configs/*.yaml` 里配的 `0.05` 会被**静默盖掉**；
      · 若声明成 `None`，`ros2 param get` 只会显示 "not set"，用户
        无法知道当前生效值（可观测性为零）。
    """
    node, cfg, _applied = declared
    mismatches = {}
    for name, path in P.PARAM_MAP.items():
        if path is None:
            continue
        declared_value = node.get_parameter(name).value
        config_value = cfg.get(path)
        if str(declared_value) != str(config_value):
            mismatches[name] = {"declared": declared_value, "config": config_value}
    assert not mismatches, (
        "参数声明默认值与配置不一致（会导致参数服务显示的值 ≠ 实际生效值）：\n"
        + "\n".join(f"  {k}: declared={v['declared']!r} config={v['config']!r}"
                    for k, v in sorted(mismatches.items()))
    )
    # 区分力：至少有一个参数的值不是 0/False/空，否则这个断言太弱
    assert float(node.get_parameter("depth_scale").value) == 1000.0
    assert node.get_parameter("detector").value == "stub"


def test_declare_params_declares_everything_but_config_file_path(declared):
    """所有有路径的参数 + `config_file` 都必须被声明（漏一个就 set 不了）。"""
    node, _cfg, _applied = declared
    expected = {"config_file"} | {n for n, p in P.PARAM_MAP.items() if p is not None}
    assert set(node._params) == expected, (
        f"声明集合不对：缺 {sorted(expected - set(node._params))}，"
        f"多 {sorted(set(node._params) - expected)}"
    )


# ==========================================================================
# 3) 覆盖值回灌（命令行/参数文件真正生效）
# ==========================================================================
def test_overrides_are_synced_back_into_config():
    """★ `-p` / 参数文件的覆盖值必须同步进 Config。

    反例（会静默出错）：声明默认值取配置值 → 覆盖值只在参数服务里生效，
    `cfg` 仍是旧值 → **`ros2 param get` 显示 0.42，算法实际按 0.05 跑**。
    """
    node = FakeParamNode({"sync_slop": 0.42, "publish_every_n": 3})
    cfg, applied = P.declare_params(node)
    assert cfg.get("deploy.ros2.sync_slop") == 0.42, "覆盖值没有回灌到 Config"
    assert cfg.get("deploy.ros2.publish_every_n") == 3
    assert set(applied) == {"sync_slop", "publish_every_n"}, \
        f"applied 应只列出被覆盖的项，实际 {applied}"


def test_no_overrides_reports_nothing_applied():
    """没有任何覆盖时 `applied` 应为空 —— 便于日志一眼看出环境差异。"""
    _cfg, applied = P.declare_params(FakeParamNode())
    assert applied == {}


def test_sync_config_from_params_can_load_its_own_config():
    """`config_from_params` 走"自行加载配置"的兼容路径，结果应与 declare 一致。"""
    node = FakeParamNode({"min_depth": 0.35})
    # 先声明（模拟别处已声明）
    for name, path in P.PARAM_MAP.items():
        if path is not None:
            node.declare_parameter(name, None)
    node.declare_parameter("config_file", "")
    cfg, applied = P.config_from_params(node)
    assert cfg.get("geometry.min_depth") == 0.35
    assert applied == {"min_depth": 0.35}


# ==========================================================================
# 4) 在线调参：热参数放行、冷参数拒绝
# ==========================================================================
def test_hot_param_set_is_applied_and_triggers_refresh(declared):
    """★ 在线改热参数必须：改到 `Config`、并且通知节点刷新运行期字段。"""
    node, cfg, _ = declared
    before = node.refresh_calls
    result = node.set_parameters({"sync_slop": 0.42, "publish_every_n": 2})[0]
    assert result.successful, f"热参数设置被拒绝：{getattr(result, 'reason', '')}"
    assert float(cfg.get("deploy.ros2.sync_slop")) == 0.42
    assert int(cfg.get("deploy.ros2.publish_every_n")) == 2
    assert node.refresh_calls == before + 1, \
        "设置热参数后必须调用 node.refresh_params()，否则节点还用旧值"


def test_cold_param_set_is_rejected_with_reason(declared):
    """★ 冷参数必须被**显式拒绝**，原因里点名是哪个参数。

    静默忽略是最坏的选择：用户看到 `ros2 param set` "成功"，实际没生效。
    """
    node, cfg, _ = declared
    result = node.set_parameters({"detector": "yolo"})[0]
    assert not result.successful, "冷参数 detector 竟然设置成功了（应该拒绝）"
    assert "detector" in result.reason, f"拒绝原因没点名参数：{result.reason!r}"
    assert "重启" in result.reason or "restart" in result.reason.lower(), \
        f"拒绝原因没说清该怎么办：{result.reason!r}"
    # 关键：被拒绝时**不能**改动配置
    assert cfg.get("perception.detector") == "stub", "被拒绝的参数却改了配置"


def test_mixed_hot_and_cold_is_rejected_atomically(declared):
    """一次设置里同时含冷参数 → **整批**拒绝。

    原子性很重要：若只应用热的、拒绝冷的，用户看到"失败"却已经有一部分生效了，
    状态会变成一个说不清的中间态。
    """
    node, cfg, _ = declared
    result = node.set_parameters({"sync_slop": 0.9, "encoder": "clip"})[0]
    assert not result.successful
    assert float(cfg.get("deploy.ros2.sync_slop")) == 0.05, \
        "整批被拒绝时不应该有任何一项生效"


def test_unknown_param_is_rejected(declared):
    """不在接线表里的参数也要拒绝（否则用户可能以为它生效了）。"""
    node, _cfg, _ = declared
    result = node.set_parameters({"definitely_not_a_param": 1})[0]
    assert not result.successful
    assert "definitely_not_a_param" in result.reason


def test_config_file_set_is_rejected_because_it_needs_a_restart(declared):
    """运行时改 `config_file` 必须被**拒绝** —— 它不可能重新加载配置。

    "接受但什么都不做"是最坏的选项：用户以为换了配置文件，其实没有。
    """
    node, cfg, _ = declared
    before = cfg.get("deploy.ros2.sync_slop")
    result = node.set_parameters({"config_file": "/tmp/other.yaml"})[0]
    assert not result.successful, "运行时改 config_file 竟然被接受了"
    assert "config_file" in result.reason
    assert cfg.get("deploy.ros2.sync_slop") == before, "被拒绝却改了配置"


def test_array_param_is_unwrapped_to_plain_list():
    """`array.array` 类的参数值要转成普通 list（否则写进 Config 后难以序列化）。"""
    import array

    assert P._unwrap(array.array("d", [1.0, 2.0])) == [1.0, 2.0]
    assert P._unwrap((1, 2)) == [1, 2]
    assert P._unwrap(3.0) == 3.0


def test_params_module_does_not_import_rclpy_at_module_level():
    """本模块顶层不能 import rclpy —— 否则"离线可测"这个前提就没了。

    用 AST 而不是正则：正则会把 docstring 里提到的 "import rclpy" 当成真 import
    （本项目的注释风格就是大段解释，很容易误伤）。
    允许顶层 `try: from rcl_interfaces.msg import …`（那只是消息定义包，
    且带 ImportError 兜底替身），但**不允许**任何 rclpy import。
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(P))

    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert not [m for m in imported if m == "rclpy" or m.startswith("rclpy.")], \
        f"params.py import 了 rclpy：{imported}；离线测试会直接失败"
    assert any(m.startswith("rcl_interfaces") for m in imported), \
        "缺少 rcl_interfaces 的（带兜底的）import"
    assert "_SetParametersResult" in inspect.getsource(P).split("def plan_param_update")[0], \
        "缺少 SetParametersResult 的离线兜底替身"
