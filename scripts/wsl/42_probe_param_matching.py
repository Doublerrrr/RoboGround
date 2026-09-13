#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""42 · 实证：ROS2 参数文件是**按节点名**匹配的（决定参数化架构怎么设计）。

背景（一个真实的设计陷阱）
========================
早期实现是"先起一个 `roboground_param_bootstrap` 节点声明参数、算出 Config，
销毁它，再用真正的 `perception` 节点跑"。看起来没问题，但：

    `--params-file` 里的 `perception: ros__parameters:` 只对**名字叫 perception**
    的节点生效。

于是 bootstrap 节点（名字不同）根本收不到参数文件里的值 → 它算出来的 Config
是**默认配置**，而真正的节点拿到的是**参数文件里的值** —— 两者不一致，
且完全静默。

本脚本用最小复现把这个语义**测出来**，而不是靠记忆推断。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import rclpy
from rclpy.node import Node


def probe(yaml_text: str, node_name: str, param: str = "sync_slop", default: float = 0.05):
    """在带 --params-file 的进程里建一个指定名字的节点，看参数取到什么值。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(yaml_text)
        path = fh.name

    argv = ["--ros-args", "--params-file", path]
    rclpy.init(args=argv)
    try:
        n = Node(node_name)
        n.declare_parameter(param, default)
        value = n.get_parameter(param).value
        real_name = n.get_name()
        n.destroy_node()
    finally:
        rclpy.shutdown()
    Path(path).unlink(missing_ok=True)
    return real_name, value


def main() -> int:
    print("=" * 78)
    print("ROS2 参数文件匹配语义实证")
    print("=" * 78)

    yaml_simple = """perception:
  ros__parameters:
    sync_slop: 0.77
"""

    cases = [
        ("参数文件键 = perception，节点名 = perception", yaml_simple, "perception", 0.77),
        ("参数文件键 = perception，节点名 = roboground_param_bootstrap",
         yaml_simple, "roboground_param_bootstrap", 0.05),
        ("参数文件键 = perception，节点名 = query", yaml_simple, "query", 0.05),
    ]

    ok = 0
    for desc, yaml_text, node_name, expected in cases:
        real_name, value = probe(yaml_text, node_name)
        got = float(value)
        hit = abs(got - expected) < 1e-9
        ok += hit
        print(f"\n  {desc}")
        print(f"    实际节点名 = {real_name!r}")
        print(f"    取到值 = {got}   期望 = {expected}   {'✓' if hit else '✗'}")

    # 通配：`/**` 能否匹配所有节点？
    yaml_wild = """/**:
  ros__parameters:
    sync_slop: 0.88
"""
    real_name, value = probe(yaml_wild, "any_node_name")
    print("\n  通配键 `/**`，节点名 = any_node_name")
    print(f"    取到值 = {float(value)}   期望 = 0.88   "
          f"{'✓（支持通配）' if abs(float(value) - 0.88) < 1e-9 else '✗（不支持通配）'}")

    # `__node` 重映射能否改变参数文件匹配的名字？
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(yaml_simple)
        path = fh.name
    rclpy.init(args=["--ros-args", "--params-file", path, "-r", "__node:=remapped_name"])
    try:
        n = Node("perception")           # 代码里写 perception，命令行重映射
        n.declare_parameter("sync_slop", 0.05)
        remapped_name = n.get_name()
        remapped_value = float(n.get_parameter("sync_slop").value)
        n.destroy_node()
    finally:
        rclpy.shutdown()
    Path(path).unlink(missing_ok=True)
    print("\n  `__node:=remapped_name` 重映射（参数文件键仍是 perception）")
    print(f"    实际节点名 = {remapped_name!r}")
    print(f"    取到值 = {remapped_value}   期望 0.05（因为键 perception 不再匹配）")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print("""
  1. `--params-file` 的顶层键就是**节点名**，必须与运行时节点名完全一致；
  2. 所以"用另一个名字的 bootstrap 节点去读参数"是**错的** ——
     它读到的是默认值，而真节点读到的是文件里的值，两者静默不一致；
  3. 正确做法：**让真正干活的节点自己声明参数**（本项目的 roboground 方案已改成这样），
     这样节点名天然一致，也不需要 bootstrap。
""")
    print(f"  前 3 个用例通过 {ok}/3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
