# -*- coding: utf-8 -*-
"""`ros2_ws/src/roboground_ros` 这个 ament 包的结构性测试（**不需要 ROS2**）。

为什么要有这个文件
================
ROS2 侧的错误有个共同特征：**在离线环境里完全看不出来，一上真机就静默错**。

  · `setup.py` 的 `data_files` 指向不存在的文件 → `colcon build` 才报错；
  · launch 参数默认值与 params 文件不一致 → "我改了 YAML 怎么没生效"；
  · TF 边四元数写成转置 → 相机位姿整体旋转，误差 1.697 m，**不报任何错**；
  · 参数接到没人读的 Config 路径 → `ros2 param set` 无效果（在
    `tests/test_config_paths.py` 里另有专门断言）。

这里把上面每一条都变成断言，让它们在 `pytest` 里就红，而不是等到 WSL/真机。
本文件只读**文本与纯 Python 模块**，不 import rclpy / launch，
因此在 Windows 上无 ROS2 也能跑（386 个测试的主体就是这种形态）。
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "ros2_ws" / "src" / "roboground_ros"
MOD_DIR = PKG_DIR / "roboground_ros"

pytestmark = pytest.mark.skipif(
    not PKG_DIR.exists(), reason=f"没有 ros2_ws 包：{PKG_DIR}"
)


def _load(module_name: str, path: Path):
    """按路径导入一个纯 Python 模块（不依赖它是否在 sys.path 上）。"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_pkg_module(dotted: str):
    """按完整包名导入 `roboground_ros.<dotted>`。

    必须走真正的 import（而不是按路径加载）：`query_node.py` 顶层有
    `from roboground_ros.perception_node import ...`，按路径加载会因为
    `roboground_ros` 不在 sys.path 上而失败。
    """
    if str(PKG_DIR) not in sys.path:
        sys.path.insert(0, str(PKG_DIR))
    import importlib

    return importlib.import_module(f"roboground_ros.{dotted}")


@pytest.fixture(scope="module")
def pnode():
    """`perception_node.py`（顶层只 import sys，可安全离线导入）。"""
    return _import_pkg_module("perception_node")


@pytest.fixture(scope="module")
def tfspec():
    """`tf_spec.py`（零依赖，纯 Python）。"""
    return _import_pkg_module("tf_spec")


# ==========================================================================
# 1) 包结构：declare 的东西必须真的存在
# ==========================================================================
def test_required_ament_files_exist():
    """ament_python 包的必备文件，缺一个 `colcon build` 就失败。"""
    required = [
        "package.xml",
        "setup.py",
        "setup.cfg",
        "resource/roboground_ros",       # ← ament 资源索引标记文件
        "config/roboground.yaml",        # ← setup.py 的 data_files 引用它
        "roboground_ros/__init__.py",
        "roboground_ros/perception_node.py",
        "roboground_ros/query_node.py",
        "roboground_ros/launch_args.py",
        "roboground_ros/tf_spec.py",
        "launch/perception.launch.py",
        "launch/full.launch.py",
        "launch/tf_tree.launch.py",
    ]
    missing = [rel for rel in required if not (PKG_DIR / rel).exists()]
    assert not missing, f"ros2_ws 包缺文件（colcon build 会失败）：{missing}"


def test_setup_py_data_files_all_exist():
    """`setup.py` 的 data_files 里每个路径都必须真实存在。

    这是 colcon build 失败最常见的单一原因：`setup.py` 声明了
    `resource/<pkg>` 或 `config/x.yaml`，但文件没建出来。
    """
    text = (PKG_DIR / "setup.py").read_text(encoding="utf-8")
    refs = re.findall(r"""["']([\w./]+\.(?:py|yaml|xml)|resource/[\w]+)["']""", text)
    checked = 0
    missing = []
    for ref in refs:
        if ref in ("setup.py",):          # 自引用，跳过
            continue
        if not (PKG_DIR / ref).exists():
            missing.append(ref)
        checked += 1
    assert checked >= 5, f"从 setup.py 里只解析出 {checked} 个路径，正则可能失效"
    assert not missing, f"setup.py 的 data_files 指向不存在的文件：{missing}"
    # `resource/<pkg>` 是字符串拼接出来的（正则看不到），单独断言
    assert (PKG_DIR / "resource" / "roboground_ros").is_file(), \
        "缺 resource/roboground_ros（ament 资源索引标记），colcon build 会失败"


def test_entry_points_resolve_to_real_functions():
    """`ros2 run roboground_ros <name>` 的入口必须是模块里真实存在的 `main`。"""
    text = (PKG_DIR / "setup.py").read_text(encoding="utf-8")
    eps = re.findall(r'"(\w+)\s*=\s*roboground_ros\.(\w+):(\w+)"', text)
    assert eps, "setup.py 里没解析到 console_scripts 入口"
    for exe, module, func in eps:
        mod_path = MOD_DIR / f"{module}.py"
        assert mod_path.exists(), f"入口 {exe} 指向不存在的模块 {module}.py"
        mod = _import_pkg_module(module)
        assert callable(getattr(mod, func, None)), \
            f"入口 {exe} 指向 {module}.{func}，但该属性不存在或不可调用"


def test_package_xml_is_format_3_ament_python():
    """package.xml 必须是 format 3 且 build_type 为 ament_python（否则 colcon 不认）。"""
    xml = (PKG_DIR / "package.xml").read_text(encoding="utf-8")
    assert 'format="3"' in xml, "package.xml 不是 format 3"
    assert "<build_type>ament_python</build_type>" in xml, "build_type 不是 ament_python"
    assert "<name>roboground_ros</name>" in xml


def test_setup_cfg_script_dir_points_at_package():
    """`setup.cfg` 的 script_dir 必须是 `lib/<pkg>`，否则 ros2 run 找不到可执行文件。"""
    cfg = (PKG_DIR / "setup.cfg").read_text(encoding="utf-8")
    assert "script_dir=$base/lib/roboground_ros" in cfg.replace(" ", ""), \
        f"setup.cfg 的 script_dir 不对：\n{cfg}"


# ==========================================================================
# 2) launch 参数与 params 文件必须一致
# ==========================================================================
def _yaml_params():
    """读 `config/roboground.yaml` 的 `perception.ros__parameters`。"""
    import yaml

    data = yaml.safe_load((PKG_DIR / "config" / "roboground.yaml").read_text(encoding="utf-8"))
    return data["perception"]["ros__parameters"]


def test_yaml_keys_match_param_table(pnode):
    """YAML 里的键必须恰好是 PARAM_MAP 里可用的参数（多一个少一个都是坑）。"""
    yaml_keys = set(_yaml_params())
    param_keys = set(pnode.PARAM_MAP) - {"node_name"}      # node_name 不写进 YAML
    assert yaml_keys == param_keys, (
        f"YAML 与 PARAM_MAP 不一致：\n"
        f"  只在 YAML：{sorted(yaml_keys - param_keys)}\n"
        f"  只在 PARAM_MAP：{sorted(param_keys - yaml_keys)}"
    )


def test_launch_overrides_match_yaml(pnode):
    """`LAUNCH_OVERRIDES`（launch 命令行的默认值）必须与 YAML 里的值一致。

    launch 命令行参数优先级**高于** params 文件，两处不一致就会出现
    "改了 YAML 却没生效"。这是最容易犯又最难查的一类配置 bug。
    """
    params = _yaml_params()
    mismatches = {}
    for key, launch_value in pnode.LAUNCH_OVERRIDES.items():
        yaml_value = params.get(key)
        # 统一按字符串比（launch 参数本来就是字符串）
        if str(yaml_value).lower() != str(launch_value).lower():
            mismatches[key] = {"launch": launch_value, "yaml": yaml_value}
    assert not mismatches, (
        "launch 默认值与 config/roboground.yaml 不一致（会覆盖 YAML）：\n"
        + "\n".join(f"  {k}: launch={v['launch']!r} yaml={v['yaml']!r}"
                    for k, v in sorted(mismatches.items()))
    )


def test_yaml_values_are_shapes_the_code_can_consume(pnode):
    """YAML 的取值必须能被节点真正消费（类型/枚举/范围）。"""
    params = _yaml_params()
    assert params["pose_source"] in {"tf", "odometry", "static", "identity"}
    assert params["detector"] in {"stub", "grounding_dino", "yolo"}
    assert params["segmenter"] in {"box", "sam", "stub"}
    assert params["encoder"] in {"color_hist", "dinov2", "clip", "siglip"}
    assert float(params["sync_slop"]) > 0
    assert int(params["publish_every_n"]) >= 1
    assert 0 < float(params["min_depth"]) < float(params["max_depth"])
    assert float(params["depth_scale"]) > 0
    assert float(params["voxel_size"]) > 0
    # 这几个在校验时必须真的是 bool（YAML 里写成字符串会被 bool("false") 吃掉）
    for key in ("optical_frame_correction", "autosync_depth"):
        assert isinstance(params[key], bool), f"{key} 不是 bool：{params[key]!r}"


def test_optical_frame_correction_matches_source_frame(pnode):
    """`optical_frame_correction` 与 `pose_source_frame` 必须自洽。

    查的是 `*_optical_frame` 就必须是 False（否则**重复旋转**，实测 1.697 m 误差）；
    查 `camera_link` 才应该是 True。这一条在项目里已经错过一次，锁死它。
    """
    params = _yaml_params()
    source = str(params["pose_source_frame"])
    correction = bool(params["optical_frame_correction"])
    if source.endswith("_optical_frame"):
        assert correction is False, (
            f"pose_source_frame={source!r} 已经是光学系，"
            "optical_frame_correction 必须为 false（否则重复旋转）"
        )
    else:
        pytest.fail(
            f"pose_source_frame={source!r} 不是光学系；"
            "确认这是有意为之，并同步修正本测试的期望"
        )


# ==========================================================================
# 3) TF 树：数值必须与代码约定一致（★ 最容易静默错的一步）
# ==========================================================================
def test_optical_quaternion_equals_r_optical_to_link(tfspec):
    """`camera_link → optical` 的四元数必须等于 `R_OPTICAL_TO_LINK` 的四元数。

    方向陷阱：tf2 的一条边存的是"子系在父系中的位姿"（p_parent = R·p_child + t），
    所以这条边要填的是 optical → camera_link 的旋转。填成转置是**另一个旋转**，
    静默产生 1.697 m 误差。
    """
    from roboground.deployment.ros2.tf import R_OPTICAL_TO_LINK

    R = np.asarray(tfspec.quaternion_to_matrix(tfspec.OPTICAL_QUATERNION_XYZW))
    assert np.allclose(R, R_OPTICAL_TO_LINK, atol=1e-12), (
        "TF 边的四元数与 R_OPTICAL_TO_LINK 不一致！\n"
        f"  四元数 {tfspec.OPTICAL_QUATERNION_XYZW} 给出：\n{R}\n"
        f"  期望 R_OPTICAL_TO_LINK：\n{np.asarray(R_OPTICAL_TO_LINK)}\n"
        "  （若转置了，说明这条边的方向写反了）"
    )
    # 转置必须**不**相等，否则本测试没有区分力
    assert not np.allclose(R, np.asarray(R_OPTICAL_TO_LINK).T), \
        "R_OPTICAL_TO_LINK 与它的转置相等？测试失去意义，检查常量"


def test_tf_chain_agrees_with_static_pose_mode(tfspec):
    """★ 核心断言：走完整 TF 链得到的相机位姿，必须与 `pose_source:=static` 逐位相同。

    这是把两套**互相独立**的实现放在一起对答案：
      · 一边是 launch 文件发布、由 tf2 组合的 `map→optical` 链；
      · 另一边是 `StaticPoseProvider` 用 (translation, rpy) + 轴纠正算出来的。

    两边一致 ⇒ 四元数方向、单位、平移量、坐标系约定**同时**正确。
    历史上 static / tf / odometry 三种来源给出过逐位相同的 0.534 m 地图误差，
    正是靠这种"独立路径对答案"的方式才敢下结论。
    """
    from roboground.deployment.ros2.tf import (
        euler_to_camera_pose,
        tf_transform_to_camera_pose,
    )

    frames = tfspec.resolve_frames()
    t, q = tfspec.lookup_transform(frames["map"], frames["optical"])

    # TF 侧：链已经走到了 optical frame，所以**不再**做轴纠正
    pose_tf = tf_transform_to_camera_pose(t, q, optical_frame_correction=False)
    # static 侧：给的是 base_link 的平移 + 零旋转，所以要**做**轴纠正
    pose_static = euler_to_camera_pose((0.0, 0.0, tfspec.CAMERA_HEIGHT_M),
                                       (0.0, 0.0, 0.0),
                                       optical_frame_correction=True)

    assert np.allclose(pose_tf.R, pose_static.R, atol=1e-12), (
        "TF 链给出的旋转与 static 模式不一致：\n"
        f"  TF     :\n{pose_tf.R}\n  static :\n{pose_static.R}"
    )
    assert np.allclose(pose_tf.t, pose_static.t, atol=1e-12), (
        f"TF 链给出的平移 {pose_tf.t} 与 static 模式 {pose_static.t} 不一致")
    # 顺带：相机高度必须真的进了位姿（否则测试对"平移填 0"没有区分力）。
    # 注意检查的是**模长**而不是 t[2]：世界系 z 上的 1.2 m 在光学系里落在
    # y 轴（向下）上 —— 这正是轴纠正的体现，也是这条断言的额外价值。
    assert np.linalg.norm(pose_static.t) > 0.5, \
        "位姿平移几乎是 0，说明相机高度没被用上，测试失去意义"
    assert abs(float(pose_static.t[1])) > 0.5 and abs(float(pose_static.t[2])) < 1e-9, (
        "世界系 z 方向的 1.2 m 应当映射到光学系的 y 轴；"
        f"实测 t={pose_static.t}，说明轴纠正方向可能不对"
    )


def test_tf_tree_is_connected_and_has_expected_frames(tfspec):
    """TF 树必须连通，且包含 REP-105 的 5 个标准帧。"""
    frames = tfspec.frame_names()
    assert frames == ["map", "odom", "base_link", "camera_link",
                      "camera_color_optical_frame"], f"帧列表不符合预期：{frames}"
    # 任意两帧之间都应该查得到（树，不是森林）
    for a in frames:
        for b in frames:
            if a != b:
                tfspec.lookup_transform(a, b)      # 不连通会抛 LookupError


def test_tf_lookup_inverse_roundtrip(tfspec):
    """`lookup_transform(A,B)` 与 `(B,A)` 必须互为逆（tf2 的对称性）。"""
    frames = tfspec.resolve_frames()
    t_ab, q_ab = tfspec.lookup_transform(frames["map"], frames["optical"])
    t_ba, q_ba = tfspec.lookup_transform(frames["optical"], frames["map"])
    R_ab = np.asarray(tfspec.quaternion_to_matrix(q_ab))
    R_ba = np.asarray(tfspec.quaternion_to_matrix(q_ba))
    assert np.allclose(R_ab @ R_ba, np.eye(3), atol=1e-12)
    # p_a = R_ab p_b + t_ab 与 p_b = R_ba p_a + t_ba 必须互逆
    t_ab = np.asarray(t_ab)
    t_ba = np.asarray(t_ba)
    assert np.allclose(R_ba @ t_ab + t_ba, np.zeros(3), atol=1e-12)


def test_unknown_frame_raises_instead_of_silently_returning_identity(tfspec):
    """查不到的帧必须**抛异常**，绝不能静默返回单位阵。

    项目原则：缺位姿时宁可丢帧，也不伪造一个位姿（伪造会把不同时刻的帧
    错误地叠在一起 → 地图重影，而且完全静默）。
    """
    with pytest.raises(LookupError):
        tfspec.lookup_transform("map", "camera_link_typo")


# ==========================================================================
# 4) launch 文件本身：不能有"声明了却没用"的参数
# ==========================================================================
def test_launch_files_do_not_declare_dead_arguments():
    """每个 `DeclareLaunchArgument` 都必须在 launch 侧被真正引用。

    历史缺陷：`full.launch.py` 声明了 `use_sim_time` 却只把值拼进注释里说的
    `--ros-args`，实际没传给节点 → 回放 rosbag 时 TF 按墙上时钟查、必然失败。
    另有 `arguments=[]` 这种纯占位写法。

    引用**跨文件**统计：参数可能声明在某个 launch 文件里，却被
    `launch_args.py` 里的公共构造逻辑消费（`use_vlm` 就是这样）。
    所以把所有 launch 侧文件拼成一份文本再数出现次数。
    """
    files = sorted((PKG_DIR / "launch").glob("*.launch.py")) + [MOD_DIR / "launch_args.py"]
    combined = "\n".join(f.read_text(encoding="utf-8") for f in files)
    declared = re.findall(r'DeclareLaunchArgument\(\s*"([^"]+)"', combined)
    assert declared, "一个 launch 参数都没解析到，正则可能失效"
    dead = [n for n in declared if len(re.findall(rf'"{re.escape(n)}"', combined)) <= 1]
    assert not dead, (
        f"声明了但从未被引用的 launch 参数：{dead}\n"
        "（声明了不用 = 用户以为能调，其实没接线）"
    )
    # 顺带守一条：不要留 `arguments=[]` 这种空占位
    for f in files:
        assert "arguments=[]" not in f.read_text(encoding="utf-8"), \
            f"{f.name} 里有空的 arguments=[]，属于无效占位写法"


def test_node_parameters_are_type_annotated_or_plain_strings():
    """数值型参数必须用 `ParameterValue(..., value_type=...)` 声明类型。

    launch 的 `LaunchConfiguration` 求值结果是**字符串**；把 `"0.05"` 当 double
    参数塞进去，只是在依赖下游恰好做了 `float()` 转换。显式声明类型才是对的。
    """
    text = (MOD_DIR / "launch_args.py").read_text(encoding="utf-8")
    for key in ("sync_slop", "publish_every_n", "use_sim_time"):
        assert re.search(rf'"{key}":\s*ParameterValue\(', text), (
            f"参数 {key} 没有用 ParameterValue 声明类型 —— "
            "launch 会把它当字符串传下去"
        )
