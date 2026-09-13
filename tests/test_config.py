"""配置系统测试。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from roboground.config import (
    DEFAULT_CONFIG,
    Config,
    load_config,
    parse_scalar,
    resolve_device,
)


def test_default_config_loads_without_file():
    cfg = load_config()
    assert cfg.project.name == "RoboGround"
    assert cfg.geometry.voxel_size > 0
    assert cfg.perception.detector == "stub"


def test_dotted_get_and_set():
    cfg = load_config()
    assert cfg.get("geometry.voxel_size") == cfg.geometry.voxel_size
    cfg.set("geometry.voxel_size", 0.02)
    assert cfg.geometry.voxel_size == 0.02
    assert cfg.get("a.b.c.d", "fallback") == "fallback"


def test_missing_key_raises_attribute_error():
    cfg = load_config()
    with pytest.raises(AttributeError):
        _ = cfg.this_key_does_not_exist


def test_require_raises_keyerror():
    cfg = load_config()
    with pytest.raises(KeyError):
        cfg.require("nope.not.here")


def test_merge_is_deep_and_does_not_mutate_source():
    base = Config({"a": {"b": 1, "c": 2}})
    merged = base.merge({"a": {"b": 9}})
    assert merged.a.b == 9
    assert merged.a.c == 2           # 未被覆盖的兄弟键保留
    assert base.a.b == 1             # 原对象未被修改


def test_list_is_replaced_not_merged():
    """列表应整体替换（语义是"完整替换"而非逐元素合并）。"""
    base = Config({"x": [1, 2, 3]})
    merged = base.merge({"x": [9]})
    assert merged.x == [9]


def test_apply_overrides_parses_types():
    cfg = load_config()
    cfg.apply_overrides([
        "a.int=3",
        "a.float=0.5",
        "a.bool=true",
        "a.none=null",
        "a.list=[1,2]",
        "a.str=hello",
    ])
    assert cfg.get("a.int") == 3
    assert cfg.get("a.float") == 0.5
    assert cfg.get("a.bool") is True
    assert cfg.get("a.none") is None
    assert cfg.get("a.list") == [1, 2]
    assert cfg.get("a.str") == "hello"


def test_override_requires_key_value_form():
    cfg = load_config()
    with pytest.raises(ValueError):
        cfg.apply_overrides(["no_equals_sign"])


def test_env_override():
    os.environ["ROBOGROUND_DATA__ROOT"] = "/tmp/roboground_data"
    try:
        cfg = load_config()
        assert cfg.get("data.root") == "/tmp/roboground_data"
    finally:
        os.environ.pop("ROBOGROUND_DATA__ROOT", None)


def test_yaml_roundtrip(tmp_path: Path):
    cfg = load_config()
    cfg.set("geometry.voxel_size", 0.033)
    path = cfg.save(tmp_path / "cfg.yaml")
    assert path.exists()

    reloaded = load_config(path)
    assert abs(reloaded.geometry.voxel_size - 0.033) < 1e-9


def test_partial_yaml_inherits_defaults(tmp_path: Path):
    p = tmp_path / "partial.yaml"
    p.write_text("geometry:\n  voxel_size: 0.01\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.geometry.voxel_size == 0.01
    # 未在 yaml 中出现的键仍然来自默认配置
    assert cfg.geometry.max_depth == DEFAULT_CONFIG["geometry"]["max_depth"]


def test_to_dict_is_deep_copy():
    cfg = load_config()
    d = cfg.to_dict()
    d["geometry"]["voxel_size"] = 999
    assert cfg.geometry.voxel_size != 999


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("False", False), ("null", None), ("42", 42),
    ("3.14", 3.14), ("[1,2]", [1, 2]), ("hello", "hello"),
])
def test_parse_scalar(raw, expected):
    assert parse_scalar(raw) == expected


def test_resolve_device_respects_force_cpu():
    os.environ["ROBOGROUND_FORCE_CPU"] = "1"
    try:
        assert resolve_device("cuda") == "cpu"
    finally:
        os.environ.pop("ROBOGROUND_FORCE_CPU", None)


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"
