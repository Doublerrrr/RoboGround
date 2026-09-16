# -*- coding: utf-8 -*-
"""编码器的**维度自检**：报出去的维度必须等于真正吐出来的维度。

为什么单独测这一条
================
这不是"边界情况"，而是**本项目真实踩过的坑**：
`perception.encoder_kwargs.feature_dim` 的历史默认是 **256**
（当年为了"各后端统一投影维度"），而 SigLIP-base 真实输出 **768** 维。
两边不一致时：

    体素场按 256 维建 → 768 维特征塞不进去 → **整张地图的特征全是 0**
    → 下游只是"查不到东西"，看起来像"模型效果不好"，实际是**根本没接上**

`scripts/44_eval_feature_field.py` 第一次跑出来"两条臂指标一模一样"，
根因就是这个（两条臂的特征场都是空的）。

修法有两道：
1. 构造时从**本地权重目录的 `config.json`** 读出真实维度并校正声明维度；
2. `encode_image` 里做维度自检，不符就**当场报错**，绝不让空特征场流到下游。

这两条都不需要真的加载模型，所以可以放进默认测试集。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from roboground.perception.encoders.siglip_encoder import (
    SigLIPEncoder,
    _dim_from_model_config,
)


# ==========================================================================
# 1) 从 config.json 读真实维度
# ==========================================================================
def _fake_model_dir(tmp_path, cfg: dict, name: str = "fake-siglip"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def test_dim_read_from_text_config(tmp_path):
    d = _fake_model_dir(tmp_path, {"text_config": {"hidden_size": 768},
                                   "vision_config": {"hidden_size": 768}})
    assert _dim_from_model_config(str(d)) == 768


def test_dim_falls_back_to_other_keys(tmp_path):
    assert _dim_from_model_config(str(_fake_model_dir(
        tmp_path / "a", {"projection_dim": 512}))) == 512
    assert _dim_from_model_config(str(_fake_model_dir(
        tmp_path / "b", {"hidden_size": 640}))) == 640

def test_dim_is_none_when_unavailable(tmp_path):
    """HF 仓库名（本地没有目录）、坏 JSON、缺字段 → `None`（保留原值，不猜）。"""
    assert _dim_from_model_config("google/siglip-base-patch16-224") is None
    assert _dim_from_model_config(str(tmp_path / "nope")) is None

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text("{ not json", encoding="utf-8")
    assert _dim_from_model_config(str(bad)) is None

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text(json.dumps({"model_type": "siglip"}),
                                       encoding="utf-8")
    assert _dim_from_model_config(str(empty)) is None


def test_encoder_overrides_the_configured_dim(tmp_path):
    """配置写 256、模型真实是 768 → 声明维度必须变成 768（并告警）。"""
    d = _fake_model_dir(tmp_path, {"text_config": {"hidden_size": 768}})
    enc = SigLIPEncoder(model_id=str(d), feature_dim=256)
    assert enc.feature_dim == 768

    # 一致时保持原值
    d2 = _fake_model_dir(tmp_path / "b", {"text_config": {"hidden_size": 256}})
    assert SigLIPEncoder(model_id=str(d2), feature_dim=256).feature_dim == 256

    # 读不到 config（HF 名）时保留配置值，不做任何猜测
    assert SigLIPEncoder(model_id="some/hf-repo", feature_dim=256).feature_dim == 256


# ==========================================================================
# 2) encode_image 的维度自检（用替身，不加载模型）
# ==========================================================================
class _StubProcessor:
    def __call__(self, **_kw):
        return {}


class _StubModel:
    """固定吐 `out_dim` 维的替身模型（只实现 `get_image_features`）。"""

    def __init__(self, out_dim: int) -> None:
        self.out_dim = int(out_dim)

    def get_image_features(self, **_kw):
        import torch  # noqa: PLC0415

        return torch.ones((1, self.out_dim), dtype=torch.float32)


def test_encode_image_raises_on_dim_mismatch(tmp_path):
    """声明 8 维、模型吐 16 维 → 必须当场报错，而不是返回错维向量。"""
    torch = pytest.importorskip("torch")
    del torch
    d = _fake_model_dir(tmp_path, {"text_config": {"hidden_size": 8}})
    enc = SigLIPEncoder(model_id=str(d), feature_dim=8)
    enc._ensure_loaded = lambda: None                     # noqa: SLF001
    enc._processor = _StubProcessor()                     # noqa: SLF001
    enc._model = _StubModel(16)                           # noqa: SLF001
    enc._device = "cpu"                                   # noqa: SLF001
    with pytest.raises(RuntimeError, match="维度不符"):
        enc.encode_image(np.zeros((16, 16, 3), dtype=np.uint8))


def test_encode_image_returns_normalised_vector_when_dims_agree(tmp_path):
    torch = pytest.importorskip("torch")
    del torch
    d = _fake_model_dir(tmp_path, {"text_config": {"hidden_size": 8}})
    enc = SigLIPEncoder(model_id=str(d), feature_dim=8)
    enc._ensure_loaded = lambda: None                     # noqa: SLF001
    enc._processor = _StubProcessor()                     # noqa: SLF001
    enc._model = _StubModel(8)                            # noqa: SLF001
    enc._device = "cpu"                                   # noqa: SLF001
    v = enc.encode_image(np.zeros((16, 16, 3), dtype=np.uint8))
    assert v.shape == (8,)
    assert float(np.linalg.norm(v)) == pytest.approx(1.0)
