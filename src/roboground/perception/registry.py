"""后端注册表：按配置名构建 检测器 / 分割器 / 编码器。

为什么要注册表？
因为"真实模型"和"离线降级实现"必须能互换，而交换逻辑不应该散落在
业务代码里。所有后端通过 `@register_*` 声明名字，`build_*` 按名构造，
并且**构造失败会自动降级**（比如 transformers 没装 → 退回 stub），
保证流水线永远能跑起来。

Examples
--------
>>> from roboground import load_config
>>> from roboground.perception import build_detector
>>> det = build_detector(load_config())      # 默认 stub
>>> det.name
'stub'
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from roboground.perception.base import Detector, Encoder, Segmenter, PerceptionPipeline
from roboground.utils.logging import get_logger

logger = get_logger("perception.registry")

# 注册表：name -> class
_DETECTORS: Dict[str, Type[Detector]] = {}
_SEGMENTERS: Dict[str, Type[Segmenter]] = {}
_ENCODERS: Dict[str, Type[Encoder]] = {}


# ==========================================================================
# 注册装饰器
# ==========================================================================
def register_detector(*names: str) -> Callable[[Type[Detector]], Type[Detector]]:
    def deco(cls: Type[Detector]) -> Type[Detector]:
        for name in names:
            _DETECTORS[name.lower()] = cls
        return cls
    return deco


def register_segmenter(*names: str) -> Callable[[Type[Segmenter]], Type[Segmenter]]:
    def deco(cls: Type[Segmenter]) -> Type[Segmenter]:
        for name in names:
            _SEGMENTERS[name.lower()] = cls
        return cls
    return deco


def register_encoder(*names: str) -> Callable[[Type[Encoder]], Type[Encoder]]:
    def deco(cls: Type[Encoder]) -> Type[Encoder]:
        for name in names:
            _ENCODERS[name.lower()] = cls
        return cls
    return deco


# ==========================================================================
# 查询
# ==========================================================================
def available_detectors() -> Dict[str, str]:
    return {k: v.__name__ for k, v in sorted(_DETECTORS.items())}


def available_segmenters() -> Dict[str, str]:
    return {k: v.__name__ for k, v in sorted(_SEGMENTERS.items())}


def available_encoders() -> Dict[str, str]:
    return {k: v.__name__ for k, v in sorted(_ENCODERS.items())}


# ==========================================================================
# 构建（带自动降级）
# ==========================================================================
def _instantiate(cls: Type, kwargs: Dict[str, Any], fallback_cls: Optional[Type] = None,
                 kind: str = "backend", name: str = "") -> Any:
    """构造后端；失败时尝试降级实现。"""
    try:
        return cls(**kwargs)
    except Exception as exc:
        if fallback_cls is None:
            raise
        logger.warn(
            f"{kind} {name!r} 构造失败（{type(exc).__name__}: {exc}），"
            f"自动降级到 {fallback_cls.__name__}。"
            "如需真实模型请先安装可选依赖（见 setup_env.md）"
        )
        return fallback_cls(**kwargs)


def build_detector(cfg, *, name: Optional[str] = None) -> Detector:
    """按配置构建检测器。

    `cfg.perception.detector` 取值：`stub` | `grounding_dino` | `yolo`。
    其它值会退回 `stub` 并给出告警（而不是崩溃）。
    """
    from roboground.perception import detectors as _detectors  # noqa: F401  触发注册

    key = (name or cfg.get("perception.detector", "stub")).lower()
    kwargs = dict(cfg.get("perception.detector_kwargs", {}) or {})
    prompts = list(cfg.get("perception.prompts", []) or [])
    kwargs.setdefault("prompts", prompts)

    cls = _DETECTORS.get(key)
    if cls is None:
        logger.warn(
            f"未知的检测器 {key!r}；可选：{sorted(_DETECTORS)}。退回 'stub'"
        )
        cls = _DETECTORS["stub"]

    fallback = _DETECTORS["stub"]
    return _instantiate(cls, kwargs, fallback_cls=(None if cls is fallback else fallback),
                        kind="detector", name=key)


def build_segmenter(cfg, *, name: Optional[str] = None) -> Optional[Segmenter]:
    """按配置构建分割器。`none` / `null` 返回 None（不做分割）。"""
    from roboground.perception import segmenters as _segmenters  # noqa: F401

    key = (name or cfg.get("perception.segmenter", "box"))
    if key is None or str(key).lower() in {"none", "null", "false"}:
        return None
    key = str(key).lower()

    kwargs = dict(cfg.get("perception.segmenter_kwargs", {}) or {})
    cls = _SEGMENTERS.get(key)
    if cls is None:
        logger.warn(f"未知的分割器 {key!r}；可选：{sorted(_SEGMENTERS)}。退回 'box'")
        cls = _SEGMENTERS["box"]

    fallback = _SEGMENTERS["box"]
    return _instantiate(cls, kwargs, fallback_cls=(None if cls is fallback else fallback),
                        kind="segmenter", name=key)


def build_encoder(cfg, *, name: Optional[str] = None) -> Optional[Encoder]:
    """按配置构建特征编码器。"""
    from roboground.perception import encoders as _encoders  # noqa: F401

    key = (name or cfg.get("perception.encoder", "color_hist"))
    if key is None or str(key).lower() in {"none", "null", "false"}:
        return None
    key = str(key).lower()

    kwargs = dict(cfg.get("perception.encoder_kwargs", {}) or {})
    cls = _ENCODERS.get(key)
    if cls is None:
        logger.warn(f"未知的编码器 {key!r}；可选：{sorted(_ENCODERS)}。退回 'color_hist'")
        cls = _ENCODERS["color_hist"]

    fallback = _ENCODERS["color_hist"]
    return _instantiate(cls, kwargs, fallback_cls=(None if cls is fallback else fallback),
                        kind="encoder", name=key)


def build_pipeline(cfg, *, prompts: Optional[Any] = None) -> PerceptionPipeline:
    """一步构建完整感知流水线（检测 + 分割 + 编码）。"""
    detector = build_detector(cfg)
    segmenter = build_segmenter(cfg)
    encoder = build_encoder(cfg)
    use_prompts = list(prompts) if prompts is not None else list(cfg.get("perception.prompts", []) or [])
    return PerceptionPipeline(detector, segmenter, encoder, prompts=use_prompts)
