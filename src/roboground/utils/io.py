"""文件 IO 与计时工具。"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """创建目录（含父目录）并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: PathLike, indent: int = 2) -> Path:
    """保存 JSON（自动建目录、支持中文、把 numpy 类型转成原生类型）。"""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(obj), fh, indent=indent, ensure_ascii=False)
    return p


def load_json(path: PathLike, default: Any = None) -> Any:
    """读取 JSON；文件不存在时返回 default（不抛异常）。"""
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_npz(path: PathLike, **arrays: Any) -> Path:
    """保存 npz；自动把非 ndarray 转成 ndarray。"""
    p = Path(path)
    ensure_dir(p.parent)
    payload = {
        key: (val if isinstance(val, np.ndarray) else np.asarray(val))
        for key, val in arrays.items()
    }
    np.savez_compressed(p, **payload)
    return p


def load_npz(path: PathLike) -> dict:
    """读取 npz 为普通 dict（而不是 NpzFile，避免文件句柄泄漏）。"""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _to_jsonable(obj: Any) -> Any:
    """递归把 numpy / Path / dataclass 转成可 JSON 序列化对象。"""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _to_jsonable(obj.to_dict())
    return obj


class Timer:
    """上下文管理器计时。

    >>> with Timer() as t:
    ...     do_something()
    >>> t.elapsed
    0.123
    """

    def __init__(self, name: str = "timer", *, verbose: bool = False) -> None:
        self.name = name
        self.verbose = verbose
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.verbose:
            print(f"[timer] {self.name}: {self.elapsed:.4f}s", flush=True)


@contextmanager
def timed(name: str = "block", sink: Optional[dict] = None) -> Iterator[dict]:
    """把耗时写进 dict 的上下文管理器（用于批量测延迟）。

    >>> stats = {}
    >>> with timed("infer", stats):
    ...     model(x)
    >>> stats["infer"] > 0
    True
    """
    start = time.perf_counter()
    box: dict = {}
    try:
        yield box
    finally:
        elapsed = time.perf_counter() - start
        box["elapsed"] = elapsed
        if sink is not None:
            sink[name] = elapsed


def human_size(num_bytes: float) -> str:
    """字节数 → 人类可读字符串。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"
