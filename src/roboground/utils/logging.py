"""轻量日志：带级别前缀与可选耗时，不依赖任何第三方库。

为什么不用 `logging`？
本项目的日志主要给人看（跑 demo、看流水线进度），`logging` 的
handler/formatter 配置在脚本里太啰嗦。这里做一个够用且零配置的实现。
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Windows 控制台默认是 GBK，直接 print 中文会变成乱码。
# 这里在导入时把标准流切成 UTF-8（失败则静默忽略，不影响功能）。
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

_VERBOSITY = 1  # 0=静默 1=正常 2=debug

_LEVEL_TAG = {
    "debug": "[.]",
    "info": "[*]",
    "ok": "[+]",
    "warn": "[!]",
    "error": "[x]",
}

_LEVEL_MIN = {"debug": 2, "info": 1, "ok": 1, "warn": 1, "error": 0}


def set_verbosity(level: int) -> None:
    """0=静默，1=正常，2=debug。"""
    global _VERBOSITY
    _VERBOSITY = int(level)


def get_verbosity() -> int:
    return _VERBOSITY


class Logger:
    """极简 logger。所有方法都返回 None，可安全用于表达式语句。"""

    def __init__(self, name: str = "roboground") -> None:
        self.name = name
        self._timers: Dict[str, float] = {}

    # ---------------- 内部 ----------------
    def _emit(self, level: str, message: str) -> None:
        if _VERBOSITY < _LEVEL_MIN.get(level, 1):
            return
        tag = _LEVEL_TAG.get(level, "[*]")
        prefix = f"{tag} {self.name}: " if self.name else f"{tag} "
        stream = sys.stderr if level in {"warn", "error"} else sys.stdout
        print(f"{prefix}{message}", file=stream, flush=True)

    # ---------------- 公开 API ----------------
    def debug(self, message: str) -> None:
        self._emit("debug", message)

    def info(self, message: str) -> None:
        self._emit("info", message)

    def ok(self, message: str) -> None:
        self._emit("ok", message)

    def warn(self, message: str) -> None:
        self._emit("warn", message)

    def error(self, message: str) -> None:
        self._emit("error", message)

    # ---------------- 计时 ----------------
    def tick(self, key: str) -> None:
        """开始计时。"""
        self._timers[key] = time.perf_counter()

    def tock(self, key: str, message: str = "") -> float:
        """结束计时并打印耗时（秒）。返回耗时。"""
        start = self._timers.pop(key, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        suffix = f" {message}" if message else ""
        self._emit("info", f"{key} 耗时 {elapsed:.3f}s{suffix}")
        return elapsed

    def kv(self, title: str, payload: Dict[str, Any]) -> None:
        """打印一组 key-value（用于每轮实验的结果汇总）。"""
        if _VERBOSITY < 1:
            return
        self._emit("info", title)
        width = max((len(str(k)) for k in payload), default=0)
        for key, value in payload.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            print(f"      {str(key).ljust(width)} : {value}", flush=True)


_CACHE: Dict[str, Logger] = {}


def get_logger(name: str = "roboground") -> Logger:
    """按名字取 logger（同名复用，避免重复构造）。"""
    if name not in _CACHE:
        _CACHE[name] = Logger(name)
    return _CACHE[name]
