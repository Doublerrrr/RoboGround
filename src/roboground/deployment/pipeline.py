"""异步快慢分级流水线（Stage 4 的"实时性"部分）。

为什么要分级（这是服务机器人感知的工程核心）
------------------------------------------
一个 VLM 动辄几百毫秒到数秒，而机器人避障/跟随需要 10~30Hz。
如果所有东西串在一个循环里，要么 VLM 拖垮实时性，要么为了实时性放弃语义。

**正确做法是把它们拆到不同频率的时间轴上：**

```
                        ┌──────────────── 快档（10 Hz）────────────────┐
RGB-D 帧 ──► 感知（检测/分割/编码）──► 反投影 ──► 体素融合 ──► 更新地图
                        └───────────────────────┬────────────────────┘
                                                │ 地图快照（线程安全）
                        ┌──────────────── 慢档（1 Hz）────────────────┐
                        │  取最新地图快照 ──► 规则引擎/VLM 推理 ──► 答案 │
                        └────────────────────────────────────────────┘
```

关键设计
--------
1. **丢帧而非阻塞**：快档队列满了就丢最旧的帧，绝不让实时链路等慢档；
2. **快照式共享**：慢档读的是地图的**不可变快照**，避免读到一半被写坏；
3. **独立的延迟统计**：每档单独统计 p50/p95，便于定位瓶颈。

⚠️ 诚实边界：Python 的 GIL 让"多线程"并不能让 CPU 密集任务真正并行。
本模块的价值在于**解耦与不阻塞**（把慢任务挪出实时循环），而不是并行加速。
真要并行需要多进程（本项目为 8GB 显存单机场景，多进程会重复占显存，故未采用）。
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

from roboground.utils.logging import get_logger

logger = get_logger("deployment.pipeline")


# ==========================================================================
# 统计
# ==========================================================================
class LatencyTracker:
    """滑动窗口延迟统计（毫秒）。"""

    def __init__(self, window: int = 50, name: str = "tier") -> None:
        self.name = name
        self.window = int(window)
        self._samples: Deque[float] = deque(maxlen=self.window)
        self._count = 0
        self._lock = threading.Lock()

    def add(self, ms: float) -> None:
        with self._lock:
            self._samples.append(float(ms))
            self._count += 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def stats(self) -> Dict[str, float]:
        with self._lock:
            data = np.asarray(list(self._samples), dtype=np.float64)
        if data.size == 0:
            return {"count": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                    "min_ms": 0.0, "max_ms": 0.0}
        return {
            "count": float(self._count),
            "mean_ms": float(data.mean()),
            "p50_ms": float(np.median(data)),
            "p95_ms": float(np.percentile(data, 95)),
            "min_ms": float(data.min()),
            "max_ms": float(data.max()),
        }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._count = 0


@dataclass
class TierStats:
    """一档流水线的运行统计。"""

    name: str
    target_hz: float = 0.0
    processed: int = 0
    dropped: int = 0
    errors: int = 0
    latency: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def actual_hz(self) -> float:
        """基于延迟均值的实际吞吐（Hz）。"""
        mean_ms = self.latency.get("mean_ms", 0.0)
        return 1000.0 / mean_ms if mean_ms > 1e-6 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target_hz": self.target_hz,
            "processed": self.processed,
            "dropped": self.dropped,
            "errors": self.errors,
            "actual_hz": round(self.actual_hz, 2),
            **{k: round(v, 3) for k, v in self.latency.items()},
            **self.extra,
        }


# ==========================================================================
# 异步流水线
# ==========================================================================
class AsyncPipeline:
    """快慢两档异步流水线。

    Parameters
    ----------
    fast_fn
        快档处理函数：`fast_fn(frame) -> Any`（通常是"感知+建图"）。
    slow_fn
        慢档处理函数：`slow_fn() -> Any`（通常是"取最新地图 → 推理"）。
        不接受参数 —— 它自己从 `get_snapshot()` 拿最新地图。
    fast_hz, slow_hz
        目标频率。快档超频会**丢帧**；慢档按自己的节奏跑。
    queue_size
        快档队列长度。满了丢最旧帧（保证低延迟）。

    Examples
    --------
    >>> pipe = AsyncPipeline(fast_fn=lambda f: f, slow_fn=lambda: "ok")
    >>> pipe.start()                            # doctest: +SKIP
    >>> pipe.submit("frame")                    # doctest: +SKIP
    >>> pipe.stop()                             # doctest: +SKIP
    """

    def __init__(
        self,
        fast_fn: Callable[[Any], Any],
        slow_fn: Optional[Callable[[], Any]] = None,
        *,
        fast_hz: float = 10.0,
        slow_hz: float = 1.0,
        queue_size: int = 4,
        snapshot_fn: Optional[Callable[[], Any]] = None,
        drop_oldest: bool = True,
        name: str = "roboground",
    ) -> None:
        self.fast_fn = fast_fn
        self.slow_fn = slow_fn
        self.snapshot_fn = snapshot_fn
        self.fast_hz = float(fast_hz)
        self.slow_hz = float(slow_hz)
        self.drop_oldest = bool(drop_oldest)
        self.name = name

        self._queue: "queue.Queue" = queue.Queue(maxsize=max(int(queue_size), 1))
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()

        self.fast_latency = LatencyTracker(name="fast")
        self.slow_latency = LatencyTracker(name="slow")
        self.fast_dropped = 0
        self.fast_processed = 0
        self.slow_processed = 0
        self.errors: List[str] = []

        self._snapshot: Any = None
        self._last_answer: Any = None

    # ---------------- 生命周期 ----------------
    def start(self) -> "AsyncPipeline":
        """启动后台线程（幂等）。"""
        if self._threads:
            return self
        self._stop.clear()
        t_fast = threading.Thread(target=self._fast_loop, name=f"{self.name}-fast", daemon=True)
        self._threads.append(t_fast)
        t_fast.start()

        if self.slow_fn is not None:
            t_slow = threading.Thread(target=self._slow_loop, name=f"{self.name}-slow", daemon=True)
            self._threads.append(t_slow)
            t_slow.start()

        logger.info(
            f"流水线已启动：快档 {self.fast_hz}Hz + "
            f"{'慢档 ' + str(self.slow_hz) + 'Hz' if self.slow_fn else '无慢档'}"
        )
        return self

    def stop(self, *, timeout: float = 2.0) -> None:
        """停止后台线程（幂等）。"""
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        logger.info("流水线已停止")

    def __enter__(self) -> "AsyncPipeline":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---------------- 数据入口 ----------------
    def submit(self, frame: Any) -> bool:
        """提交一帧。返回 True 表示入队成功，False 表示因队列满被丢弃。"""
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            if self.drop_oldest:
                try:
                    self._queue.get_nowait()      # 丢最旧的
                    self.fast_dropped += 1
                    self._queue.put_nowait(frame)
                    return True
                except queue.Empty:
                    pass
            self.fast_dropped += 1
            return False

    # ---------------- 状态读取 ----------------
    def get_snapshot(self) -> Any:
        """线程安全地读取最新的地图快照。"""
        with self._lock:
            return self._snapshot

    def set_snapshot(self, snapshot: Any) -> None:
        with self._lock:
            self._snapshot = snapshot

    @property
    def last_answer(self) -> Any:
        return self._last_answer

    # ---------------- 内部循环 ----------------
    def _fast_loop(self) -> None:
        period = 1.0 / self.fast_hz if self.fast_hz > 0 else 0.0
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                result = self.fast_fn(frame)
                self.fast_processed += 1
                if self.snapshot_fn is not None:
                    self.set_snapshot(self.snapshot_fn())
                elif result is not None:
                    self.set_snapshot(result)
            except Exception as exc:
                self.errors.append(f"fast: {exc}")
                logger.warn(f"快档处理失败：{exc}")
            finally:
                self.fast_latency.add((time.perf_counter() - t0) * 1000.0)

            # 节流到目标频率
            if period > 0:
                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    self._stop.wait(period - elapsed)

    def _slow_loop(self) -> None:
        period = 1.0 / self.slow_hz if self.slow_hz > 0 else 1.0
        while not self._stop.is_set():
            started = time.perf_counter()
            t0 = time.perf_counter()
            try:
                if self.slow_fn is not None:
                    self._last_answer = self.slow_fn()
                    self.slow_processed += 1
            except Exception as exc:
                self.errors.append(f"slow: {exc}")
                logger.warn(f"慢档处理失败：{exc}")
            finally:
                self.slow_latency.add((time.perf_counter() - t0) * 1000.0)

            elapsed = time.perf_counter() - started
            if elapsed < period:
                self._stop.wait(period - elapsed)

    # ---------------- 统计 ----------------
    def stats(self) -> Dict[str, Any]:
        return {
            "fast": TierStats(
                name="fast", target_hz=self.fast_hz,
                processed=self.fast_processed, dropped=self.fast_dropped,
                latency=self.fast_latency.stats(),
                extra={"queue_size": self._queue.qsize()},
            ).to_dict(),
            "slow": TierStats(
                name="slow", target_hz=self.slow_hz,
                processed=self.slow_processed,
                latency=self.slow_latency.stats(),
            ).to_dict(),
            "errors": len(self.errors),
            "running": bool(self._threads),
        }

    def describe(self) -> str:
        s = self.stats()
        f, sl = s["fast"], s["slow"]
        return (
            f"AsyncPipeline(fast: {f['processed']} 帧, 丢弃 {f['dropped']}, "
            f"p50={f['p50_ms']:.1f}ms p95={f['p95_ms']:.1f}ms | "
            f"slow: {sl['processed']} 次, p50={sl['p50_ms']:.1f}ms | "
            f"错误 {s['errors']})"
        )

    def __repr__(self) -> str:
        return f"AsyncPipeline(fast_hz={self.fast_hz}, slow_hz={self.slow_hz}, running={bool(self._threads)})"
