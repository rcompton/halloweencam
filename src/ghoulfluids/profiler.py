from __future__ import annotations
import time
from collections import deque
from contextlib import contextmanager

from .logging import get_logger

_profiler = None


def get_profiler():
    global _profiler
    if _profiler is None:
        _profiler = Profiler()
    return _profiler


class Profiler:
    def __init__(self, ema_alpha=0.1, maxlen=100):
        self._timings = {}
        self._w_stats = {}
        self.ema_alpha = ema_alpha
        self.maxlen = maxlen
        self.logger = get_logger("Profiler")

    @contextmanager
    def record(self, name: str):
        start_t = time.perf_counter()
        try:
            yield
        finally:
            end_t = time.perf_counter()
            dt = end_t - start_t

            if name not in self._timings:
                self._timings[name] = deque(maxlen=self.maxlen)
            self._timings[name].append(dt)

            if name not in self._w_stats:
                self._w_stats[name] = dt
            else:
                self._w_stats[name] = (
                    self.ema_alpha * dt + (1.0 - self.ema_alpha) * self._w_stats[name]
                )

    def get_timings(self):
        return self._w_stats.copy()

    def log_stats(self):
        stats = []
        for k, v in sorted(self._w_stats.items()):
            stats.append(f"{k}: {v*1000:.2f}ms")
        self.logger.info(" | ".join(stats))
