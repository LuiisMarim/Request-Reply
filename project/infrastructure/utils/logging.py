from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class TimingInfo:
    elapsed_ms: float
    peak_mem_kb: Optional[int] = None


class Timer:
    """Context manager para medir tempo e memória (tracemalloc)."""

    def __init__(self, measure_memory: bool = True) -> None:
        self.measure_memory = measure_memory
        self._start: float = 0.0
        self._elapsed_ms: float = 0.0
        self._peak_kb: Optional[int] = None

    def __enter__(self) -> "Timer":
        if self.measure_memory:
            tracemalloc.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end = time.perf_counter()
        self._elapsed_ms = (end - self._start) * 1000.0
        if self.measure_memory:
            _, peak = tracemalloc.get_traced_memory()
            self._peak_kb = peak // 1024
            tracemalloc.stop()

    @property
    def info(self) -> TimingInfo:
        return TimingInfo(elapsed_ms=self._elapsed_ms, peak_mem_kb=self._peak_kb)


def measure_time(func: Callable[..., object]) -> Callable[..., Dict[str, float]]:
    """Decorator simples que retorna dict com métricas de tempo."""

    def wrapper(*args, **kwargs):
        with Timer(measure_memory=False) as t:
            result = func(*args, **kwargs)
        return {"result": result, "elapsed_ms": t.info.elapsed_ms}

    return wrapper
