from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimingInfo:
    """
    Estrutura com informações de tempo e memória.

    Attributes:
        elapsed_ms (float): Tempo decorrido em milissegundos.
        peak_mem_kb (Optional[int]): Memória de pico em KB (se medido).
    """
    elapsed_ms: float
    peak_mem_kb: Optional[int] = None


class Timer:
    """Context manager para medir tempo e memória (tracemalloc)."""

    def __init__(self, measure_memory: bool = True) -> None:
        if not isinstance(measure_memory, bool):
            logger.error("Parâmetro inválido para Timer.measure_memory: %s", measure_memory)
            raise ValueError("measure_memory deve ser booleano.")
        self.measure_memory = measure_memory
        self._start: float = 0.0
        self._elapsed_ms: float = 0.0
        self._peak_kb: Optional[int] = None

    def __enter__(self) -> "Timer":
        if self.measure_memory:
            tracemalloc.start()
            logger.debug("Medição de memória iniciada.")
        self._start = time.perf_counter()
        logger.debug("Timer iniciado.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end = time.perf_counter()
        self._elapsed_ms = (end - self._start) * 1000.0
        if self.measure_memory:
            try:
                _, peak = tracemalloc.get_traced_memory()
                self._peak_kb = peak // 1024
                logger.debug("Pico de memória registrado: %d KB", self._peak_kb)
            finally:
                tracemalloc.stop()
                logger.debug("Medição de memória encerrada.")
        logger.debug("Timer finalizado. Tempo decorrido: %.3f ms", self._elapsed_ms)

    @property
    def info(self) -> TimingInfo:
        """Retorna informações medidas (tempo e memória)."""
        return TimingInfo(elapsed_ms=self._elapsed_ms, peak_mem_kb=self._peak_kb)


def measure_time(func: Callable[..., Any]) -> Callable[..., Dict[str, Any]]:
    """
    Decorator que mede tempo de execução de uma função.

    Args:
        func (Callable): Função alvo.

    Returns:
        Callable: Função decorada que retorna dict com resultado e tempo.

    Observação:
        Em caso de erro, a exceção é relançada após logar o tempo até a falha.
    """

    def wrapper(*args, **kwargs):
        with Timer(measure_memory=False) as t:
            try:
                result = func(*args, **kwargs)
                logger.info("Função %s executada em %.3f ms", func.__name__, t.info.elapsed_ms)
                return {"result": result, "elapsed_ms": t.info.elapsed_ms}
            except Exception as e:
                logger.exception("Erro na função %s após %.3f ms: %s", func.__name__, t.info.elapsed_ms, str(e))
                raise

    return wrapper
