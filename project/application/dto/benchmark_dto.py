from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkRequest:
    """Request DTO para benchmark de desempenho."""

    profile: str = "medium"
    iterations: int = 1000


@dataclass(frozen=True)
class BenchmarkResponse:
    """Response DTO para benchmark de desempenho."""

    success: bool
    avg_latency_ms: float
    throughput_fps: float
