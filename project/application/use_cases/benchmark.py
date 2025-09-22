import time
from application.dto.benchmark_dto import BenchmarkRequest, BenchmarkResponse
from domain.services.classifier import IClassifier


class BenchmarkUseCase:
    """Caso de uso: benchmark de inferência."""

    def __init__(self, classifier: IClassifier):
        self._classifier = classifier

    def execute(self, request: BenchmarkRequest) -> BenchmarkResponse:
        dummy_input = [[0.0] * 128]  # placeholder
        latencies = []

        for _ in range(request.iterations):
            start = time.perf_counter()
            self._classifier.predict(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency_ms = sum(latencies) / len(latencies)
        throughput_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

        return BenchmarkResponse(
            success=True,
            avg_latency_ms=avg_latency_ms,
            throughput_fps=throughput_fps,
        )
