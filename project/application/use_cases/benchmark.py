import time
import logging
import numpy as np
from application.dto.benchmark_dto import BenchmarkRequest, BenchmarkResponse
from domain.services.classifier import IClassifier

logger = logging.getLogger(__name__)


class BenchmarkUseCase:
    """
    Caso de uso: benchmark de inferência.

    Este caso de uso executa múltiplas iterações de inferência usando o classificador
    fornecido, calcula a latência média e o throughput em frames por segundo (FPS).

    Attributes:
        _classifier (IClassifier): Implementação concreta do classificador a ser utilizado.
    """

    def __init__(self, classifier: IClassifier):
        if not isinstance(classifier, IClassifier):
            logger.error("Classificador inválido fornecido: %s", type(classifier))
            raise ValueError("O parâmetro 'classifier' deve implementar IClassifier.")
        self._classifier = classifier

    def execute(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """
        Executa o benchmark de inferência.

        Args:
            request (BenchmarkRequest): DTO contendo parâmetros do benchmark
                (número de iterações, perfil de execução).

        Returns:
            BenchmarkResponse: Resultado do benchmark, incluindo sucesso, latência média e throughput.
        """
        if not isinstance(request, BenchmarkRequest):
            logger.error("Tipo inválido para request: %s", type(request))
            raise ValueError("O parâmetro 'request' deve ser uma instância de BenchmarkRequest.")

        if request.iterations <= 0:
            logger.error("Número de iterações inválido: %s", request.iterations)
            raise ValueError("O número de iterações deve ser positivo.")

        logger.info("Iniciando benchmark com %d iterações e perfil '%s'.", request.iterations, request.profile)

        try:
            # Geração de entrada dummy simulada (exemplo: vetor de 128 dimensões)
            dummy_input = np.random.rand(1, 128).tolist()
            latencies = []

            for i in range(request.iterations):
                start = time.perf_counter()
                self._classifier.predict(dummy_input)
                end = time.perf_counter()
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)

                if (i + 1) % max(1, request.iterations // 10) == 0:
                    logger.debug("Iteração %d concluída - latência %.3f ms", i + 1, latency_ms)

            avg_latency_ms = sum(latencies) / len(latencies)
            throughput_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

            logger.info(
                "Benchmark concluído: latência média = %.3f ms, throughput = %.2f FPS",
                avg_latency_ms,
                throughput_fps,
            )

            return BenchmarkResponse(
                success=True,
                avg_latency_ms=avg_latency_ms,
                throughput_fps=throughput_fps,
            )

        except Exception as e:
            logger.exception("Erro durante execução do benchmark: %s", str(e))
            return BenchmarkResponse(success=False, avg_latency_ms=0.0, throughput_fps=0.0)
