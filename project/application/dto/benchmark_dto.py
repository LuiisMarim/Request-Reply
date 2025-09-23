from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkRequest:
    """
    Request DTO para benchmark de desempenho.

    Attributes:
        profile (str): Perfil de execução a ser utilizado (ex.: 'medium', 'fast', 'accurate').
        iterations (int): Número de iterações do benchmark. Deve ser um valor positivo.
    """

    profile: str = "medium"
    iterations: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or not self.profile.strip():
            logger.error("Perfil inválido: %s", self.profile)
            raise ValueError("O campo 'profile' deve ser uma string não vazia.")

        if not isinstance(self.iterations, int) or self.iterations <= 0:
            logger.error("Número de iterações inválido: %s", self.iterations)
            raise ValueError("O campo 'iterations' deve ser um inteiro positivo.")

        logger.info("BenchmarkRequest inicializado com sucesso: %s", self)


@dataclass(frozen=True)
class BenchmarkResponse:
    """
    Response DTO para benchmark de desempenho.

    Attributes:
        success (bool): Indica se o benchmark foi concluído com sucesso.
        avg_latency_ms (float): Latência média em milissegundos.
        throughput_fps (float): Taxa de processamento em frames por segundo.
    """

    success: bool
    avg_latency_ms: float
    throughput_fps: float

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            logger.error("Valor inválido para success: %s", self.success)
            raise ValueError("O campo 'success' deve ser um booleano.")

        if not isinstance(self.avg_latency_ms, (int, float)) or self.avg_latency_ms < 0:
            logger.error("Latência média inválida: %s", self.avg_latency_ms)
            raise ValueError("O campo 'avg_latency_ms' deve ser um número não negativo.")

        if not isinstance(self.throughput_fps, (int, float)) or self.throughput_fps < 0:
            logger.error("Throughput inválido: %s", self.throughput_fps)
            raise ValueError("O campo 'throughput_fps' deve ser um número não negativo.")

        logger.info("BenchmarkResponse inicializado com sucesso: %s", self)
