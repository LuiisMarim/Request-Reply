from __future__ import annotations

import argparse
import logging
import sys

from application.use_cases.benchmark import BenchmarkUseCase
from application.dto.benchmark_dto import BenchmarkRequest

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Executa benchmark de desempenho do pipeline.

    Mede tempo médio de execução em múltiplas iterações,
    usando perfil configurado (low/medium/high).
    """
    parser = argparse.ArgumentParser(description="Benchmark do pipeline")
    parser.add_argument("--profile", default="medium", help="Perfil de execução (low/medium/high)")
    parser.add_argument("--iterations", type=int, default=100, help="Número de iterações")
    args = parser.parse_args()

    if not args.profile.strip():
        logger.error("Profile inválido: '%s'", args.profile)
        sys.exit(1)
    if args.iterations <= 0:
        logger.error("Número de iterações inválido: %d", args.iterations)
        sys.exit(1)

    try:
        req = BenchmarkRequest(profile=args.profile, iterations=args.iterations)
        logger.info("Iniciando benchmark (profile=%s, iterations=%d)", args.profile, args.iterations)

        uc = BenchmarkUseCase()
        res = uc.execute(req)

        print(f"✅ Benchmark concluído. Tempo médio: {res.avg_time:.3f}s")
        logger.info("Benchmark concluído. Tempo médio: %.3fs", res.avg_time)
    except Exception as e:
        logger.exception("Falha crítica durante benchmark: %s", str(e))
        print(f"❌ Falha durante benchmark: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
