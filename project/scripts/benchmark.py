from __future__ import annotations

import argparse
from application.use_cases.benchmark import BenchmarkUseCase
from application.dto.benchmark_dto import BenchmarkRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark do pipeline")
    parser.add_argument("--profile", default="medium", help="Perfil de execução (low/medium/high)")
    parser.add_argument("--iterations", type=int, default=100, help="Número de iterações")
    args = parser.parse_args()

    req = BenchmarkRequest(profile=args.profile, iterations=args.iterations)
    uc = BenchmarkUseCase()
    res = uc.execute(req)

    print(f"✅ Benchmark concluído. Tempo médio: {res.avg_time:.3f}s")


if __name__ == "__main__":
    main()
