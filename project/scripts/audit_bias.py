from __future__ import annotations

import argparse
from application.use_cases.train import TrainUseCase
from application.dto.train_dto import TrainRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria de viés em modelos")
    parser.add_argument("--features", required=True, help="Caminho para features extraídas")
    parser.add_argument("--models-dir", required=True, help="Diretório para salvar modelos")
    parser.add_argument("--protocol", choices=["loso", "kfold"], default="loso")
    parser.add_argument("--profile", default="medium", help="Perfil de execução (low/medium/high)")
    args = parser.parse_args()

    req = TrainRequest(
        features_path=args.features,
        models_dir=args.models_dir,
        protocol=args.protocol,
        profile=args.profile,
        enable_xai=True,
        audit_fairness=True,
    )
    uc = TrainUseCase()
    res = uc.execute(req)

    print("✅ Auditoria concluída.")
    print(f"Relatórios em {res.report_dir}")


if __name__ == "__main__":
    main()
