from __future__ import annotations

import argparse
import logging
import sys

from application.use_cases.train import TrainUseCase
from application.dto.train_dto import TrainRequest

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Executa auditoria de viés em modelos de classificação facial.

    O script treina modelos com fairness auditing habilitado e gera relatórios.
    Deve ser executado a partir da linha de comando.
    """
    parser = argparse.ArgumentParser(description="Auditoria de viés em modelos")
    parser.add_argument("--features", required=True, help="Caminho para features extraídas")
    parser.add_argument("--models-dir", required=True, help="Diretório para salvar modelos")
    parser.add_argument("--protocol", choices=["loso", "kfold"], default="loso")
    parser.add_argument("--profile", default="medium", help="Perfil de execução (low/medium/high)")
    args = parser.parse_args()

    if not args.features.strip() or not args.models_dir.strip():
        logger.error("Parâmetros inválidos: features='%s', models_dir='%s'", args.features, args.models_dir)
        sys.exit(1)

    try:
        req = TrainRequest(
            features_path=args.features,
            models_dir=args.models_dir,
            protocol=args.protocol,
            profile=args.profile,
            enable_xai=True,
            audit_fairness=True,
        )
        logger.info("Iniciando auditoria de viés com protocolo=%s, profile=%s", args.protocol, args.profile)

        uc = TrainUseCase()
        res = uc.execute(req)

        print("✅ Auditoria concluída.")
        print(f"Relatórios em {res.report_dir}")
        logger.info("Auditoria concluída com sucesso. Relatórios em %s", res.report_dir)
    except Exception as e:
        logger.exception("Falha crítica durante auditoria: %s", str(e))
        print(f"❌ Falha durante auditoria: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
