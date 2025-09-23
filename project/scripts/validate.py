from __future__ import annotations

import argparse
import logging
import sys

from application.use_cases.validate_data import ValidateDataUseCase
from application.dto.validate_data_dto import ValidateDataRequest

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Executa validação de dados brutos.

    Gera relatório de consistência e prepara dataset processado
    para etapas posteriores do pipeline.
    """
    parser = argparse.ArgumentParser(description="Validação de dados brutos")
    parser.add_argument("--input", required=True, help="Diretório com dados brutos")
    parser.add_argument("--out", required=True, help="Diretório de saída para dados processados")
    parser.add_argument("--report", required=True, help="Caminho do relatório HTML")
    args = parser.parse_args()

    if not args.input.strip() or not args.out.strip() or not args.report.strip():
        logger.error("Parâmetros inválidos: input='%s', out='%s', report='%s'", args.input, args.out, args.report)
        sys.exit(1)

    try:
        req = ValidateDataRequest(input_path=args.input, output_path=args.out, report_path=args.report)
        uc = ValidateDataUseCase()
        res = uc.execute(req)

        print(f"✅ Dados validados. Relatório salvo em {res.report_path}")
        logger.info("Validação concluída com sucesso. Relatório salvo em %s", res.report_path)
    except Exception as e:
        logger.exception("Falha crítica durante validação: %s", str(e))
        print(f"❌ Falha durante validação: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
