from __future__ import annotations

import argparse
from application.use_cases.validate_data import ValidateDataUseCase
from application.dto.validate_data_dto import ValidateDataRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validação de dados brutos")
    parser.add_argument("--input", required=True, help="Diretório com dados brutos")
    parser.add_argument("--out", required=True, help="Diretório de saída para dados processados")
    parser.add_argument("--report", required=True, help="Caminho do relatório HTML")
    args = parser.parse_args()

    req = ValidateDataRequest(input_path=args.input, output_path=args.out, report_path=args.report)
    uc = ValidateDataUseCase()
    res = uc.execute(req)

    print(f"✅ Dados validados. Relatório salvo em {res.report_path}")


if __name__ == "__main__":
    main()
