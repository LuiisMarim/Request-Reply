import os
import logging
from typing import Protocol, runtime_checkable 
from application.dto.validate_data_dto import ValidateDataRequest, ValidateDataResponse
from domain.services.repository import IDataRepository

logger = logging.getLogger(__name__)

@runtime_checkable
class IDataValidator(Protocol):
    """Protocolo para validador de dados (infrastructure.data.validators)."""

    def validate(self, input_dir: str, output_dir: str, report_path: str) -> bool:
        """
        Executa a validação de dados.

        Args:
            input_dir (str): Diretório de entrada contendo os dados a validar.
            output_dir (str): Diretório de saída para resultados da validação.
            report_path (str): Caminho completo do relatório a ser gerado.

        Returns:
            bool: True se a validação foi concluída com sucesso, False caso contrário.
        """
        ...


class ValidateDataUseCase:
    """
    Caso de uso: validação e preparação de dados.

    Orquestra a validação de dados utilizando um repositório de dados e um validador.
    """

    def __init__(self, repository: IDataRepository = None, validator: IDataValidator = None):
        if repository is not None and not isinstance(repository, IDataRepository):
            logger.error("Repositório inválido fornecido: %s", type(repository))
            raise ValueError("O parâmetro 'repository' deve implementar IDataRepository.")

        if validator is not None and not isinstance(validator, IDataValidator):
            logger.error("Validador inválido fornecido: %s", type(validator))
            raise ValueError("O parâmetro 'validator' deve implementar IDataValidator.")

        self._repository = repository
        self._validator = validator

    def execute(self, request: ValidateDataRequest) -> ValidateDataResponse:
        """
        Executa a validação e preparação de dados.

        Args:
            request (ValidateDataRequest): DTO contendo diretórios de entrada, saída e caminho do relatório.

        Returns:
            ValidateDataResponse: DTO contendo o status da validação e caminho do relatório.
        """
        if not isinstance(request, ValidateDataRequest):
            logger.error("Tipo inválido para request: %s", type(request))
            raise ValueError("O parâmetro 'request' deve ser uma instância de ValidateDataRequest.")

        if not os.path.isdir(request.input_dir):
            logger.error("Diretório de entrada inválido: %s", request.input_dir)
            raise ValueError(f"Diretório de entrada inválido: {request.input_dir}")

        if not request.output_dir:
            logger.error("Diretório de saída não fornecido.")
            raise ValueError("Diretório de saída não pode ser vazio.")

        report_path = request.report_path or os.path.join(request.output_dir, "data_quality.html")

        logger.info(
            "Iniciando validação de dados. Entrada: '%s', Saída: '%s', Relatório: '%s'",
            request.input_dir,
            request.output_dir,
            report_path,
        )

        try:
            success = self._validator.validate(
                input_dir=request.input_dir,
                output_dir=request.output_dir,
                report_path=report_path,
            )

            if success:
                logger.info("Validação de dados concluída com sucesso. Relatório em: %s", report_path)
            else:
                logger.warning("Validação de dados concluída com falhas.")

            return ValidateDataResponse(success=success, report_path=report_path)

        except Exception as e:
            logger.exception("Erro durante a validação de dados: %s", str(e))
            return ValidateDataResponse(success=False, report_path=report_path)
