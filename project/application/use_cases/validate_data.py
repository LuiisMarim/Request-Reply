from application.dto.validate_data_dto import ValidateDataRequest, ValidateDataResponse
from domain.services.repository import IDataRepository
from typing import Protocol


class IDataValidator(Protocol):
    """Protocolo para validador de dados (infrastructure.data.validators)."""

    def validate(self, input_dir: str, output_dir: str, report_path: str) -> bool:
        ...


class ValidateDataUseCase:
    """Caso de uso: validação e preparação de dados."""

    def __init__(self, repository: IDataRepository, validator: IDataValidator):
        self._repository = repository
        self._validator = validator

    def execute(self, request: ValidateDataRequest) -> ValidateDataResponse:
        report_path = request.report_path or f"{request.output_dir}/data_quality.html"
        success = self._validator.validate(
            input_dir=request.input_dir,
            output_dir=request.output_dir,
            report_path=report_path,
        )
        return ValidateDataResponse(success=success, report_path=report_path)
