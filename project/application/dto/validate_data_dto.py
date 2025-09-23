from dataclasses import dataclass
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidateDataRequest:
    """
    Request DTO para validação de dados.

    Attributes:
        input_dir (str): Caminho do diretório de entrada contendo os dados a serem validados.
        output_dir (str): Caminho do diretório onde os resultados da validação serão salvos.
        report_path (Optional[str]): Caminho opcional para um relatório pré-existente.
    """

    input_dir: str
    output_dir: str
    report_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.input_dir or not os.path.isdir(self.input_dir):
            logger.error("Diretório de entrada inválido: %s", self.input_dir)
            raise ValueError(f"Diretório de entrada inválido: {self.input_dir}")

        if not self.output_dir:
            logger.error("Diretório de saída não fornecido.")
            raise ValueError("Diretório de saída não pode ser vazio.")

        if self.report_path is not None and not isinstance(self.report_path, str):
            logger.error("Caminho de relatório inválido: %s", self.report_path)
            raise ValueError("O campo 'report_path' deve ser string ou None.")

        logger.info("ValidateDataRequest inicializado com sucesso: %s", self)


@dataclass(frozen=True)
class ValidateDataResponse:
    """
    Response DTO para validação de dados.

    Attributes:
        success (bool): Indica se a validação foi concluída com sucesso.
        report_path (str): Caminho para o relatório gerado pela validação.
    """

    success: bool
    report_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            logger.error("Valor inválido para success: %s", self.success)
            raise ValueError("O campo 'success' deve ser um booleano.")

        if not self.report_path:
            logger.error("Caminho do relatório não pode ser vazio.")
            raise ValueError("O campo 'report_path' não pode ser vazio.")

        logger.info("ValidateDataResponse inicializado com sucesso: %s", self)
