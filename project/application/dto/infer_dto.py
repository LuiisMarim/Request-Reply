from dataclasses import dataclass
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferRequest:
    """
    Request DTO para inferência em novos dados.

    Attributes:
        input_dir (str): Caminho para o diretório de entrada contendo os dados a serem inferidos.
        models_dir (str): Caminho para o diretório que contém os modelos treinados.
        output_dir (str): Caminho para o diretório de saída onde os resultados serão salvos.
        profile (str): Perfil de execução (ex.: 'medium', 'fast', 'accurate').
    """

    input_dir: str
    models_dir: str
    output_dir: str
    profile: str = "medium"

    def __post_init__(self) -> None:
        if not self.input_dir or not os.path.isdir(self.input_dir):
            logger.error("Diretório de entrada inválido: %s", self.input_dir)
            raise ValueError(f"Diretório de entrada inválido: {self.input_dir}")

        if not self.models_dir or not os.path.isdir(self.models_dir):
            logger.error("Diretório de modelos inválido: %s", self.models_dir)
            raise ValueError(f"Diretório de modelos inválido: {self.models_dir}")

        if not self.output_dir:
            logger.error("Diretório de saída não fornecido.")
            raise ValueError("Diretório de saída não pode ser vazio.")

        if not isinstance(self.profile, str) or not self.profile.strip():
            logger.error("Perfil de execução inválido: %s", self.profile)
            raise ValueError("Perfil de execução inválido.")

        logger.info("InferRequest inicializado com sucesso: %s", self)


@dataclass(frozen=True)
class InferResponse:
    """
    Response DTO para inferência.

    Attributes:
        success (bool): Indica se a inferência foi concluída com sucesso.
        results_path (str): Caminho para os resultados gerados pela inferência.
        report_path (Optional[str]): Caminho para o relatório de execução, se disponível.
    """

    success: bool
    results_path: str
    report_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            logger.error("Valor inválido para success: %s", self.success)
            raise ValueError("O campo 'success' deve ser um booleano.")

        if not self.results_path:
            logger.error("Caminho de resultados não pode ser vazio.")
            raise ValueError("Caminho de resultados não pode ser vazio.")

        logger.info("InferResponse inicializado com sucesso: %s", self)
