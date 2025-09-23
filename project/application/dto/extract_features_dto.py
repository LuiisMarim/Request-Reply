from dataclasses import dataclass
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractFeaturesRequest:
    """
    Request DTO para extração de features.

    Attributes:
        input_dir (str): Caminho para o diretório de entrada contendo os dados.
        output_dir (str): Caminho para o diretório de saída para salvar as features extraídas.
        profile (str): Perfil de execução a ser utilizado (ex.: 'medium', 'fast', 'accurate').
    """

    input_dir: str
    output_dir: str
    profile: str = "medium"

    def __post_init__(self) -> None:
        if not self.input_dir or not os.path.isdir(self.input_dir):
            logger.error("Diretório de entrada inválido: %s", self.input_dir)
            raise ValueError(f"Diretório de entrada inválido: {self.input_dir}")

        if not self.output_dir:
            logger.error("Diretório de saída não fornecido.")
            raise ValueError("Diretório de saída não pode ser vazio.")

        if not isinstance(self.profile, str) or not self.profile.strip():
            logger.error("Perfil de execução inválido: %s", self.profile)
            raise ValueError("Perfil de execução inválido.")

        logger.info("ExtractFeaturesRequest inicializado com sucesso: %s", self)


@dataclass(frozen=True)
class ExtractFeaturesResponse:
    """
    Response DTO para extração de features.

    Attributes:
        success (bool): Indica se a extração de features foi concluída com sucesso.
        features_dir (str): Caminho do diretório que contém as features extraídas.
        log_path (Optional[str]): Caminho para o log de execução, se disponível.
    """

    success: bool
    features_dir: str
    log_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            logger.error("Valor inválido para success: %s", self.success)
            raise ValueError("O campo 'success' deve ser um booleano.")

        if not self.features_dir or not os.path.isdir(self.features_dir):
            logger.error("Diretório de features inválido: %s", self.features_dir)
            raise ValueError(f"Diretório de features inválido: {self.features_dir}")

        if self.log_path is not None and not isinstance(self.log_path, str):
            logger.error("Caminho de log inválido: %s", self.log_path)
            raise ValueError("O campo 'log_path' deve ser uma string ou None.")

        logger.info("ExtractFeaturesResponse inicializado com sucesso: %s", self)
