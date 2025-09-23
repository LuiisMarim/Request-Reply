import logging
from dataclasses import dataclass
from typing import Dict, Optional
from .label import Label

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    """
    Representa uma amostra de imagem ou sequência de imagens.

    Attributes:
        id (str): Identificador único da amostra.
        file_path (str): Caminho para o arquivo da amostra.
        label (Optional[Label]): Rótulo clínico associado, se disponível.
        demographics (Dict[str, str]): Informações demográficas (idade, gênero, etnia, etc.).
        metadata (Dict[str, str]): Metadados adicionais (dimensões, canais, qualidade, etc.).
    """

    id: str
    file_path: str
    label: Optional[Label]
    demographics: Dict[str, str]
    metadata: Dict[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            logger.error("ID inválido para Sample: %s", self.id)
            raise ValueError("O campo 'id' deve ser uma string não vazia.")

        if not isinstance(self.file_path, str) or not self.file_path.strip():
            logger.error("Caminho de arquivo inválido para Sample: %s", self.file_path)
            raise ValueError("O campo 'file_path' deve ser uma string não vazia.")

        if self.label is not None and not isinstance(self.label, Label):
            logger.error("Rótulo inválido para Sample: %s", self.label)
            raise ValueError("O campo 'label' deve ser uma instância de Label ou None.")

        if not isinstance(self.demographics, dict):
            logger.error("Demographics inválido: %s", type(self.demographics))
            raise ValueError("O campo 'demographics' deve ser um dicionário.")

        for key, value in self.demographics.items():
            if not isinstance(key, str) or not isinstance(value, str):
                logger.error("Entrada inválida em demographics: %s -> %s", key, value)
                raise ValueError("As chaves e valores de 'demographics' devem ser strings.")

        if not isinstance(self.metadata, dict):
            logger.error("Metadata inválido: %s", type(self.metadata))
            raise ValueError("O campo 'metadata' deve ser um dicionário.")

        for key, value in self.metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                logger.error("Entrada inválida em metadata: %s -> %s", key, value)
                raise ValueError("As chaves e valores de 'metadata' devem ser strings.")

        logger.info("Sample inicializado com sucesso: id=%s, file_path=%s", self.id, self.file_path)

    def is_labeled(self) -> bool:
        """
        Verifica se a amostra possui rótulo definido.

        Returns:
            bool: True se o rótulo está definido, False caso contrário.
        """
        labeled = self.label is not None
        logger.debug("Verificação de rótulo para Sample %s: %s", self.id, labeled)
        return labeled
