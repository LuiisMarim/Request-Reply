from abc import ABC, abstractmethod
from typing import Dict, Any


class IFeatureFusion(ABC):
    """
    Interface para fusão de features multimodais.

    Define o contrato para implementações responsáveis por unificar diferentes
    tipos de features (ex.: AUs, microexpressões, embeddings) em um vetor único.
    """

    @abstractmethod
    def fuse(self, features: Dict[str, Any]) -> Any:
        """
        Realiza a fusão de features em um vetor unificado.

        Args:
            features (Dict[str, Any]): Dicionário contendo diferentes modalidades
                                       de features (ex.: {"aus": ..., "embeddings": ...}).

        Returns:
            Any: Vetor unificado representando a amostra.

        Raises:
            ValueError: Se o dicionário de features for inválido ou inconsistente.
            RuntimeError: Se ocorrer erro durante a fusão das features.
        """
        raise NotImplementedError
