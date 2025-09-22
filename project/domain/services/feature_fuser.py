from abc import ABC, abstractmethod
from typing import Dict, Any


class IFeatureFusion(ABC):
    """Interface para fusão de features (AUs, microexpressões, embeddings)."""

    @abstractmethod
    def fuse(self, features: Dict[str, Any]) -> Any:
        """Recebe dicionário de features e retorna vetor unificado."""
        raise NotImplementedError
