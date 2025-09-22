from abc import ABC, abstractmethod
from typing import Any, List


class IEmbeddingExtractor(ABC):
    """Interface para extração de embeddings profundos."""

    @abstractmethod
    def extract(self, image: Any) -> List[float]:
        """Extrai embedding vetorial de uma imagem."""
        raise NotImplementedError
