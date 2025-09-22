from abc import ABC, abstractmethod
from typing import Any
from domain.entities.action_units import ActionUnits


class IAUExtractorService(ABC):
    """Interface para extração de Action Units (AUs)."""

    @abstractmethod
    def extract(self, image: Any) -> ActionUnits:
        """Extrai intensidades das AUs de uma imagem."""
        raise NotImplementedError
