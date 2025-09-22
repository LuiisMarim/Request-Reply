from abc import ABC, abstractmethod
from typing import Any


class ILightingNormalizer(ABC):
    """Interface para normalização de iluminação."""

    @abstractmethod
    def normalize(self, image: Any) -> Any:
        """Aplica técnicas de normalização de iluminação (ex.: CLAHE)."""
        raise NotImplementedError
