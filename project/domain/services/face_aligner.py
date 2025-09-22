from abc import ABC, abstractmethod
from typing import Any


class IFaceAlignerService(ABC):
    """Interface para alinhamento facial."""

    @abstractmethod
    def align(self, image: Any, keypoints: Any) -> Any:
        """Recebe imagem + pontos faciais e retorna face alinhada."""
        raise NotImplementedError
