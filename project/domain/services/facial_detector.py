from abc import ABC, abstractmethod
from typing import Any


class IFaceDetectorService(ABC):
    """Interface para detecção de faces."""

    @abstractmethod
    def detect(self, image: Any) -> Any:
        """Detecta faces em uma imagem e retorna bounding boxes ou keypoints."""
        raise NotImplementedError
