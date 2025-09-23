from abc import ABC, abstractmethod
from typing import Any


class IFaceDetectorService(ABC):
    """
    Interface para serviços de detecção de faces.

    Esta interface define o contrato para implementações que detectam
    faces em uma imagem e retornam bounding boxes e/ou keypoints.
    """

    @abstractmethod
    def detect(self, image: Any) -> Any:
        """
        Detecta faces em uma imagem.

        Args:
            image (Any): Representação da imagem de entrada (ex.: matriz NumPy, frame de vídeo).

        Returns:
            Any: Estrutura contendo bounding boxes e/ou keypoints das faces detectadas.
                 O formato exato depende da implementação.

        Raises:
            ValueError: Se a imagem fornecida for inválida.
            RuntimeError: Se ocorrer falha durante o processo de detecção.
        """
        raise NotImplementedError
