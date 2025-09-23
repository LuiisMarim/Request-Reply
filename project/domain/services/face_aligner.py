from abc import ABC, abstractmethod
from typing import Any


class IFaceAlignerService(ABC):
    """
    Interface para serviços de alinhamento facial.

    Esta interface define o contrato para implementações que recebem uma imagem
    e pontos faciais (landmarks), retornando a face alinhada.
    """

    @abstractmethod
    def align(self, image: Any, keypoints: Any) -> Any:
        """
        Alinha uma face em uma imagem com base em pontos faciais.

        Args:
            image (Any): Representação da imagem de entrada (ex.: matriz NumPy).
            keypoints (Any): Estrutura contendo pontos faciais detectados
                             (ex.: dicionário, lista ou array de coordenadas).

        Returns:
            Any: Imagem com a face alinhada (mesmo tipo da entrada `image`).

        Raises:
            ValueError: Se os parâmetros forem inválidos.
            RuntimeError: Se ocorrer falha durante o processo de alinhamento.
        """
        raise NotImplementedError
