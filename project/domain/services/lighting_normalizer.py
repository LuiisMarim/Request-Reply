from abc import ABC, abstractmethod
from typing import Any


class ILightingNormalizer(ABC):
    """
    Interface para normalização de iluminação em imagens.

    Define o contrato para implementações que aplicam técnicas de normalização
    de iluminação (ex.: CLAHE, histogram equalization, Retinex).
    """

    @abstractmethod
    def normalize(self, image: Any) -> Any:
        """
        Aplica técnicas de normalização de iluminação à imagem fornecida.

        Args:
            image (Any): Representação da imagem de entrada (ex.: matriz NumPy).

        Returns:
            Any: Imagem processada com iluminação normalizada.

        Raises:
            ValueError: Se a imagem de entrada for inválida.
            RuntimeError: Se ocorrer erro durante a normalização.
        """
        raise NotImplementedError
