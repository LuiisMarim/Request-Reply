from abc import ABC, abstractmethod
from typing import Any, List


class IEmbeddingExtractor(ABC):
    """
    Interface para extração de embeddings profundos.

    Esta interface define o contrato para implementações de extratores
    de embeddings que transformam imagens em vetores numéricos.
    """

    @abstractmethod
    def extract(self, image: Any) -> List[float]:
        """
        Extrai embedding vetorial de uma imagem.

        Args:
            image (Any): Representação da imagem de entrada (ex.: matriz NumPy, frame de vídeo).

        Returns:
            List[float]: Vetor numérico representando a embedding da imagem.

        Raises:
            ValueError: Se a entrada for inválida ou corrompida.
            RuntimeError: Se ocorrer falha durante o processo de extração.
        """
        raise NotImplementedError
