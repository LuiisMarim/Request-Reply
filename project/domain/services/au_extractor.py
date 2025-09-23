from abc import ABC, abstractmethod
from typing import Any
from domain.entities.action_units import ActionUnits


class IAUExtractorService(ABC):
    """
    Interface para serviços de extração de Action Units (AUs).

    Esta interface define o contrato para implementação de extratores de AUs,
    que recebem uma imagem como entrada e retornam um objeto `ActionUnits`.

    Responsabilidades das implementações concretas:
        - Validar a entrada (garantir que a imagem seja do tipo esperado).
        - Tratar erros internos de bibliotecas externas.
        - Registrar logs informativos e de erro quando apropriado.
    """

    @abstractmethod
    def extract(self, image: Any) -> ActionUnits:
        """
        Extrai intensidades das AUs a partir de uma imagem.

        Args:
            image (Any): Representação da imagem de entrada (ex.: matriz NumPy, frame de vídeo).

        Returns:
            ActionUnits: Objeto contendo o mapeamento de AUs e suas intensidades.

        Raises:
            ValueError: Se a entrada for inválida.
            RuntimeError: Se ocorrer erro durante o processo de extração.
        """
        raise NotImplementedError
