from abc import ABC, abstractmethod
from typing import Any, List
from domain.entities.microexpressions import MicroExpression


class IMicroexpressionExtractor(ABC):
    """
    Interface para extração de microexpressões em sequências.

    Esta interface define o contrato para implementações responsáveis por
    detectar microexpressões em uma sequência de frames.
    """

    @abstractmethod
    def extract(self, frames: List[Any]) -> List[MicroExpression]:
        """
        Extrai microexpressões de uma sequência de frames.

        Args:
            frames (List[Any]): Sequência de frames (ex.: arrays NumPy, imagens PIL).

        Returns:
            List[MicroExpression]: Lista de microexpressões detectadas na sequência.

        Raises:
            ValueError: Se os frames forem inválidos ou inconsistentes.
            RuntimeError: Se ocorrer falha durante a detecção das microexpressões.
        """
        raise NotImplementedError
