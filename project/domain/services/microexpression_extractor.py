from abc import ABC, abstractmethod
from typing import Any, List
from domain.entities.microexpressions import MicroExpression


class IMicroexpressionExtractor(ABC):
    """Interface para extração de microexpressões em sequências."""

    @abstractmethod
    def extract(self, frames: List[Any]) -> List[MicroExpression]:
        """Extrai microexpressões de uma sequência de frames."""
        raise NotImplementedError
