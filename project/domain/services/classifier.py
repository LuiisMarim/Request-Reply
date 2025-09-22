from abc import ABC, abstractmethod
from typing import Any
from domain.entities.metrics import Metrics


class IClassifier(ABC):
    """Interface para classificadores (RF, MLP, etc.)."""

    @abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """Treina classificador no dataset fornecido."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Retorna predições para X."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> Metrics:
        """Avalia modelo e retorna métricas."""
        raise NotImplementedError
