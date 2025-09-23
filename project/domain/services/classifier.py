from abc import ABC, abstractmethod
from typing import Any
from domain.entities.metrics import Metrics


class IClassifier(ABC):
    """
    Interface para classificadores (ex.: Random Forest, MLP, etc.).

    Esta interface define o contrato para implementação de classificadores
    responsáveis por treinar, predizer e avaliar modelos.
    """

    @abstractmethod
    def train(self, X: Any, y: Any) -> None:
        """
        Treina o classificador no dataset fornecido.

        Args:
            X (Any): Conjunto de features de entrada (ex.: DataFrame, array NumPy).
            y (Any): Rótulos correspondentes.

        Raises:
            ValueError: Se os dados de entrada forem inválidos.
            RuntimeError: Se ocorrer erro durante o processo de treino.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Gera predições para o conjunto de entrada.

        Args:
            X (Any): Conjunto de features para inferência.

        Returns:
            Any: Predições correspondentes (ex.: lista, array NumPy).

        Raises:
            ValueError: Se os dados de entrada forem inválidos.
            RuntimeError: Se ocorrer erro durante o processo de predição.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> Metrics:
        """
        Avalia o modelo treinado e retorna métricas de desempenho.

        Args:
            X (Any): Conjunto de features de validação/teste.
            y (Any): Rótulos reais para comparação.

        Returns:
            Metrics: Objeto contendo métricas como acurácia, AUC e F1-score.

        Raises:
            ValueError: Se os dados de entrada forem inválidos.
            RuntimeError: Se ocorrer erro durante o processo de avaliação.
        """
        raise NotImplementedError
