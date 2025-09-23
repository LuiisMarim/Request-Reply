from abc import ABC, abstractmethod
from typing import Any
from domain.entities.fairness import FairnessReport


class IFairnessAuditor(ABC):
    """
    Interface para auditoria de fairness.

    Define o contrato para implementação de auditores de fairness que
    avaliam desempenho do modelo em diferentes subgrupos sensíveis.
    """

    @abstractmethod
    def audit(self, X: Any, y_true: Any, y_pred: Any, sensitive_features: Any) -> FairnessReport:
        """
        Audita modelo e gera relatório de fairness por subgrupo.

        Args:
            X (Any): Conjunto de features de entrada.
            y_true (Any): Rótulos verdadeiros.
            y_pred (Any): Predições geradas pelo modelo.
            sensitive_features (Any): Atributos sensíveis (ex.: gênero, etnia, idade).

        Returns:
            FairnessReport: Relatório contendo métricas de fairness por subgrupo.

        Raises:
            ValueError: Se os parâmetros fornecidos forem inválidos.
            RuntimeError: Se ocorrer falha durante a auditoria.
        """
        raise NotImplementedError
