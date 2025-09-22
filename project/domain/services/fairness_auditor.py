from abc import ABC, abstractmethod
from typing import Any
from domain.entities.fairness import FairnessReport


class IFairnessAuditor(ABC):
    """Interface para auditoria de fairness."""

    @abstractmethod
    def audit(self, X: Any, y_true: Any, y_pred: Any, sensitive_features: Any) -> FairnessReport:
        """Audita modelo e gera relatório de fairness por subgrupo."""
        raise NotImplementedError
 