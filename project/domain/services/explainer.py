from abc import ABC, abstractmethod
from typing import Any, Dict


class IXAIExplainer(ABC):
    """Interface para explicabilidade (SHAP, Grad-CAM, attention maps)."""

    @abstractmethod
    def explain(self, X: Any, model: Any) -> Dict[str, Any]:
        """Gera explicações locais/globais para entrada e modelo."""
        raise NotImplementedError
