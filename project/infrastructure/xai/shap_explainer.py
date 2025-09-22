from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import shap

from domain.services.explainer import IXAIExplainer


class ShapTabularExplainer(IXAIExplainer):
    """Explainer baseado em SHAP (Tree/Kernel/Linear) para features tabulares.

    - Usa heurística para escolher o Explainer adequado ao estimador.
    - Entrada X pode ser numpy array ou pandas DataFrame.
    """

    def __init__(self, background: Optional[Any] = None) -> None:
        self.background = background

    def _select_explainer(self, model: Any, X: Any):
        # Heurística simples: se modelo tem predict_proba e é árvore -> TreeExplainer
        try:
            import sklearn  # noqa: F401
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

            if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                return shap.TreeExplainer(model)
        except Exception:
            pass
        # Fallback genérico
        bg = self.background if self.background is not None else X[:100]
        return shap.KernelExplainer(model.predict_proba, bg)  # type: ignore[arg-type]

    def explain(self, X: Any, model: Any) -> Dict[str, Any]:
        X_np = np.asarray(X)
        explainer = self._select_explainer(model, X_np)
        try:
            # Prefer SHAP values da classe positiva
            shap_values = explainer.shap_values(X_np)
        except Exception:
            # Fallback: valores absolutos aproximados
            preds = model.predict_proba(X_np)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_np)
            shap_values = np.expand_dims(preds - preds.mean(), axis=1)

        result = {
            "shap_values": shap_values,
            "expected_value": getattr(explainer, "expected_value", None),
        }
        return result
