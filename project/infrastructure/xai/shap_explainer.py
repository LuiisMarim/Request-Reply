from __future__ import annotations

from typing import Any, Dict, Optional
import logging

import numpy as np
import shap

from domain.services.explainer import IXAIExplainer

logger = logging.getLogger(__name__)


class ShapTabularExplainer(IXAIExplainer):
    """
    Explainer baseado em SHAP (Tree/Kernel/Linear) para features tabulares.

    Estratégia:
        - Usa heurística para escolher o Explainer adequado ao modelo.
        - Entrada X pode ser numpy array ou pandas DataFrame.
        - Retorna SHAP values e expected_value, com fallback seguro.
    """

    def __init__(self, background: Optional[Any] = None) -> None:
        self.background = background

    def _select_explainer(self, model: Any, X: Any):
        """
        Seleciona o tipo de explainer SHAP com base no modelo.

        Args:
            model (Any): Modelo treinado.
            X (Any): Features de entrada.

        Returns:
            shap.Explainer: Explainer apropriado.

        Raises:
            RuntimeError: Se não conseguir inicializar explainer.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

            if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
                logger.info("Selecionado TreeExplainer para modelo %s", type(model).__name__)
                return shap.TreeExplainer(model)
        except Exception as e:
            logger.warning("Falha ao importar sklearn para TreeExplainer: %s", str(e))

        # Fallback genérico
        try:
            bg = self.background if self.background is not None else X[:100]
            logger.info("Selecionado KernelExplainer como fallback.")
            return shap.KernelExplainer(model.predict_proba, bg)  # type: ignore[arg-type]
        except Exception as e:
            logger.exception("Falha ao inicializar KernelExplainer: %s", str(e))
            raise RuntimeError("Não foi possível inicializar um explainer SHAP.") from e

    def explain(self, X: Any, model: Any) -> Dict[str, Any]:
        """
        Gera explicações SHAP para o modelo e entradas fornecidas.

        Args:
            X (Any): Features de entrada.
            model (Any): Modelo treinado.

        Returns:
            Dict[str, Any]: Dicionário contendo shap_values e expected_value.

        Raises:
            ValueError: Se X ou model forem inválidos.
        """
        if model is None:
            logger.error("Modelo inválido fornecido ao explainer.")
            raise ValueError("model não pode ser None.")

        try:
            X_np = np.asarray(X)
        except Exception as e:
            logger.exception("Falha ao converter X para np.ndarray: %s", str(e))
            raise ValueError("X deve ser conversível para np.ndarray.") from e

        try:
            explainer = self._select_explainer(model, X_np)
            shap_values = explainer.shap_values(X_np)
            logger.info("SHAP values gerados com sucesso.")
        except Exception as e:
            logger.warning("Falha ao calcular SHAP values, aplicando fallback: %s", str(e))
            try:
                preds = (
                    model.predict_proba(X_np)[:, 1]
                    if hasattr(model, "predict_proba")
                    else model.predict(X_np)
                )
                shap_values = np.expand_dims(preds - preds.mean(), axis=1)
            except Exception as e2:
                logger.exception("Falha também no fallback SHAP: %s", str(e2))
                raise RuntimeError("Falha ao gerar explicações SHAP.") from e2

        result = {
            "shap_values": shap_values,
            "expected_value": getattr(explainer, "expected_value", None),
        }
        return result
