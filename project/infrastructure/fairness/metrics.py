from __future__ import annotations

from typing import Any, Dict

import numpy as np
import logging
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate,
)

logger = logging.getLogger(__name__)


def compute_fairness_metrics(
    y_true: Any, y_pred: Any, sensitive_features: Any
) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas de fairness por subgrupo e diferenças agregadas.

    Args:
        y_true (Any): Rótulos verdadeiros.
        y_pred (Any): Predições do modelo.
        sensitive_features (Any): Atributos sensíveis (ex.: gênero, etnia, idade).

    Returns:
        Dict[str, Dict[str, float]]: Dicionário com métricas globais e por subgrupo.

    Raises:
        ValueError: Se entradas forem inválidas ou inconsistentes.
    """
    try:
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
    except Exception as e:
        logger.exception("Falha ao converter y_true ou y_pred em arrays NumPy: %s", str(e))
        raise ValueError("y_true e y_pred devem ser convertíveis para arrays NumPy inteiros.") from e

    if y_true.shape[0] != y_pred.shape[0]:
        logger.error("Tamanho incompatível entre y_true (%d) e y_pred (%d).", len(y_true), len(y_pred))
        raise ValueError("y_true e y_pred devem ter o mesmo comprimento.")

    if len(sensitive_features) != len(y_true):
        logger.error("Tamanho incompatível entre sensitive_features (%d) e y_true (%d).", len(sensitive_features), len(y_true))
        raise ValueError("sensitive_features deve ter o mesmo comprimento que y_true.")

    logger.info("Calculando métricas de fairness para %d amostras.", len(y_true))

    try:
        frame = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        # Diferenças globais (pior - melhor subgrupo)
        dp_diff = demographic_parity_difference(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
        )
        eo_diff = equalized_odds_difference(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
        )

        per_group = {
            str(k): {
                "selection_rate": float(frame.by_group["selection_rate"][k]),
                "tpr": float(frame.by_group["tpr"][k]),
                "fpr": float(frame.by_group["fpr"][k]),
            }
            for k in frame.by_group["selection_rate"].keys()
        }

        summary = {
            "overall": {
                "demographic_parity_diff": float(dp_diff),
                "equalized_odds_diff": float(eo_diff),
            },
            "per_group": per_group,
        }

        logger.info(
            "Métricas de fairness calculadas: demographic_parity_diff=%.4f, equalized_odds_diff=%.4f",
            float(dp_diff),
            float(eo_diff),
        )

        return summary

    except Exception as e:
        logger.exception("Erro durante cálculo das métricas de fairness: %s", str(e))
        raise
