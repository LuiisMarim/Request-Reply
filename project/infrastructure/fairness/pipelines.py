from __future__ import annotations

from typing import Any, Dict
import numpy as np
import logging
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def build_threshold_optimizer(
    estimator: Any,
    constraints: str = "equalized_odds",
    predict_method: str = "predict_proba",
) -> ThresholdOptimizer:
    """
    Constrói pós-processamento via ThresholdOptimizer (Fairlearn).

    Args:
        estimator (Any): Estimador treinado (modelo compatível com scikit-learn).
        constraints (str): Restrição de fairness (ex.: 'equalized_odds', 'demographic_parity').
        predict_method (str): Método de previsão usado pelo estimador.

    Returns:
        ThresholdOptimizer: Objeto configurado para pós-processamento.

    Raises:
        ValueError: Se o estimador for inválido ou se constraints não for string.
    """
    if not isinstance(constraints, str) or not constraints.strip():
        logger.error("Constraints inválido: %s", constraints)
        raise ValueError("O parâmetro 'constraints' deve ser uma string não vazia.")

    if estimator is None:
        logger.error("Estimator inválido: None fornecido.")
        raise ValueError("Um estimator válido deve ser fornecido.")

    logger.info("Construindo ThresholdOptimizer com constraint='%s' e método='%s'.", constraints, predict_method)

    try:
        optimizer = ThresholdOptimizer(
            estimator=estimator,
            constraints=constraints,
            predict_method=predict_method,
        )
        logger.info("ThresholdOptimizer criado com sucesso.")
        return optimizer
    except Exception as e:
        logger.exception("Erro ao construir ThresholdOptimizer: %s", str(e))
        raise


def evaluate_auc_per_group(
    estimator: Any, X: Any, y: Any, sensitive_features: Any
) -> Dict[str, float]:
    """
    Calcula AUC por subgrupo (para diagnóstico de fairness).

    Args:
        estimator (Any): Modelo treinado para avaliação.
        X (Any): Features de entrada.
        y (Any): Rótulos verdadeiros.
        sensitive_features (Any): Atributos sensíveis para estratificação.

    Returns:
        Dict[str, float]: Dicionário com AUC calculado por grupo.

    Raises:
        ValueError: Se tamanhos de X, y e sensitive_features não coincidirem.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups = np.asarray(sensitive_features)

    if len(X) != len(y) or len(y) != len(groups):
        logger.error(
            "Tamanhos incompatíveis: X=%d, y=%d, sensitive_features=%d",
            len(X),
            len(y),
            len(groups),
        )
        raise ValueError("X, y e sensitive_features devem ter o mesmo comprimento.")

    logger.info("Iniciando cálculo de AUC por grupo para %d amostras.", len(y))

    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)[:, 1]
        except Exception as e:
            logger.exception("Erro ao calcular probabilidades com predict_proba: %s", str(e))
            raise
    elif hasattr(estimator, "predict"):
        try:
            proba = estimator.predict(X)
        except Exception as e:
            logger.exception("Erro ao calcular predições: %s", str(e))
            raise
    else:
        logger.error("Estimator não possui método 'predict_proba' nem 'predict'.")
        raise ValueError("O estimator deve implementar 'predict_proba' ou 'predict'.")

    aucs: Dict[str, float] = {}
    for g in np.unique(groups):
        idx = groups == g
        try:
            auc_value = float(roc_auc_score(y[idx], proba[idx]))
            aucs[str(g)] = auc_value
            logger.debug("AUC para grupo '%s': %.4f", g, auc_value)
        except Exception as e:
            logger.warning("Falha ao calcular AUC para grupo '%s': %s", g, str(e))
            aucs[str(g)] = float("nan")

    logger.info("Cálculo de AUC por grupo concluído.")
    return aucs
