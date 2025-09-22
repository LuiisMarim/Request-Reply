from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import roc_auc_score


def build_threshold_optimizer(
    estimator: Any,
    constraints: str = "equalized_odds",
    predict_method: str = "predict_proba",
) -> ThresholdOptimizer:
    """Constrói pós-processamento via ThresholdOptimizer (Fairlearn)."""
    return ThresholdOptimizer(
        estimator=estimator,
        constraints=constraints,
        predict_method=predict_method,
    )


def evaluate_auc_per_group(
    estimator: Any, X: Any, y: Any, sensitive_features: Any
) -> Dict[str, float]:
    """Calcula AUC por subgrupo (para diagnóstico)."""
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)[:, 1]
    else:
        proba = estimator.predict(X)

    groups = np.asarray(sensitive_features)
    aucs: Dict[str, float] = {}
    for g in np.unique(groups):
        idx = groups == g
        try:
            aucs[str(g)] = float(roc_auc_score(y[idx], proba[idx]))
        except Exception:
            aucs[str(g)] = float("nan")
    return aucs
