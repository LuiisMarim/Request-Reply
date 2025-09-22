from __future__ import annotations

from typing import Any, Dict

import numpy as np
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate,
)


def compute_fairness_metrics(
    y_true: Any, y_pred: Any, sensitive_features: Any
) -> Dict[str, Dict[str, float]]:
    """Calcula métricas de fairness por subgrupo e diferenças agregadas."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

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
    dp_diff = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    eo_diff = equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)

    per_group = {
        str(k): {
            "selection_rate": float(frame.by_group["selection_rate"][k]),
            "tpr": float(frame.by_group["tpr"][k]),
            "fpr": float(frame.by_group["fpr"][k]),
        }
        for k in frame.by_group["selection_rate"].keys()
    }

    summary = {
        "overall": {"demographic_parity_diff": float(dp_diff), "equalized_odds_diff": float(eo_diff)},
        "per_group": per_group,
    }
    return summary
