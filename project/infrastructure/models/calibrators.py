from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt


CalibrationMethod = Literal["sigmoid", "isotonic"]


@dataclass(frozen=True)
class CalibrationResult:
    """Resultado da calibração com curva de confiabilidade."""
    method: CalibrationMethod
    prob_true: np.ndarray
    prob_pred: np.ndarray
    estimator: BaseEstimator


def calibrate_classifier(
    estimator: ClassifierMixin,
    X: Any,
    y: Any,
    method: CalibrationMethod = "sigmoid",
    cv: int = 5,
) -> BaseEstimator:
    """Empacota um estimador com CalibratedClassifierCV."""
    calib = CalibratedClassifierCV(estimator, method=method, cv=cv)
    return calib.fit(X, y)


def calibration_diagnostics(
    calibrated_estimator: ClassifierMixin, X: Any, y: Any, n_bins: int = 10
) -> CalibrationResult:
    """Calcula curvas de calibração (reliability curve)."""
    proba = calibrated_estimator.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="uniform")
    method: CalibrationMethod = "sigmoid" if isinstance(calibrated_estimator, CalibratedClassifierCV) and calibrated_estimator.method == "sigmoid" else "isotonic"
    return CalibrationResult(method=method, prob_true=prob_true, prob_pred=prob_pred, estimator=calibrated_estimator)


def plot_reliability_curve(result: CalibrationResult, out_path: str) -> str:
    """Gera gráfico de calibração e salva em arquivo."""
    plt.figure()
    plt.plot(result.prob_pred, result.prob_true, marker="o", label="Calibrated")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.title(f"Reliability Curve ({result.method})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
