from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

CalibrationMethod = Literal["sigmoid", "isotonic"]


@dataclass(frozen=True)
class CalibrationResult:
    """
    Resultado da calibração com curva de confiabilidade.

    Attributes:
        method (CalibrationMethod): Método de calibração utilizado.
        prob_true (np.ndarray): Frequências observadas.
        prob_pred (np.ndarray): Probabilidades preditas.
        estimator (BaseEstimator): Estimador calibrado.
    """

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
    """
    Empacota um estimador com CalibratedClassifierCV e o treina.

    Args:
        estimator (ClassifierMixin): Estimador base.
        X (Any): Features de treino.
        y (Any): Labels de treino.
        method (CalibrationMethod): Método de calibração ('sigmoid' ou 'isotonic').
        cv (int): Número de folds para validação cruzada.

    Returns:
        BaseEstimator: Estimador calibrado.

    Raises:
        ValueError: Se entradas forem inválidas.
        RuntimeError: Se ocorrer falha durante o processo de calibração.
    """
    if method not in ("sigmoid", "isotonic"):
        logger.error("Método de calibração inválido: %s", method)
        raise ValueError("O método deve ser 'sigmoid' ou 'isotonic'.")

    if not hasattr(estimator, "fit"):
        logger.error("Estimator não possui método 'fit': %s", type(estimator))
        raise ValueError("O estimador fornecido não é válido para calibração.")

    try:
        logger.info("Iniciando calibração com método '%s' e cv=%d.", method, cv)
        calib = CalibratedClassifierCV(estimator, method=method, cv=cv)
        fitted = calib.fit(X, y)
        logger.info("Calibração concluída com sucesso.")
        return fitted
    except Exception as e:
        logger.exception("Erro durante calibração: %s", str(e))
        raise RuntimeError(f"Erro durante calibração: {str(e)}") from e


def calibration_diagnostics(
    calibrated_estimator: ClassifierMixin, X: Any, y: Any, n_bins: int = 10
) -> CalibrationResult:
    """
    Calcula curvas de calibração (reliability curve).

    Args:
        calibrated_estimator (ClassifierMixin): Estimador previamente calibrado.
        X (Any): Features de avaliação.
        y (Any): Labels de avaliação.
        n_bins (int): Número de bins para a curva de calibração.

    Returns:
        CalibrationResult: Objeto contendo resultados da calibração.

    Raises:
        RuntimeError: Se falhar o cálculo da curva.
    """
    try:
        logger.info("Calculando curva de calibração com %d bins.", n_bins)
        proba = calibrated_estimator.predict_proba(X)[:, 1]
        prob_true, prob_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="uniform")

        method: CalibrationMethod = (
            "sigmoid"
            if isinstance(calibrated_estimator, CalibratedClassifierCV)
            and getattr(calibrated_estimator, "method", "sigmoid") == "sigmoid"
            else "isotonic"
        )

        logger.info("Curva de calibração calculada com sucesso.")
        return CalibrationResult(method=method, prob_true=prob_true, prob_pred=prob_pred, estimator=calibrated_estimator)
    except Exception as e:
        logger.exception("Erro ao calcular curva de calibração: %s", str(e))
        raise RuntimeError(f"Erro ao calcular curva de calibração: {str(e)}") from e


def plot_reliability_curve(result: CalibrationResult, out_path: str) -> str:
    """
    Gera gráfico de calibração e salva em arquivo.

    Args:
        result (CalibrationResult): Resultado da calibração.
        out_path (str): Caminho para salvar a imagem.

    Returns:
        str: Caminho do arquivo salvo.

    Raises:
        ValueError: Se out_path for inválido.
        RuntimeError: Se falhar a geração ou o salvamento do gráfico.
    """
    if not isinstance(out_path, str) or not out_path.strip():
        logger.error("Caminho de saída inválido: %s", out_path)
        raise ValueError("O caminho de saída deve ser uma string válida.")

    try:
        logger.info("Gerando gráfico de calibração em: %s", out_path)
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
        logger.info("Gráfico de calibração salvo em: %s", out_path)
        return out_path
    except Exception as e:
        logger.exception("Erro ao salvar gráfico de calibração: %s", str(e))
        raise RuntimeError(f"Erro ao salvar gráfico de calibração: {str(e)}") from e
