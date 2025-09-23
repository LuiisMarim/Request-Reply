from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging

from domain.entities.metrics import Metrics
from domain.services.classifier import IClassifier

logger = logging.getLogger(__name__)


class RandomForestClassifierWrapper(IClassifier):
    """
    Wrapper de RandomForest (scikit-learn) compatível com IClassifier.

    Suporta calibração opcional (Platt/Isotonic) via CalibratedClassifierCV.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        class_weight: Optional[Dict[int, float]] = None,
        calibration: Optional[str] = "sigmoid",  # "sigmoid" (Platt) | "isotonic" | None
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if calibration not in ("sigmoid", "isotonic", None):
            raise ValueError("calibration deve ser 'sigmoid', 'isotonic' ou None.")
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.calibration = calibration
        self.clf: Any = self.rf  # substituído após calibração
        logger.info(
            "RandomForestClassifierWrapper inicializado (n_estimators=%d, max_depth=%s, calibration=%s)",
            n_estimators,
            str(max_depth),
            calibration,
        )

    def _ensure_calibrated(self, X: np.ndarray, y: np.ndarray) -> None:
        """Treina o modelo com ou sem calibração, conforme configuração."""
        if self.calibration is None:
            self.clf = self.rf.fit(X, y)
        else:
            self.clf = CalibratedClassifierCV(self.rf, method=self.calibration, cv=5)
            self.clf = self.clf.fit(X, y)

    def train(self, X: Any, y: Any) -> None:
        """
        Treina o classificador RandomForest.

        Args:
            X (Any): Features de treino.
            y (Any): Labels de treino.

        Raises:
            ValueError: Se X ou y forem inválidos.
        """
        if X is None or y is None:
            raise ValueError("X e y não podem ser None para treino.")
        try:
            X_np, y_np = self._to_numpy(X, y)
            self._ensure_calibrated(X_np, y_np)
            logger.info("Treinamento concluído com %d amostras.", len(X_np))
        except Exception as e:
            logger.exception("Erro durante treinamento: %s", str(e))
            raise

    def predict(self, X: Any) -> Any:
        """
        Gera predições binárias para as features fornecidas.

        Args:
            X (Any): Features de entrada.

        Returns:
            Any: Predições (array binário).

        Raises:
            ValueError: Se X for inválido.
        """
        if X is None:
            raise ValueError("X não pode ser None para predição.")
        try:
            X_np = self._to_numpy(X)
            if hasattr(self.clf, "predict_proba"):
                proba = self.clf.predict_proba(X_np)[:, 1]
                preds = (proba >= 0.5).astype(int)
            else:
                preds = self.clf.predict(X_np)
            logger.info("Predição concluída. Total de amostras: %d", len(preds))
            return preds
        except Exception as e:
            logger.exception("Erro durante predição: %s", str(e))
            raise

    def evaluate(self, X: Any, y: Any) -> Metrics:
        """
        Avalia o modelo em dados fornecidos e retorna métricas de desempenho.

        Args:
            X (Any): Features de entrada.
            y (Any): Rótulos verdadeiros.

        Returns:
            Metrics: Objeto com métricas calculadas.
        """
        if X is None or y is None:
            raise ValueError("X e y não podem ser None para avaliação.")
        try:
            X_np, y_np = self._to_numpy(X, y)
            if hasattr(self.clf, "predict_proba"):
                proba = self.clf.predict_proba(X_np)[:, 1]
            else:
                proba = self.clf.predict(X_np)
            y_pred = (proba >= 0.5).astype(int)

            acc = accuracy_score(y_np, y_pred)
            try:
                auc = roc_auc_score(y_np, proba)
            except Exception:
                logger.warning("Falha ao calcular AUC, retornando NaN.")
                auc = float("nan")
            f1 = f1_score(y_np, y_pred)

            logger.info("Avaliação concluída. Acc=%.3f, AUC=%.3f, F1=%.3f", acc, auc, f1)

            return Metrics(accuracy=acc, auc=auc, f1_score=f1, per_group={})
        except Exception as e:
            logger.exception("Erro durante avaliação: %s", str(e))
            raise

    @staticmethod
    def _to_numpy(X: Any, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Converte entradas em arrays NumPy.

        Args:
            X (Any): Features.
            y (Optional[Any]): Labels.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Arrays NumPy de X e y.
        """
        X_np = np.asarray(X)
        y_np = None if y is None else np.asarray(y).astype(int)
        return X_np, y_np

    def save(self, path: str) -> None:
        """Salva o modelo treinado em disco."""
        try:
            import joblib
            joblib.dump(self.clf, path)
            logger.info("Modelo salvo em: %s", path)
        except Exception as e:
            logger.exception("Erro ao salvar modelo: %s", str(e))
            raise

    def load(self, path: str) -> None:
        """Carrega o modelo treinado de disco."""
        try:
            import joblib
            self.clf = joblib.load(path)
            logger.info("Modelo carregado de: %s", path)
        except Exception as e:
            logger.exception("Erro ao carregar modelo: %s", str(e))
            raise
