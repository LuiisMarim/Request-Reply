from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from domain.entities.metrics import Metrics
from domain.services.classifier import IClassifier


class RandomForestClassifierWrapper(IClassifier):
    """Wrapper de RandomForest (scikit-learn) compatível com IClassifier.

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
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.calibration = calibration
        self.clf: Any = self.rf  # substituído após calibração

    def _ensure_calibrated(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.calibration is None:
            self.clf = self.rf.fit(X, y)
        else:
            self.clf = CalibratedClassifierCV(self.rf, method=self.calibration, cv=5)
            self.clf = self.clf.fit(X, y)

    def train(self, X: Any, y: Any) -> None:
        X_np, y_np = self._to_numpy(X, y)
        self._ensure_calibrated(X_np, y_np)

    def predict(self, X: Any) -> Any:
        X_np = self._to_numpy(X)
        # retorna probabilidade de classe positiva (autistic=1, por exemplo)
        if hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(X_np)[:, 1]
            return (proba >= 0.5).astype(int)
        return self.clf.predict(X_np)

    def evaluate(self, X: Any, y: Any) -> Metrics:
        X_np, y_np = self._to_numpy(X, y)
        if hasattr(self.clf, "predict_proba"):
            proba = self.clf.predict_proba(X_np)[:, 1]
        else:
            # fallback: usa decisão binária como proxy para proba
            proba = self.clf.predict(X_np)
        y_pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_np, y_pred)
        try:
            auc = roc_auc_score(y_np, proba)
        except Exception:
            auc = float("nan")
        f1 = f1_score(y_np, y_pred)

        per_group: Dict[str, Dict[str, float]] = {}
        return Metrics(accuracy=acc, auc=auc, f1_score=f1, per_group=per_group)

    @staticmethod
    def _to_numpy(X: Any, y: Optional[Any] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_np = np.asarray(X)
        y_np = None if y is None else np.asarray(y).astype(int)
        return X_np, y_np
