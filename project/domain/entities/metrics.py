from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Metrics:
    """Métricas de desempenho em treino/validação/teste."""

    accuracy: float
    auc: float
    f1_score: float
    per_group: Dict[str, Dict[str, float]]  # métricas por subgrupo demográfico

    def summary(self) -> Dict[str, float]:
        """Resumo das métricas principais."""
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "f1_score": self.f1_score,
        }
