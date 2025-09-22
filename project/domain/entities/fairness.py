from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class FairnessReport:
    """Relatório de fairness por subgrupo."""

    metrics: Dict[str, Dict[str, float]]  # subgrupo -> métrica -> valor

    def get_metric(self, subgroup: str, metric: str) -> float:
        """Retorna valor de uma métrica para um subgrupo específico."""
        return self.metrics.get(subgroup, {}).get(metric, float("nan"))
