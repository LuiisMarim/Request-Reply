from dataclasses import dataclass
from typing import Optional
from domain.entities.metrics import Metrics


@dataclass(frozen=True)
class TrainRequest:
    """Request DTO para treino de modelos."""

    features_dir: str
    models_dir: str
    profile: str = "medium"
    enable_xai: bool = False
    audit_fairness: bool = False
    protocol: str = "loso"  # ou "kfold"


@dataclass(frozen=True)
class TrainResponse:
    """Response DTO para treino de modelos."""

    success: bool
    model_path: str
    metrics: Optional[Metrics] = None
    fairness_report_path: Optional[str] = None
    xai_report_path: Optional[str] = None
