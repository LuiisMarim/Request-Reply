from application.dto.train_dto import TrainRequest, TrainResponse
from domain.services.classifier import IClassifier
from domain.services.explainer import IXAIExplainer
from domain.services.fairness_auditor import IFairnessAuditor
from typing import Optional


class TrainUseCase:
    """Caso de uso: treino de modelo com auditoria e XAI."""

    def __init__(
        self,
        classifier: IClassifier,
        explainer: Optional[IXAIExplainer] = None,
        fairness_auditor: Optional[IFairnessAuditor] = None,
    ):
        self._classifier = classifier
        self._explainer = explainer
        self._fairness_auditor = fairness_auditor

    def execute(self, request: TrainRequest) -> TrainResponse:
        # Treino
        X, y = self._load_features(request.features_dir)
        self._classifier.train(X, y)
        metrics = self._classifier.evaluate(X, y)

        fairness_report_path = None
        xai_report_path = None

        if request.audit_fairness and self._fairness_auditor:
            fairness_report_path = f"{request.models_dir}/fairness_report.html"

        if request.enable_xai and self._explainer:
            xai_report_path = f"{request.models_dir}/xai_report.html"

        model_path = f"{request.models_dir}/model.pkl"
        return TrainResponse(
            success=True,
            model_path=model_path,
            metrics=metrics,
            fairness_report_path=fairness_report_path,
            xai_report_path=xai_report_path,
        )

    def _load_features(self, features_dir: str):
        """Carrega features do disco (stub; será implementado em infrastructure)."""
        # Placeholder
        return [], []
