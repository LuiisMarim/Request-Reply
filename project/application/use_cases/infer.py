from application.dto.infer_dto import InferRequest, InferResponse
from domain.services.classifier import IClassifier


class InferUseCase:
    """Caso de uso: inferência em novos dados."""

    def __init__(self, classifier: IClassifier):
        self._classifier = classifier

    def execute(self, request: InferRequest) -> InferResponse:
        X = self._load_features(request.input_dir)
        preds = self._classifier.predict(X)

        results_path = f"{request.output_dir}/results.csv"
        report_path = f"{request.output_dir}/inference_report.html"

        return InferResponse(
            success=True,
            results_path=results_path,
            report_path=report_path,
        )

    def _load_features(self, input_dir: str):
        """Carrega features do disco (stub; será implementado em infrastructure)."""
        return []
