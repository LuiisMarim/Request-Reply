import os
import logging
import pandas as pd
from typing import Optional
from application.dto.train_dto import TrainRequest, TrainResponse
from domain.services.classifier import IClassifier
from domain.services.explainer import IXAIExplainer
from domain.services.fairness_auditor import IFairnessAuditor

logger = logging.getLogger(__name__)


class TrainUseCase:
    """
    Caso de uso: treino de modelo com auditoria e XAI.

    Responsável por carregar features, treinar modelo, avaliar desempenho
    e opcionalmente executar auditoria de fairness e gerar relatórios de XAI.
    """

    def __init__(
        self,
        classifier: IClassifier,
        explainer: Optional[IXAIExplainer] = None,
        fairness_auditor: Optional[IFairnessAuditor] = None,
    ):
        if not isinstance(classifier, IClassifier):
            logger.error("Classificador inválido fornecido: %s", type(classifier))
            raise ValueError("O parâmetro 'classifier' deve implementar IClassifier.")

        if explainer is not None and not isinstance(explainer, IXAIExplainer):
            logger.error("Explainer inválido fornecido: %s", type(explainer))
            raise ValueError("O parâmetro 'explainer' deve implementar IXAIExplainer.")

        if fairness_auditor is not None and not isinstance(fairness_auditor, IFairnessAuditor):
            logger.error("FairnessAuditor inválido fornecido: %s", type(fairness_auditor))
            raise ValueError("O parâmetro 'fairness_auditor' deve implementar IFairnessAuditor.")

        self._classifier = classifier
        self._explainer = explainer
        self._fairness_auditor = fairness_auditor

    def execute(self, request: TrainRequest) -> TrainResponse:
        """
        Executa o processo de treino do modelo.

        Args:
            request (TrainRequest): DTO contendo diretórios, perfil e opções de auditoria/XAI.

        Returns:
            TrainResponse: Resultado do treino, incluindo métricas, caminho do modelo e relatórios.
        """
        if not isinstance(request, TrainRequest):
            logger.error("Tipo inválido para request: %s", type(request))
            raise ValueError("O parâmetro 'request' deve ser uma instância de TrainRequest.")

        logger.info("Iniciando processo de treino com dados em '%s'.", request.features_dir)

        try:
            # Carregar features
            X, y = self._load_features(request.features_dir)
            logger.info("Features e labels carregadas com sucesso (%d amostras).", len(X))

            # Treinar modelo
            self._classifier.train(X, y)
            logger.info("Treinamento do modelo concluído.")

            # Avaliar modelo
            metrics = self._classifier.evaluate(X, y)
            logger.info("Avaliação do modelo concluída. Métricas: %s", metrics)

            fairness_report_path = None
            xai_report_path = None

            if request.audit_fairness and self._fairness_auditor:
                fairness_report_path = os.path.join(request.models_dir, "fairness_report.html")
                try:
                    self._fairness_auditor.audit(X, y, fairness_report_path)
                    logger.info("Relatório de fairness gerado em: %s", fairness_report_path)
                except Exception as e:
                    logger.warning("Falha ao gerar relatório de fairness: %s", str(e))
                    fairness_report_path = None

            if request.enable_xai and self._explainer:
                xai_report_path = os.path.join(request.models_dir, "xai_report.html")
                try:
                    self._explainer.explain(X, y, xai_report_path)
                    logger.info("Relatório de XAI gerado em: %s", xai_report_path)
                except Exception as e:
                    logger.warning("Falha ao gerar relatório de XAI: %s", str(e))
                    xai_report_path = None

            model_path = os.path.join(request.models_dir, "model.pkl")
            try:
                self._classifier.save(model_path)
                logger.info("Modelo salvo em: %s", model_path)
            except Exception as e:
                logger.error("Falha ao salvar modelo em %s: %s", model_path, str(e))
                return TrainResponse(success=False, model_path="", metrics=None)

            return TrainResponse(
                success=True,
                model_path=model_path,
                metrics=metrics,
                fairness_report_path=fairness_report_path,
                xai_report_path=xai_report_path,
            )

        except Exception as e:
            logger.exception("Erro durante execução do treino: %s", str(e))
            return TrainResponse(success=False, model_path="", metrics=None)

    def _load_features(self, features_dir: str):
        """
        Carrega features e labels do disco a partir de arquivos CSV.

        Args:
            features_dir (str): Diretório contendo arquivos `features.csv` e `labels.csv`.

        Returns:
            tuple: (X, y) onde X é um DataFrame de features e y é uma Série de labels.
        """
        features_path = os.path.join(features_dir, "features.csv")
        labels_path = os.path.join(features_dir, "labels.csv")

        if not os.path.isfile(features_path):
            logger.error("Arquivo de features não encontrado: %s", features_path)
            raise FileNotFoundError(f"Arquivo de features não encontrado: {features_path}")

        if not os.path.isfile(labels_path):
            logger.error("Arquivo de labels não encontrado: %s", labels_path)
            raise FileNotFoundError(f"Arquivo de labels não encontrado: {labels_path}")

        try:
            X = pd.read_csv(features_path)
            y = pd.read_csv(labels_path).squeeze("columns")
            return X, y
        except Exception as e:
            logger.exception("Erro ao carregar features/labels: %s", str(e))
            raise
