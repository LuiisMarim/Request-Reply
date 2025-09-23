from dataclasses import dataclass
from typing import Optional
import os
import logging
from domain.entities.metrics import Metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainRequest:
    """
    Request DTO para treino de modelos.

    Attributes:
        features_dir (str): Caminho para o diretório contendo as features extraídas.
        models_dir (str): Caminho para o diretório onde os modelos treinados serão salvos.
        profile (str): Perfil de execução (ex.: 'medium', 'fast', 'accurate').
        enable_xai (bool): Se True, habilita geração de relatórios de interpretabilidade (XAI).
        audit_fairness (bool): Se True, habilita auditoria de fairness.
        protocol (str): Protocolo de validação cruzada. Valores aceitos: "loso" ou "kfold".
    """

    features_dir: str
    models_dir: str
    profile: str = "medium"
    enable_xai: bool = False
    audit_fairness: bool = False
    protocol: str = "loso"  # ou "kfold"

    def __post_init__(self) -> None:
        if not self.features_dir or not os.path.isdir(self.features_dir):
            logger.error("Diretório de features inválido: %s", self.features_dir)
            raise ValueError(f"Diretório de features inválido: {self.features_dir}")

        if not self.models_dir:
            logger.error("Diretório de modelos não fornecido.")
            raise ValueError("Diretório de modelos não pode ser vazio.")

        if not isinstance(self.profile, str) or not self.profile.strip():
            logger.error("Perfil de execução inválido: %s", self.profile)
            raise ValueError("Perfil de execução inválido.")

        if not isinstance(self.enable_xai, bool):
            logger.error("Tipo inválido para enable_xai: %s", type(self.enable_xai))
            raise ValueError("O campo 'enable_xai' deve ser booleano.")

        if not isinstance(self.audit_fairness, bool):
            logger.error("Tipo inválido para audit_fairness: %s", type(self.audit_fairness))
            raise ValueError("O campo 'audit_fairness' deve ser booleano.")

        if self.protocol not in ("loso", "kfold"):
            logger.error("Protocolo inválido: %s", self.protocol)
            raise ValueError("O campo 'protocol' deve ser 'loso' ou 'kfold'.")

        logger.info("TrainRequest inicializado com sucesso: %s", self)


@dataclass(frozen=True)
class TrainResponse:
    """
    Response DTO para treino de modelos.

    Attributes:
        success (bool): Indica se o treino foi concluído com sucesso.
        model_path (str): Caminho para o modelo treinado salvo.
        metrics (Optional[Metrics]): Métricas de desempenho obtidas no treino.
        fairness_report_path (Optional[str]): Caminho para o relatório de fairness, se gerado.
        xai_report_path (Optional[str]): Caminho para o relatório de interpretabilidade (XAI), se gerado.
    """

    success: bool
    model_path: str
    metrics: Optional[Metrics] = None
    fairness_report_path: Optional[str] = None
    xai_report_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            logger.error("Valor inválido para success: %s", self.success)
            raise ValueError("O campo 'success' deve ser um booleano.")

        if not self.model_path:
            logger.error("Caminho do modelo não pode ser vazio.")
            raise ValueError("O campo 'model_path' não pode ser vazio.")

        if self.metrics is not None and not isinstance(self.metrics, Metrics):
            logger.error("Tipo inválido para metrics: %s", type(self.metrics))
            raise ValueError("O campo 'metrics' deve ser instância de Metrics ou None.")

        if self.fairness_report_path is not None and not isinstance(self.fairness_report_path, str):
            logger.error("Tipo inválido para fairness_report_path: %s", self.fairness_report_path)
            raise ValueError("O campo 'fairness_report_path' deve ser string ou None.")

        if self.xai_report_path is not None and not isinstance(self.xai_report_path, str):
            logger.error("Tipo inválido para xai_report_path: %s", self.xai_report_path)
            raise ValueError("O campo 'xai_report_path' deve ser string ou None.")

        logger.info("TrainResponse inicializado com sucesso: %s", self)
