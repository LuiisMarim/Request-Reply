import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metrics:
    """
    Métricas de desempenho em treino/validação/teste.

    Attributes:
        accuracy (float): Acurácia global do modelo (0–1).
        auc (float): Área sob a curva ROC (0–1).
        f1_score (float): F1-score global (0–1).
        per_group (Dict[str, Dict[str, float]]): Métricas por subgrupo demográfico.
    """

    accuracy: float
    auc: float
    f1_score: float
    per_group: Dict[str, Dict[str, float]]

    def __post_init__(self) -> None:
        for metric_name, value in {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "f1_score": self.f1_score,
        }.items():
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                logger.error("Valor inválido para %s: %s", metric_name, value)
                raise ValueError(f"O campo '{metric_name}' deve ser numérico entre 0 e 1.")

        if not isinstance(self.per_group, dict):
            logger.error("Tipo inválido para per_group: %s", type(self.per_group))
            raise ValueError("O campo 'per_group' deve ser um dicionário.")

        for group, group_metrics in self.per_group.items():
            if not isinstance(group, str):
                logger.error("Nome de grupo inválido em per_group: %s", group)
                raise ValueError("As chaves de 'per_group' devem ser strings.")
            if not isinstance(group_metrics, dict):
                logger.error("Métricas inválidas para grupo '%s': %s", group, group_metrics)
                raise ValueError("Os valores de 'per_group' devem ser dicionários de métricas.")

            for m_name, m_val in group_metrics.items():
                if not isinstance(m_name, str):
                    logger.error("Nome de métrica inválido no grupo '%s': %s", group, m_name)
                    raise ValueError("Os nomes das métricas devem ser strings.")
                if not isinstance(m_val, (int, float)):
                    logger.error("Valor inválido para métrica '%s' no grupo '%s': %s", m_name, group, m_val)
                    raise ValueError("Os valores das métricas por grupo devem ser numéricos.")

        logger.info("Metrics inicializado: accuracy=%.3f, auc=%.3f, f1_score=%.3f", self.accuracy, self.auc, self.f1_score)

    def summary(self) -> Dict[str, float]:
        """
        Retorna um resumo das métricas principais.

        Returns:
            Dict[str, float]: Dicionário contendo accuracy, auc e f1_score.
        """
        logger.debug("Resumo das métricas principais gerado.")
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "f1_score": self.f1_score,
        }

    def get_group_metrics(self, group: str) -> Dict[str, float]:
        """
        Retorna métricas associadas a um subgrupo específico.

        Args:
            group (str): Nome do subgrupo.

        Returns:
            Dict[str, float]: Dicionário de métricas do subgrupo ou vazio se não encontrado.
        """
        metrics = self.per_group.get(group, {})
        if not metrics:
            logger.warning("Nenhuma métrica encontrada para o grupo '%s'.", group)
        else:
            logger.debug("Métricas do grupo '%s': %s", group, metrics)
        return metrics
