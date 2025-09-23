import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FairnessReport:
    """
    Relatório de fairness por subgrupo.

    Attributes:
        metrics (Dict[str, Dict[str, float]]): Mapeamento de subgrupo para métricas e valores numéricos.
    """

    metrics: Dict[str, Dict[str, float]]

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, dict):
            logger.error("Tipo inválido para metrics: %s", type(self.metrics))
            raise ValueError("O campo 'metrics' deve ser um dicionário.")

        for subgroup, subgroup_metrics in self.metrics.items():
            if not isinstance(subgroup, str):
                logger.error("Subgrupo inválido no relatório de fairness: %s", subgroup)
                raise ValueError("As chaves de 'metrics' devem ser strings representando subgrupos.")

            if not isinstance(subgroup_metrics, dict):
                logger.error("Métricas inválidas para subgrupo '%s': %s", subgroup, subgroup_metrics)
                raise ValueError("Cada subgrupo deve mapear para um dicionário de métricas.")

            for metric, value in subgroup_metrics.items():
                if not isinstance(metric, str):
                    logger.error("Nome de métrica inválido no subgrupo '%s': %s", subgroup, metric)
                    raise ValueError("Os nomes de métricas devem ser strings.")
                if not isinstance(value, (int, float)):
                    logger.error("Valor inválido para métrica '%s' no subgrupo '%s': %s", metric, subgroup, value)
                    raise ValueError("Os valores de métricas devem ser numéricos.")

        logger.info("FairnessReport inicializado com %d subgrupos.", len(self.metrics))

    def get_metric(self, subgroup: str, metric: str) -> float:
        """
        Retorna o valor de uma métrica para um subgrupo específico.

        Args:
            subgroup (str): Nome do subgrupo.
            metric (str): Nome da métrica.

        Returns:
            float: Valor da métrica ou NaN se não encontrada.
        """
        value = self.metrics.get(subgroup, {}).get(metric, float("nan"))
        if value != value:  # NaN check
            logger.warning("Métrica '%s' não encontrada para o subgrupo '%s'.", metric, subgroup)
        else:
            logger.debug("Métrica '%s' para subgrupo '%s': %.3f", metric, subgroup, value)
        return value
