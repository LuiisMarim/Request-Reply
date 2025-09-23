import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionUnits:
    """
    Representa intensidades das Action Units (AUs) faciais.

    Attributes:
        intensities (Dict[str, float]): Mapeamento de nome da AU para intensidade (0–100).
    """

    intensities: Dict[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.intensities, dict):
            logger.error("Tipo inválido para intensities: %s", type(self.intensities))
            raise ValueError("O campo 'intensities' deve ser um dicionário.")

        for au, val in self.intensities.items():
            if not isinstance(au, str):
                logger.error("Chave inválida no dicionário de AUs: %s", au)
                raise ValueError("As chaves de 'intensities' devem ser strings.")
            if not isinstance(val, (int, float)):
                logger.error("Valor inválido para AU '%s': %s", au, val)
                raise ValueError("Os valores de 'intensities' devem ser numéricos.")

        logger.info("ActionUnits inicializado com %d entradas.", len(self.intensities))

    def get_intensity(self, au: str) -> float:
        """
        Retorna a intensidade de uma AU específica.

        Args:
            au (str): Nome da Action Unit (ex.: 'AU01').

        Returns:
            float: Intensidade associada (0–100), ou 0.0 se não existir.
        """
        intensity = self.intensities.get(au, 0.0)
        logger.debug("Consultando intensidade da AU '%s': %.2f", au, intensity)
        return intensity

    def normalize(self) -> "ActionUnits":
        """
        Normaliza as intensidades para o intervalo 0–100.

        Qualquer valor abaixo de 0 será truncado para 0, e valores acima de 100 serão truncados para 100.

        Returns:
            ActionUnits: Nova instância com intensidades normalizadas.
        """
        normed = {}
        for au, val in self.intensities.items():
            if val < 0.0 or val > 100.0:
                logger.warning("Valor fora do intervalo encontrado para AU '%s': %.2f", au, val)
            normed[au] = max(0.0, min(100.0, val))

        logger.info("Normalização concluída. %d AUs ajustadas.", len(normed))
        return ActionUnits(intensities=normed)
