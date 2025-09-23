import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MicroExpression:
    """
    Representa uma microexpressão detectada em uma sequência curta.

    Attributes:
        start_frame (int): Quadro inicial da microexpressão.
        end_frame (int): Quadro final da microexpressão.
        features (Dict[str, float]): Atributos associados, como intensidade ou fluxo óptico.
    """

    start_frame: int
    end_frame: int
    features: Dict[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.start_frame, int) or self.start_frame < 0:
            logger.error("Valor inválido para start_frame: %s", self.start_frame)
            raise ValueError("O campo 'start_frame' deve ser um inteiro não negativo.")

        if not isinstance(self.end_frame, int) or self.end_frame <= self.start_frame:
            logger.error("Valor inválido para end_frame: %s", self.end_frame)
            raise ValueError("O campo 'end_frame' deve ser um inteiro maior que 'start_frame'.")

        if not isinstance(self.features, dict):
            logger.error("Tipo inválido para features: %s", type(self.features))
            raise ValueError("O campo 'features' deve ser um dicionário.")

        for key, value in self.features.items():
            if not isinstance(key, str):
                logger.error("Chave inválida em features: %s", key)
                raise ValueError("As chaves de 'features' devem ser strings.")
            if not isinstance(value, (int, float)):
                logger.error("Valor inválido para feature '%s': %s", key, value)
                raise ValueError("Os valores de 'features' devem ser numéricos.")

        logger.info(
            "MicroExpression inicializada: start_frame=%d, end_frame=%d, duração=%d frames",
            self.start_frame,
            self.end_frame,
            self.duration(),
        )

    def duration(self) -> int:
        """
        Calcula a duração da microexpressão em frames.

        Returns:
            int: Número de frames entre start_frame e end_frame.
        """
        duration_frames = self.end_frame - self.start_frame
        logger.debug("Duração calculada da microexpressão: %d frames", duration_frames)
        return duration_frames
