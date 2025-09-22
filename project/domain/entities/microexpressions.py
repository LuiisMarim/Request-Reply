from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MicroExpression:
    """Representa uma microexpressão detectada em uma sequência curta."""

    start_frame: int
    end_frame: int
    features: Dict[str, float]  # ex: intensidade, direção fluxo óptico

    def duration(self) -> int:
        """Duração em frames."""
        return self.end_frame - self.start_frame
