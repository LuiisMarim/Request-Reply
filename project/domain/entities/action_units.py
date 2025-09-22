from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ActionUnits:
    """Representa intensidades das Action Units (AUs) faciais."""

    intensities: Dict[str, float]  # AU -> intensidade [0–100]

    def get_intensity(self, au: str) -> float:
        """Retorna intensidade de uma AU, ou 0 se não existir."""
        return self.intensities.get(au, 0.0)

    def normalize(self) -> "ActionUnits":
        """Normaliza intensidades para 0–100 (caso já não estejam calibradas)."""
        normed = {
            au: max(0.0, min(100.0, val)) for au, val in self.intensities.items()
        }
        return ActionUnits(intensities=normed)
