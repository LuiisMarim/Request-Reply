from __future__ import annotations

import cv2
import numpy as np

from domain.services.lighting_normalizer import ILightingNormalizer


class CLAHELightingNormalizer(ILightingNormalizer):
    """Normaliza iluminação com CLAHE moderado."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self.clahe.apply(l)
        merged = cv2.merge((l2, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
