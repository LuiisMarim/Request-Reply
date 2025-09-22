from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np


class DataAugmenter:
    """Aplica augmentations leves para balanceamento e robustez."""

    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def random_flip(self, image: Any) -> Any:
        """Flip horizontal com probabilidade 0.5."""
        if random.random() < 0.5:
            return cv2.flip(image, 1)
        return image

    def random_blur(self, image: Any) -> Any:
        """Aplica blur gaussiano leve com probabilidade 0.3."""
        if random.random() < 0.3:
            k = random.choice([3, 5])
            return cv2.GaussianBlur(image, (k, k), 0)
        return image

    def color_jitter(self, image: Any) -> Any:
        """Perturba brilho/contraste levemente."""
        if random.random() < 0.3:
            alpha = 1.0 + (0.1 * (random.random() - 0.5))  # contraste
            beta = 10 * (random.random() - 0.5)  # brilho
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image

    def augment(self, image: Any) -> Any:
        """Aplica sequência de augmentations."""
        img = self.random_flip(image)
        img = self.random_blur(img)
        img = self.color_jitter(img)
        return img
