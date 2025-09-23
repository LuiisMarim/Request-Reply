from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataAugmenter:
    """
    Aplica augmentations leves para balanceamento e robustez do modelo.

    As transformações incluem:
        - Flip horizontal
        - Blur gaussiano
        - Perturbações de brilho e contraste
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Inicializa o DataAugmenter com semente fixa para reprodutibilidade.

        Args:
            seed (int): Valor da semente para geração de números aleatórios.
        """
        random.seed(seed)
        np.random.seed(seed)
        logger.info("DataAugmenter inicializado com seed=%d", seed)

    def _validate_image(self, image: Any) -> None:
        """Valida se a entrada é uma imagem NumPy válida."""
        if not isinstance(image, np.ndarray):
            logger.error("Entrada inválida para augmentação: %s", type(image))
            raise ValueError("A entrada deve ser uma imagem representada como np.ndarray.")

        if image.ndim < 2:
            logger.error("Imagem inválida: dimensões insuficientes %s", image.shape)
            raise ValueError("A imagem deve ter pelo menos 2 dimensões.")

    def random_flip(self, image: Any) -> Any:
        """
        Realiza flip horizontal na imagem com probabilidade 0.5.

        Args:
            image (Any): Imagem de entrada.

        Returns:
            Any: Imagem processada.
        """
        self._validate_image(image)
        if random.random() < 0.5:
            logger.debug("Flip horizontal aplicado.")
            return cv2.flip(image, 1)
        return image

    def random_blur(self, image: Any) -> Any:
        """
        Aplica blur gaussiano leve na imagem com probabilidade 0.3.

        Args:
            image (Any): Imagem de entrada.

        Returns:
            Any: Imagem processada.
        """
        self._validate_image(image)
        if random.random() < 0.3:
            k = random.choice([3, 5])
            logger.debug("Blur gaussiano aplicado com kernel=%d.", k)
            return cv2.GaussianBlur(image, (k, k), 0)
        return image

    def color_jitter(self, image: Any) -> Any:
        """
        Perturba levemente brilho e contraste da imagem com probabilidade 0.3.

        Args:
            image (Any): Imagem de entrada.

        Returns:
            Any: Imagem processada.
        """
        self._validate_image(image)
        if random.random() < 0.3:
            alpha = 1.0 + (0.1 * (random.random() - 0.5))  # contraste
            beta = 10 * (random.random() - 0.5)  # brilho
            logger.debug("Color jitter aplicado (alpha=%.3f, beta=%.3f).", alpha, beta)
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return image

    def augment(self, image: Any) -> Any:
        """
        Aplica sequência de augmentations à imagem.

        Args:
            image (Any): Imagem de entrada.

        Returns:
            Any: Imagem processada após augmentations.
        """
        self._validate_image(image)
        logger.info("Aplicando sequência de augmentations.")
        img = self.random_flip(image)
        img = self.random_blur(img)
        img = self.color_jitter(img)
        logger.info("Augmentations concluídas.")
        return img
