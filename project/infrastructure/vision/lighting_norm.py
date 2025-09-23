from __future__ import annotations

import cv2
import numpy as np
import logging

from domain.services.lighting_normalizer import ILightingNormalizer

logger = logging.getLogger(__name__)


class CLAHELightingNormalizer(ILightingNormalizer):
    """
    Normaliza iluminação usando CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Estratégia:
        - Converte a imagem para o espaço de cor LAB.
        - Aplica CLAHE apenas no canal L (luminância).
        - Reconverte para BGR.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> None:
        if clip_limit <= 0:
            logger.error("clip_limit inválido: %s", clip_limit)
            raise ValueError("clip_limit deve ser maior que 0.")
        if not isinstance(tile_grid_size, tuple) or len(tile_grid_size) != 2:
            logger.error("tile_grid_size inválido: %s", tile_grid_size)
            raise ValueError("tile_grid_size deve ser uma tupla (x, y).")

        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        logger.info("CLAHELightingNormalizer inicializado (clip_limit=%.2f, grid=%s)", clip_limit, tile_grid_size)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica normalização de iluminação a uma imagem.

        Args:
            image (np.ndarray): Imagem de entrada (H, W, 3) em BGR.

        Returns:
            np.ndarray: Imagem normalizada.

        Raises:
            ValueError: Se a imagem de entrada for inválida.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            logger.error("Imagem inválida fornecida: %s", type(image))
            raise ValueError("A entrada deve ser uma imagem NumPy colorida (H, W, 3).")

        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            merged = cv2.merge((l2, a, b))
            normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            logger.debug("Normalização CLAHE aplicada com sucesso.")
            return normalized
        except Exception as e:
            logger.exception("Falha na normalização de iluminação: %s", str(e))
            return image
