from __future__ import annotations

from typing import Dict, List
import logging

import cv2
import numpy as np

from domain.services.microexpression_extractor import IMicroExpressionExtractor

logger = logging.getLogger(__name__)


class OpticalFlowMicroExpressionExtractor(IMicroExpressionExtractor):
    """
    Extrai microexpressões usando Optical Flow (TV-L1).

    Estratégia:
        - Converte frames consecutivos para escala de cinza.
        - Calcula fluxo óptico com TV-L1.
        - Extrai magnitude média como intensidade de microexpressão.
    """

    def __init__(self) -> None:
        self.of = cv2.optflow.DualTVL1OpticalFlow_create()
        logger.info("OpticalFlowMicroExpressionExtractor inicializado.")

    def extract(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Extrai intensidade de microexpressões de uma sequência de frames.

        Args:
            frames (List[np.ndarray]): Lista de frames (H, W, 3) em BGR.

        Returns:
            Dict[str, float]: Intensidade média da microexpressão.
        """
        if not isinstance(frames, list) or not all(isinstance(f, np.ndarray) for f in frames):
            logger.error("Frames inválidos fornecidos: %s", type(frames))
            raise ValueError("frames deve ser uma lista de np.ndarray.")

        if len(frames) < 2:
            logger.warning("Sequência de frames insuficiente para extração.")
            return {}

        flows = []
        try:
            for i in range(len(frames) - 1):
                prev, nxt = frames[i], frames[i + 1]
                if prev.ndim != 3 or nxt.ndim != 3 or prev.shape[2] != 3 or nxt.shape[2] != 3:
                    logger.error("Frames inválidos para optical flow: shapes %s, %s", prev.shape, nxt.shape)
                    raise ValueError("Todos os frames devem ser imagens coloridas (H, W, 3).")

                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
                flow = self.of.calc(prev_gray, nxt_gray, None)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flows.append(np.mean(mag))

            intensity = {"microexpression_intensity": float(np.mean(flows))}
            logger.info("Extração de microexpressão concluída. Intensidade=%.4f", intensity["microexpression_intensity"])
            return intensity
        except Exception as e:
            logger.exception("Falha na extração de microexpressões: %s", str(e))
            return {}
