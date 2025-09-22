from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from domain.services.microexpression_extractor import IMicroExpressionExtractor


class OpticalFlowMicroExpressionExtractor(IMicroExpressionExtractor):
    """Extrai microexpressões usando Optical Flow (TV-L1)."""

    def __init__(self) -> None:
        self.of = cv2.optflow.DualTVL1OpticalFlow_create()

    def extract(self, frames: list[np.ndarray]) -> Dict[str, float]:
        if len(frames) < 2:
            return {}

        flows = []
        for i in range(len(frames) - 1):
            prev = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            flow = self.of.calc(prev, nxt, None)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flows.append(np.mean(mag))

        return {"microexpression_intensity": float(np.mean(flows))}
