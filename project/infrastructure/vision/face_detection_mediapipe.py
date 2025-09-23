from __future__ import annotations

from typing import List, Tuple
import logging

import cv2
import mediapipe as mp
import numpy as np

from domain.services.facial_detector import IFacialDetector

logger = logging.getLogger(__name__)


class MediaPipeFaceDetector(IFacialDetector):
    """Detecção facial baseada no MediaPipe Face Detection (GPU-friendly)."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        """
        Inicializa o detector MediaPipe.

        Args:
            min_confidence (float): Confiança mínima para considerar uma detecção.
        """
        if not (0.0 < min_confidence <= 1.0):
            logger.error("min_confidence inválido: %s", min_confidence)
            raise ValueError("min_confidence deve estar no intervalo (0, 1].")

        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence
        )
        logger.info("MediaPipeFaceDetector inicializado (min_confidence=%.2f)", min_confidence)

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta faces em uma imagem.

        Args:
            image (np.ndarray): Imagem de entrada em formato BGR (H, W, 3).

        Returns:
            List[Tuple[int, int, int, int]]: Lista de bounding boxes no formato (x, y, w, h).
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            logger.error("Imagem inválida para detecção: %s", type(image))
            raise ValueError("A entrada deve ser uma imagem NumPy colorida (H, W, 3).")

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)
            boxes: List[Tuple[int, int, int, int]] = []

            if results.detections:
                h, w, _ = image.shape
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    bw, bh = int(bbox.width * w), int(bbox.height * h)
                    boxes.append((x, y, bw, bh))

            logger.info("Detecção concluída: %d rosto(s) encontrado(s).", len(boxes))
            return boxes

        except Exception as e:
            logger.exception("Falha durante detecção facial: %s", str(e))
            return []
