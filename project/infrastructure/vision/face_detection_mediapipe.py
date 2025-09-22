from __future__ import annotations

from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from domain.services.facial_detector import IFacialDetector


class MediaPipeFaceDetector(IFacialDetector):
    """Detecção facial com MediaPipe Face Detection (GPU-friendly)."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence
        )

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Retorna bounding boxes [x, y, w, h] em pixels."""
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
        return boxes
