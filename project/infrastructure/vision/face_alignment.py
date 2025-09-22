from __future__ import annotations

from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np

from domain.services.face_aligner import IFaceAligner


class EyeCornerFaceAligner(IFaceAligner):
    """Alinha rostos baseado nos cantos dos olhos (MediaPipe Face Mesh)."""

    def __init__(self) -> None:
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    def align(self, image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        if not results.multi_face_landmarks:
            return image

        h, w, _ = image.shape
        lm = results.multi_face_landmarks[0].landmark
        left = (int(lm[33].x * w), int(lm[33].y * h))  # canto olho esquerdo
        right = (int(lm[263].x * w), int(lm[263].y * h))  # canto olho direito

        dx, dy = right[0] - left[0], right[1] - left[1]
        angle = np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D(center=left, angle=angle, scale=1.0)
        return cv2.warpAffine(image, M, (w, h))
