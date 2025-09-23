from __future__ import annotations

from typing import Tuple
import logging

import cv2
import mediapipe as mp
import numpy as np

from domain.services.face_aligner import IFaceAligner

logger = logging.getLogger(__name__)


class EyeCornerFaceAligner(IFaceAligner):
    """
    Alinha rostos baseado nos cantos dos olhos usando MediaPipe Face Mesh.

    Estratégia:
        - Detecta landmarks faciais com MediaPipe.
        - Usa os cantos internos dos olhos (landmarks 33 e 263).
        - Calcula o ângulo de rotação e alinha a face.
    """

    def __init__(self) -> None:
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

    def align(self, image: np.ndarray) -> np.ndarray:
        """
        Alinha a face em uma imagem com base nos cantos dos olhos.

        Args:
            image (np.ndarray): Imagem de entrada (H, W, 3) em BGR.

        Returns:
            np.ndarray: Imagem alinhada.

        Raises:
            ValueError: Se a imagem for inválida.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            logger.error("Imagem inválida para alinhamento: %s", type(image))
            raise ValueError("A entrada deve ser uma imagem NumPy colorida (H, W, 3).")

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mesh.process(rgb)
            if not results.multi_face_landmarks:
                logger.warning("Nenhum rosto detectado na imagem, retornando original.")
                return image

            h, w, _ = image.shape
            lm = results.multi_face_landmarks[0].landmark
            left = (int(lm[33].x * w), int(lm[33].y * h))   # canto olho esquerdo
            right = (int(lm[263].x * w), int(lm[263].y * h))  # canto olho direito

            dx, dy = right[0] - left[0], right[1] - left[1]
            angle = np.degrees(np.arctan2(dy, dx))
            logger.debug("Ângulo calculado para alinhamento: %.2f graus", angle)

            M = cv2.getRotationMatrix2D(center=left, angle=angle, scale=1.0)
            aligned = cv2.warpAffine(image, M, (w, h))
            logger.info("Alinhamento facial concluído com sucesso.")
            return aligned
        except Exception as e:
            logger.exception("Falha no alinhamento facial: %s", str(e))
            return image
