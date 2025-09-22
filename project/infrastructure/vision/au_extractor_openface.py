from __future__ import annotations

from typing import Dict

import numpy as np

from domain.services.au_extractor import IAUExtractor


class OpenFaceAUExtractor(IAUExtractor):
    """Extrator de AUs usando OpenFace (placeholder chamando binário externo).

    Obs.: Aqui apenas simulamos. Na prática, chamaria o executável do OpenFace
    e parsearia o CSV gerado.
    """

    def __init__(self, openface_bin: str = "FeatureExtraction") -> None:
        self.openface_bin = openface_bin

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        # Placeholder: retornando intensidades dummy
        return {"AU01": 0.0, "AU12": 0.0, "AU25": 0.0}
