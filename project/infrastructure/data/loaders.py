from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd


def load_images_from_dir(directory: str, limit: int | None = None) -> List[np.ndarray]:
    """Carrega imagens de um diretório, retornando lista de np.ndarray."""
    images: List[np.ndarray] = []
    count = 0
    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, fn)
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    count += 1
                if limit and count >= limit:
                    return images
    return images


def load_metadata_to_dataframe(directory: str) -> pd.DataFrame:
    """Cria DataFrame com caminhos e metadados básicos de imagens."""
    records: List[Dict[str, Any]] = []
    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, fn)
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w, c = img.shape
                records.append(
                    {"path": path, "height": h, "width": w, "channels": c}
                )
    return pd.DataFrame(records)
