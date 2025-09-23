from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_images_from_dir(directory: str, limit: Optional[int] = None) -> List[np.ndarray]:
    """
    Carrega imagens de um diretório, retornando uma lista de np.ndarray.

    Args:
        directory (str): Caminho do diretório a ser varrido.
        limit (Optional[int]): Número máximo de imagens a carregar.

    Returns:
        List[np.ndarray]: Lista de imagens carregadas como arrays NumPy.

    Raises:
        ValueError: Se o diretório não existir ou não contiver imagens válidas.
    """
    if not os.path.isdir(directory):
        logger.error("Diretório inválido: %s", directory)
        raise ValueError(f"Diretório inválido: {directory}")

    images: List[np.ndarray] = []
    count = 0

    logger.info("Carregando imagens do diretório: %s", directory)

    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, fn)
                try:
                    img = cv2.imread(path)
                    if img is not None:
                        images.append(img)
                        count += 1
                        logger.debug("Imagem carregada: %s", path)
                    else:
                        logger.warning("Falha ao carregar imagem: %s", path)
                except Exception as e:
                    logger.exception("Erro ao ler imagem %s: %s", path, str(e))

                if limit and count >= limit:
                    logger.info("Limite de %d imagens atingido.", limit)
                    return images

    if not images:
        logger.warning("Nenhuma imagem válida encontrada em: %s", directory)

    logger.info("Total de imagens carregadas: %d", len(images))
    return images


def load_metadata_to_dataframe(directory: str) -> pd.DataFrame:
    """
    Cria DataFrame com caminhos e metadados básicos de imagens.

    Args:
        directory (str): Caminho do diretório contendo imagens.

    Returns:
        pd.DataFrame: DataFrame com colunas [path, height, width, channels].

    Raises:
        ValueError: Se o diretório não existir.
    """
    if not os.path.isdir(directory):
        logger.error("Diretório inválido: %s", directory)
        raise ValueError(f"Diretório inválido: {directory}")

    records: List[Dict[str, Any]] = []

    logger.info("Extraindo metadados de imagens em: %s", directory)

    for root, _, files in os.walk(directory):
        for fn in files:
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, fn)
                try:
                    img = cv2.imread(path)
                    if img is None:
                        logger.warning("Não foi possível ler imagem para metadados: %s", path)
                        continue
                    h, w, c = img.shape
                    records.append(
                        {"path": path, "height": h, "width": w, "channels": c}
                    )
                    logger.debug("Metadados extraídos: %s", path)
                except Exception as e:
                    logger.exception("Erro ao processar imagem %s: %s", path, str(e))

    df = pd.DataFrame(records)
    logger.info("Total de imagens processadas para metadados: %d", len(df))
    return df
