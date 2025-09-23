from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict

import numpy as np
import pandas as pd
import logging

from domain.services.au_extractor import IAUExtractor

logger = logging.getLogger(__name__)


class OpenFaceAUExtractor(IAUExtractor):
    """
    Extrator de Action Units (AUs) usando o binário OpenFace.

    Este extrator chama o executável `FeatureExtraction` do OpenFace para processar
    uma imagem, parseia o CSV de saída e retorna as intensidades das AUs.
    """

    def __init__(self, openface_bin: str = "FeatureExtraction") -> None:
        """
        Inicializa o extrator OpenFace.

        Args:
            openface_bin (str): Caminho para o binário `FeatureExtraction` do OpenFace.
        """
        self.openface_bin = openface_bin

    def extract(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrai intensidades das AUs a partir de uma imagem.

        Args:
            image (np.ndarray): Imagem de entrada no formato NumPy (H, W, C).

        Returns:
            Dict[str, float]: Dicionário AU -> intensidade.

        Raises:
            ValueError: Se a imagem for inválida.
            RuntimeError: Se falhar a execução do OpenFace ou parsing.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            logger.error("Imagem inválida fornecida para AUExtractor: %s", type(image))
            raise ValueError("A entrada deve ser uma imagem NumPy com 3 dimensões (H, W, C).")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.png")
            out_csv = os.path.join(tmpdir, "output.csv")

            try:
                import cv2
                cv2.imwrite(img_path, image)
                logger.debug("Imagem temporária salva em: %s", img_path)
            except Exception as e:
                logger.exception("Falha ao salvar imagem temporária: %s", str(e))
                raise RuntimeError(f"Falha ao salvar imagem temporária: {str(e)}") from e

            cmd = [
                self.openface_bin,
                "-f", img_path,
                "-out_dir", tmpdir,
                "-aus"
            ]
            logger.info("Executando OpenFace: %s", " ".join(cmd))

            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error("OpenFace falhou: %s", e.stderr.decode("utf-8", errors="ignore"))
                raise RuntimeError(f"Falha ao executar OpenFace: {e.stderr.decode('utf-8', errors='ignore')}") from e
            except Exception as e:
                logger.exception("Erro inesperado ao executar OpenFace: %s", str(e))
                raise RuntimeError(f"Erro inesperado ao executar OpenFace: {str(e)}") from e

            if not os.path.exists(out_csv):
                logger.error("Arquivo de saída do OpenFace não encontrado: %s", out_csv)
                raise RuntimeError("OpenFace não gerou arquivo de saída esperado.")

            try:
                df = pd.read_csv(out_csv)
                au_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
                if df.empty or not au_cols:
                    logger.warning("Nenhuma AU encontrada no CSV gerado.")
                    return {}
                intensities = df[au_cols].iloc[-1].to_dict()
                intensities = {k.replace("_r", ""): float(v) for k, v in intensities.items()}
                logger.info("Extração de AUs concluída com sucesso.")
                return intensities
            except Exception as e:
                logger.exception("Falha ao parsear saída do OpenFace: %s", str(e))
                raise RuntimeError(f"Falha ao parsear saída do OpenFace: {str(e)}") from e
