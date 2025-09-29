from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict

import numpy as np
import pandas as pd
import logging
import cv2
import glob

from domain.services.au_extractor import IAUExtractorService

logger = logging.getLogger(__name__)


class OpenFaceAUExtractor(IAUExtractorService):
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

    def extract(self, input_dir: str, output_dir: str, profile: str) -> bool:
        """
        Extrai AUs de todas as imagens em input_dir e salva no output_dir.

        Args:
            input_dir (str): Diretório com imagens.
            output_dir (str): Diretório para salvar features extraídas.
            profile (str): Perfil de execução (não utilizado aqui).

        Returns:
            bool: True se todas as extrações foram bem-sucedidas.
        """
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        image_paths = glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True) + \
                      glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
        success = True

        for img_path in image_paths:
            try:
                image = cv2.imread(img_path)
                aus = self.extract_single(image, os.path.basename(img_path))
                out_csv = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".csv")
                pd.DataFrame([aus]).to_csv(out_csv, index=False)
            except Exception as e:
                logger.error("Falha ao extrair AUs de %s: %s", img_path, str(e))
                success = False

        return success
    
    
    def extract_single(self, image: np.ndarray, image_name: str = "input.png") -> Dict[str, float]:
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
            try:
                img_path = os.path.join(tmpdir, image_name)
                cv2.imwrite(img_path, image)
                out_csv = os.path.join(tmpdir, os.path.splitext(image_name)[0] + ".csv")
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
