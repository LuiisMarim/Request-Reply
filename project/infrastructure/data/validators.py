from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import logging

from infrastructure.utils.errors import DataValidationError

logger = logging.getLogger(__name__)


def validate_image_file(path: str) -> Tuple[bool, Dict[str, str]]:
    """
    Valida uma imagem individual, verificando corrupção, dimensões e canais.

    Args:
        path (str): Caminho para a imagem.

    Returns:
        Tuple[bool, Dict[str, str]]: Indicador de validade e metadados ou erros associados.
    """
    if not os.path.exists(path):
        logger.error("Arquivo não encontrado: %s", path)
        return False, {"error": "Arquivo não encontrado"}

    try:
        img = cv2.imread(path)
        if img is None:
            logger.error("Imagem corrompida ou ilegível: %s", path)
            return False, {"error": "Imagem corrompida"}

        h, w, c = img.shape
        if c not in (1, 3):
            logger.error("Número inválido de canais na imagem %s: %d", path, c)
            return False, {"error": f"Canais inválidos: {c}"}

        logger.debug("Imagem válida: %s (h=%d, w=%d, c=%d)", path, h, w, c)
        return True, {"height": str(h), "width": str(w), "channels": str(c)}

    except Exception as e:
        logger.exception("Erro ao validar imagem %s: %s", path, str(e))
        return False, {"error": str(e)}


def validate_dataset(input_dir: str) -> List[Dict[str, str]]:
    """
    Valida todas as imagens em um diretório.

    Args:
        input_dir (str): Diretório contendo as imagens.

    Returns:
        List[Dict[str, str]]: Lista com resultados de validação por arquivo.

    Raises:
        ValueError: Se o diretório não existir.
    """
    if not os.path.isdir(input_dir):
        logger.error("Diretório inválido para validação: %s", input_dir)
        raise ValueError(f"Diretório inválido: {input_dir}")

    logger.info("Iniciando validação do dataset em: %s", input_dir)

    results: List[Dict[str, str]] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if not fn.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            fpath = os.path.join(root, fn)
            ok, meta = validate_image_file(fpath)
            results.append(
                {
                    "file": fpath,
                    "valid": str(ok),
                    **meta,
                }
            )

    logger.info("Validação concluída. Total de arquivos processados: %d", len(results))
    return results


class DataValidator:
    """Wrapper orientado a objetos para validação de datasets."""

    def validate(self, input_dir: str, output_dir: str, report_path: str) -> bool:
        """
        Executa validação de dataset e gera relatório em HTML.

        Args:
            input_dir (str): Diretório contendo as imagens a validar.
            output_dir (str): Diretório onde o relatório será salvo.
            report_path (str): Caminho completo do relatório de saída.

        Returns:
            bool: True se a validação foi bem-sucedida, False caso contrário.

        Raises:
            DataValidationError: Se ocorrer falha durante a validação.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Gerando relatório de validação em: %s", report_path)

            results = validate_dataset(input_dir)

            with open(report_path, "w", encoding="utf-8") as f:
                f.write("<html><body><h1>Data Quality Report</h1><ul>")
                for r in results:
                    color = "green" if r["valid"] == "True" else "red"
                    f.write(
                        f"<li style='color:{color}'>{r['file']} - valid={r['valid']} {r}</li>"
                    )
                f.write("</ul></body></html>")

            logger.info("Relatório de validação gerado com sucesso em: %s", report_path)
            return True

        except Exception as exc:
            logger.exception("Falha ao validar dataset: %s", str(exc))
            raise DataValidationError("Falha ao validar dataset", error=str(exc))
