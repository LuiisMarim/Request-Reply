from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2

from infrastructure.utils.errors import DataValidationError


def validate_image_file(path: str) -> Tuple[bool, Dict[str, str]]:
    """Valida uma imagem individual (corrupção, dimensões, canais)."""
    if not os.path.exists(path):
        return False, {"error": "Arquivo não encontrado"}

    img = cv2.imread(path)
    if img is None:
        return False, {"error": "Imagem corrompida"}

    h, w, c = img.shape
    if c not in (1, 3):
        return False, {"error": f"Canais inválidos: {c}"}

    return True, {"height": str(h), "width": str(w), "channels": str(c)}


def validate_dataset(input_dir: str) -> List[Dict[str, str]]:
    """Valida dataset de imagens sob input_dir."""
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
    return results


class DataValidator:
    """Wrapper OO para validação de datasets."""

    def validate(self, input_dir: str, output_dir: str, report_path: str) -> bool:
        try:
            os.makedirs(output_dir, exist_ok=True)
            results = validate_dataset(input_dir)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("<html><body><h1>Data Quality Report</h1><ul>")
                for r in results:
                    color = "green" if r["valid"] == "True" else "red"
                    f.write(
                        f"<li style='color:{color}'>{r['file']} - valid={r['valid']} {r}</li>"
                    )
                f.write("</ul></body></html>")
            return True
        except Exception as exc:
            raise DataValidationError("Falha ao validar dataset", error=str(exc))
