from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ModelCardReporter:
    """Gera Model Cards (documentação ética, técnica e clínica)."""

    def __init__(self, out_dir: str = "artifacts/modelcards") -> None:
        if not isinstance(out_dir, str) or not out_dir.strip():
            logger.error("Diretório inválido para ModelCardReporter: %s", out_dir)
            raise ValueError("out_dir deve ser uma string não vazia.")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        logger.info("ModelCardReporter inicializado em: %s", self.out_dir)

    def write(self, metadata: Dict[str, Any], name: str = "model_card") -> str:
        """
        Salva Model Card em JSON.

        Args:
            metadata (Dict[str, Any]): Metadados técnicos, éticos e clínicos do modelo.
            name (str): Nome base do arquivo (sem extensão).

        Returns:
            str: Caminho do arquivo JSON gerado.

        Raises:
            ValueError: Se metadata não for dicionário ou estiver vazio.
            RuntimeError: Se falhar a escrita do arquivo.
        """
        if not isinstance(metadata, dict) or not metadata:
            logger.error("Metadata inválido fornecido: %s", metadata)
            raise ValueError("metadata deve ser um dicionário não vazio.")

        if not isinstance(name, str) or not name.strip():
            logger.error("Nome inválido para Model Card: %s", name)
            raise ValueError("name deve ser uma string não vazia.")

        path = os.path.join(self.out_dir, f"{name}.json")

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info("Model Card salvo em: %s", path)
            return path
        except Exception as e:
            logger.exception("Falha ao salvar Model Card em %s: %s", path, str(e))
            raise RuntimeError(f"Falha ao salvar Model Card: {str(e)}") from e
