from __future__ import annotations

import json
import os
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class DataCardReporter:
    """Gera Data Cards documentando datasets, riscos e usos."""

    def __init__(self, out_dir: str = "artifacts/datacards") -> None:
        if not isinstance(out_dir, str) or not out_dir.strip():
            logger.error("Diretório inválido para DataCardReporter: %s", out_dir)
            raise ValueError("out_dir deve ser uma string não vazia.")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        logger.info("DataCardReporter inicializado em: %s", self.out_dir)

    def write(self, metadata: Dict[str, Any], name: str = "data_card") -> str:
        """
        Salva Data Card em JSON.

        Args:
            metadata (Dict[str, Any]): Metadados do dataset, riscos, usos etc.
            name (str): Nome base do arquivo (sem extensão).

        Returns:
            str: Caminho do arquivo JSON gerado.

        Raises:
            ValueError: Se metadata não for um dicionário ou estiver vazio.
            RuntimeError: Se ocorrer falha ao salvar o arquivo.
        """
        if not isinstance(metadata, dict) or not metadata:
            logger.error("Metadata inválido fornecido: %s", metadata)
            raise ValueError("metadata deve ser um dicionário não vazio.")

        if not isinstance(name, str) or not name.strip():
            logger.error("Nome inválido para Data Card: %s", name)
            raise ValueError("name deve ser uma string não vazia.")

        path = os.path.join(self.out_dir, f"{name}.json")

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info("Data Card salvo com sucesso em: %s", path)
            return path
        except Exception as e:
            logger.exception("Falha ao salvar Data Card em %s: %s", path, str(e))
            raise RuntimeError(f"Falha ao salvar Data Card: {str(e)}") from e
