from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class JSONReporter:
    """Gera relatórios em JSON (métricas, fairness, XAI)."""

    def __init__(self, out_dir: str = "artifacts/reports") -> None:
        if not isinstance(out_dir, str) or not out_dir.strip():
            logger.error("Diretório inválido para JSONReporter: %s", out_dir)
            raise ValueError("out_dir deve ser uma string não vazia.")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        logger.info("JSONReporter inicializado em: %s", self.out_dir)

    def write(self, data: Dict[str, Any], name: str) -> str:
        """
        Salva relatório em formato JSON.

        Args:
            data (Dict[str, Any]): Dados a incluir no relatório.
            name (str): Nome base do arquivo (sem extensão).

        Returns:
            str: Caminho do arquivo JSON gerado.

        Raises:
            ValueError: Se data ou name forem inválidos.
            RuntimeError: Se falhar a escrita do arquivo.
        """
        if not isinstance(data, dict) or not data:
            logger.error("Dados inválidos fornecidos ao JSONReporter: %s", data)
            raise ValueError("data deve ser um dicionário não vazio.")

        if not isinstance(name, str) or not name.strip():
            logger.error("Nome inválido para relatório JSON: %s", name)
            raise ValueError("name deve ser uma string não vazia.")

        path = os.path.join(self.out_dir, f"{name}.json")

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Relatório JSON salvo em: %s", path)
            return path
        except Exception as e:
            logger.exception("Falha ao salvar relatório JSON em %s: %s", path, str(e))
            raise RuntimeError(f"Falha ao salvar relatório JSON: {str(e)}") from e
