from __future__ import annotations

import os
import logging
from typing import Any, Dict

from .json_reporter import JSONReporter
from .html_reporter import HTMLReporter

logger = logging.getLogger(__name__)


class FairnessReporter:
    """Gera relatórios de fairness em JSON e HTML a partir das métricas."""

    def __init__(self, out_dir: str = "artifacts/fairness") -> None:
        if not isinstance(out_dir, str) or not out_dir.strip():
            logger.error("Diretório inválido para FairnessReporter: %s", out_dir)
            raise ValueError("out_dir deve ser uma string não vazia.")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.json = JSONReporter(out_dir=self.out_dir)
        self.html = HTMLReporter(out_dir=self.out_dir)
        logger.info("FairnessReporter inicializado em: %s", self.out_dir)

    def report(self, summary: Dict[str, Any]) -> Dict[str, str]:
        """
        Gera relatórios de fairness em JSON e HTML.

        Args:
            summary (Dict[str, Any]): Resumo de métricas de fairness.

        Returns:
            Dict[str, str]: Caminhos dos arquivos gerados {"json": ..., "html": ...}.

        Raises:
            ValueError: Se summary for inválido.
            RuntimeError: Se falhar a escrita dos relatórios.
        """
        if not isinstance(summary, dict) or not summary:
            logger.error("Resumo inválido fornecido ao FairnessReporter: %s", summary)
            raise ValueError("summary deve ser um dicionário não vazio.")

        try:
            logger.info("Gerando relatórios de fairness...")
            json_path = self.json.write(summary, "fairness_report")
            html_path = self.html.write(summary, "fairness_report", title="Fairness Report")
            logger.info("Relatórios de fairness gerados: JSON=%s, HTML=%s", json_path, html_path)
            return {"json": json_path, "html": html_path}
        except Exception as e:
            logger.exception("Falha ao gerar relatórios de fairness: %s", str(e))
            raise RuntimeError(f"Falha ao gerar relatórios de fairness: {str(e)}") from e
