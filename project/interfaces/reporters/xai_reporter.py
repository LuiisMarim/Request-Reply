from __future__ import annotations

import os
import logging
from typing import Any, Dict

from .json_reporter import JSONReporter
from .html_reporter import HTMLReporter

logger = logging.getLogger(__name__)


class XAIReporter:
    """Gera relatórios de interpretabilidade (SHAP, Grad-CAM, Attention Maps)."""

    def __init__(self, out_dir: str = "artifacts/xai") -> None:
        if not isinstance(out_dir, str) or not out_dir.strip():
            logger.error("Diretório inválido para XAIReporter: %s", out_dir)
            raise ValueError("out_dir deve ser uma string não vazia.")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.json = JSONReporter(out_dir=self.out_dir)
        self.html = HTMLReporter(out_dir=self.out_dir)
        logger.info("XAIReporter inicializado em: %s", self.out_dir)

    def report(self, shap_values: Any, attention: Dict[str, float], temporal: Any) -> Dict[str, str]:
        """
        Exporta resultados de interpretabilidade em JSON e HTML.

        Args:
            shap_values (Any): Valores SHAP (ou representação equivalente).
            attention (Dict[str, float]): Pesos de atenção.
            temporal (Any): Importâncias temporais (ex.: Grad-CAM temporal).

        Returns:
            Dict[str, str]: Caminhos dos relatórios gerados {"json": ..., "html": ...}.

        Raises:
            ValueError: Se entradas forem inválidas.
            RuntimeError: Se falhar a geração dos relatórios.
        """
        if shap_values is None:
            logger.error("shap_values inválido: None")
            raise ValueError("shap_values não pode ser None.")
        if not isinstance(attention, dict):
            logger.error("attention inválido: %s", type(attention))
            raise ValueError("attention deve ser um dicionário.")
        if temporal is None:
            logger.error("temporal inválido: None")
            raise ValueError("temporal não pode ser None.")

        data = {
            "shap": str(type(shap_values)),
            "attention": attention,
            "temporal": temporal,
        }

        try:
            logger.info("Gerando relatórios XAI...")
            json_path = self.json.write(data, "xai_report")
            html_path = self.html.write(data, "xai_report", title="XAI Report")
            logger.info("Relatórios XAI gerados: JSON=%s, HTML=%s", json_path, html_path)
            return {"json": json_path, "html": html_path}
        except Exception as e:
            logger.exception("Falha ao gerar relatórios XAI: %s", str(e))
            raise RuntimeError(f"Falha ao gerar relatórios XAI: {str(e)}") from e
