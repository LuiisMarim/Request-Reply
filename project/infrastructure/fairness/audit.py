from __future__ import annotations

import json
import os
import logging
from typing import Any, Dict

from domain.entities.fairness import FairnessReport
from domain.services.fairness_auditor import IFairnessAuditor
from .metrics import compute_fairness_metrics

logger = logging.getLogger(__name__)


class FairlearnAuditor(IFairnessAuditor):
    """
    Auditor de Fairness que gera relatórios em JSON e HTML.

    Este auditor calcula métricas de fairness por subgrupo e salva
    relatórios para consulta posterior.
    """

    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        """
        Inicializa o auditor.

        Args:
            artifacts_dir (str): Diretório base onde relatórios serão salvos.
        """
        self.artifacts_dir = artifacts_dir

    def _ensure_dirs(self) -> str:
        """Garante que o diretório de relatórios exista e retorna seu caminho."""
        out_dir = os.path.join(self.artifacts_dir, "reports")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def audit(
        self, X: Any, y_true: Any, y_pred: Any, sensitive_features: Any
    ) -> FairnessReport:
        """
        Executa auditoria de fairness e gera relatórios.

        Args:
            X (Any): Features de entrada (não utilizado diretamente neste auditor).
            y_true (Any): Rótulos verdadeiros.
            y_pred (Any): Predições do modelo.
            sensitive_features (Any): Atributos sensíveis para análise (ex.: gênero, etnia).

        Returns:
            FairnessReport: Objeto contendo métricas de fairness por subgrupo.

        Raises:
            ValueError: Se entradas obrigatórias forem inválidas.
        """
        if y_true is None or y_pred is None or sensitive_features is None:
            logger.error("Entradas inválidas para auditoria: y_true=%s, y_pred=%s, sensitive_features=%s",
                         y_true, y_pred, sensitive_features)
            raise ValueError("y_true, y_pred e sensitive_features não podem ser None.")

        out_dir = self._ensure_dirs()
        logger.info("Iniciando auditoria de fairness. Relatórios em: %s", out_dir)

        try:
            summary = compute_fairness_metrics(
                y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
            )
        except Exception as e:
            logger.exception("Erro ao calcular métricas de fairness: %s", str(e))
            raise

        json_path = os.path.join(out_dir, "fairness_report.json")
        html_path = os.path.join(out_dir, "fairness_report.html")

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info("Relatório JSON salvo em: %s", json_path)

            self._write_html(summary, html_path)
            logger.info("Relatório HTML salvo em: %s", html_path)

        except Exception as e:
            logger.exception("Erro ao salvar relatórios de fairness: %s", str(e))
            raise

        return FairnessReport(metrics=summary.get("per_group", {}))

    def _write_html(self, summary: Dict[str, Any], path: str) -> None:
        """
        Gera arquivo HTML a partir do resumo de métricas.

        Args:
            summary (Dict[str, Any]): Métricas calculadas.
            path (str): Caminho do relatório HTML de saída.
        """
        try:
            overall = summary.get("overall", {})
            per_group = summary.get("per_group", {})

            with open(path, "w", encoding="utf-8") as f:
                f.write("<html><head><meta charset='utf-8'><title>Fairness Report</title></head><body>")
                f.write("<h1>Fairness Report</h1>")
                f.write("<h2>Overall</h2><ul>")
                for k, v in overall.items():
                    f.write(f"<li>{k}: {v}</li>")
                f.write("</ul><h2>Per Group</h2><ul>")
                for g, metrics in per_group.items():
                    f.write(f"<li><b>{g}</b><ul>")
                    for mk, mv in metrics.items():
                        f.write(f"<li>{mk}: {mv}</li>")
                    f.write("</ul></li>")
                f.write("</ul>")
                f.write("<p style='font-size:12px;color:#666'>")
                f.write("Nota ética: Ferramenta de apoio à decisão; não substitui avaliação multiprofissional. ")
                f.write("Requer aprovação ética e conformidade LGPD/GDPR. Uso sob supervisão de profissional de saúde.")
                f.write("</p>")
                f.write("</body></html>")
        except Exception as e:
            logger.exception("Erro ao escrever relatório HTML: %s", str(e))
            raise
