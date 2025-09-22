from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from domain.entities.fairness import FairnessReport
from domain.services.fairness_auditor import IFairnessAuditor
from .metrics import compute_fairness_metrics


class FairlearnAuditor(IFairnessAuditor):
    """Auditor de Fairness que gera relatório JSON e HTML simples."""

    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir

    def _ensure_dirs(self) -> str:
        out_dir = os.path.join(self.artifacts_dir, "reports")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def audit(
        self, X: Any, y_true: Any, y_pred: Any, sensitive_features: Any
    ) -> FairnessReport:
        out_dir = self._ensure_dirs()
        summary = compute_fairness_metrics(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)

        json_path = os.path.join(out_dir, "fairness_report.json")
        html_path = os.path.join(out_dir, "fairness_report.html")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._write_html(summary, html_path)
        return FairnessReport(metrics=summary.get("per_group", {}))

    def _write_html(self, summary: Dict[str, Any], path: str) -> None:
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
            f.write("</ul><p style='font-size:12px;color:#666'>Nota ética: Ferramenta de apoio à decisão; não substitui avaliação multiprofissional. Requer aprovação ética e conformidade LGPD/GDPR. Uso sob supervisão de profissional de saúde.</p>")
            f.write("</body></html>")
