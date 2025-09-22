from __future__ import annotations

import os
from typing import Any, Dict

from .json_reporter import JSONReporter
from .html_reporter import HTMLReporter


class FairnessReporter:
    """Gera relatórios de fairness em JSON e HTML a partir das métricas."""

    def __init__(self, out_dir: str = "artifacts/fairness") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.json = JSONReporter(out_dir=self.out_dir)
        self.html = HTMLReporter(out_dir=self.out_dir)

    def report(self, summary: Dict[str, Any]) -> Dict[str, str]:
        json_path = self.json.write(summary, "fairness_report")
        html_path = self.html.write(summary, "fairness_report", title="Fairness Report")
        return {"json": json_path, "html": html_path}
