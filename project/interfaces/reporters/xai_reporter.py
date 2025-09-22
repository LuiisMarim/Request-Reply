from __future__ import annotations

import os
from typing import Any, Dict

from .json_reporter import JSONReporter
from .html_reporter import HTMLReporter


class XAIReporter:
    """Gera relatórios de interpretabilidade (SHAP, Grad-CAM, Attention Maps)."""

    def __init__(self, out_dir: str = "artifacts/xai") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.json = JSONReporter(out_dir=self.out_dir)
        self.html = HTMLReporter(out_dir=self.out_dir)

    def report(self, shap_values: Any, attention: Dict[str, float], temporal: Any) -> Dict[str, str]:
        """Exporta resultados em JSON e HTML."""
        data = {"shap": str(type(shap_values)), "attention": attention, "temporal": temporal}
        json_path = self.json.write(data, "xai_report")
        html_path = self.html.write(data, "xai_report", title="XAI Report")
        return {"json": json_path, "html": html_path}
