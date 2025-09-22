from __future__ import annotations

import os
from typing import Any, Dict


class HTMLReporter:
    """Gera relatórios HTML simples e autoexplicativos."""

    def __init__(self, out_dir: str = "artifacts/reports") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def write(self, data: Dict[str, Any], name: str, title: str = "Report") -> str:
        path = os.path.join(self.out_dir, f"{name}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'>")
            f.write(f"<title>{title}</title></head><body>")
            f.write(f"<h1>{title}</h1><ul>")
            for k, v in data.items():
                f.write(f"<li><b>{k}</b>: {v}</li>")
            f.write("</ul>")
            f.write(
                "<p style='font-size:12px;color:#666'>Nota ética: Ferramenta de apoio à decisão; "
                "não substitui avaliação multiprofissional. Requer aprovação ética e conformidade "
                "LGPD/GDPR. Uso sob supervisão de profissional de saúde.</p>"
            )
            f.write("</body></html>")
        return path
