from __future__ import annotations

import json
import os
from typing import Any, Dict


class JSONReporter:
    """Gera relatórios em JSON (métricas, fairness, XAI)."""

    def __init__(self, out_dir: str = "artifacts/reports") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def write(self, data: Dict[str, Any], name: str) -> str:
        path = os.path.join(self.out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path
