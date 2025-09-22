from __future__ import annotations

import json
import os
from typing import Any, Dict


class ModelCardReporter:
    """Gera Model Cards (documentação ética, técnica e clínica)."""

    def __init__(self, out_dir: str = "artifacts/modelcards") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def write(self, metadata: Dict[str, Any], name: str = "model_card") -> str:
        path = os.path.join(self.out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return path
