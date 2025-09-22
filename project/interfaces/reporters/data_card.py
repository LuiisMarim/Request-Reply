from __future__ import annotations

import json
import os
from typing import Any, Dict


class DataCardReporter:
    """Gera Data Cards documentando datasets, riscos e usos."""

    def __init__(self, out_dir: str = "artifacts/datacards") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def write(self, metadata: Dict[str, Any], name: str = "data_card") -> str:
        path = os.path.join(self.out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return path
