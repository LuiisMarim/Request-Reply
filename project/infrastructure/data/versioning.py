from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict


def compute_file_hash(path: str, block_size: int = 65536) -> str:
    """Computa SHA256 de um arquivo em blocos."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha.update(block)
    return sha.hexdigest()[:16]


def save_version_metadata(dataset_dir: str, out_path: str, extra_meta: Dict[str, Any] | None = None) -> str:
    """Salva metadados de versionamento do dataset (hashes + timestamp)."""
    records: Dict[str, Any] = {}
    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            fpath = os.path.join(root, fn)
            rel = os.path.relpath(fpath, dataset_dir)
            records[rel] = compute_file_hash(fpath)

    meta = {
        "dataset_dir": dataset_dir,
        "timestamp": datetime.utcnow().isoformat(),
        "files": records,
        "extra": extra_meta or {},
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_path
