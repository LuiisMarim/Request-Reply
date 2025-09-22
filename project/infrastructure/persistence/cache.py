from __future__ import annotations

import os
from typing import Any, Optional

from joblib import Memory, dump, load

from infrastructure.utils.errors import IOFailure


class CacheManager:
    """Cache simples baseado em joblib.Memory + utilitários dump/load.

    - Útil para memorizar funções caras (ex.: extração de embeddings).
    - Também expõe dump/load para objetos arbitrários.
    """

    def __init__(self, cache_dir: str, verbose: int = 0) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        self.memory = Memory(location=cache_dir, verbose=verbose)

    def cache_fn(self, fn):
        """Decora função com cache (memoization)."""
        return self.memory.cache(fn)

    def save(self, obj: Any, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            dump(obj, path)
        except Exception as exc:
            raise IOFailure("Falha ao salvar objeto em cache", path=path, error=str(exc))

    def load(self, path: str) -> Any:
        try:
            return load(path)
        except Exception as exc:
            raise IOFailure("Falha ao carregar objeto do cache", path=path, error=str(exc))
