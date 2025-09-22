from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Define seeds globais para reprodutibilidade.

    - random, numpy e (opcional) torch.
    - Para torch: ativa determinismo quando solicitado.

    Obs.: Torch é importado de forma tardia para evitar dependência rígida aqui.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        # Torch pode não estar disponível em alguns ambientes (tests/lightweight)
        pass


def get_seed_from_env(default: int = 42) -> int:
    """Obtém seed do ambiente (GLOBAL_SEED) ou retorna default."""
    try:
        return int(os.getenv("GLOBAL_SEED", str(default)))
    except ValueError:
        return default
