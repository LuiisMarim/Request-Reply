from __future__ import annotations

import os
import random
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """
    Define seeds globais para reprodutibilidade.

    Configura seeds para os módulos:
      - random
      - numpy
      - torch (se disponível)

    Args:
        seed (int): Valor da semente.
        deterministic_torch (bool): Se True, força determinismo em operações do Torch.

    Raises:
        ValueError: Se seed não for um inteiro válido.
    """
    if not isinstance(seed, int) or seed < 0:
        logger.error("Seed inválido fornecido: %s", seed)
        raise ValueError("seed deve ser um inteiro não negativo.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seed global definida: %d", seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        logger.info("Seed configurada também para Torch (deterministic=%s).", deterministic_torch)
    except Exception as e:
        logger.warning("Torch não disponível ou falha ao configurar seed: %s", str(e))


def get_seed_from_env(default: int = 42) -> int:
    """
    Obtém seed do ambiente (variável GLOBAL_SEED) ou retorna valor default.

    Args:
        default (int): Valor padrão caso a variável não esteja definida ou seja inválida.

    Returns:
        int: Valor da seed.
    """
    try:
        seed = int(os.getenv("GLOBAL_SEED", str(default)))
        logger.info("Seed obtida do ambiente: %d", seed)
        return seed
    except ValueError:
        logger.warning("Valor inválido em GLOBAL_SEED, retornando default=%d.", default)
        return default
