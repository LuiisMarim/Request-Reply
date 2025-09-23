from __future__ import annotations

from typing import Dict
import logging

import torch

logger = logging.getLogger(__name__)


def export_attention_weights(
    attention_logits: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Normaliza logits de atenção por fonte e retorna pesos escalares médios.

    Estratégia:
        - Concatena logits de diferentes fontes no batch.
        - Aplica softmax ao longo das fontes (dim=1).
        - Faz média ao longo do batch.
        - Retorna dicionário {fonte: peso médio}.

    Args:
        attention_logits (Dict[str, torch.Tensor]): Dicionário {fonte: Tensor(B,1)}.

    Returns:
        Dict[str, float]: Pesos médios normalizados por fonte.

    Raises:
        ValueError: Se o dicionário for inválido ou os tensores não forem compatíveis.
    """
    if not attention_logits or not isinstance(attention_logits, dict):
        logger.error("attention_logits inválido: %s", type(attention_logits))
        raise ValueError("attention_logits deve ser um dicionário não vazio.")

    keys = list(attention_logits.keys())
    batch_logits = []

    for k in keys:
        tensor = attention_logits[k]
        if not isinstance(tensor, torch.Tensor):
            logger.error("Logit inválido para chave '%s': %s", k, type(tensor))
            raise ValueError(f"Valor de attention_logits['{k}'] deve ser um torch.Tensor.")
        if tensor.ndim != 2 or tensor.shape[1] != 1:
            logger.error("Tensor inválido para chave '%s': shape %s", k, tensor.shape)
            raise ValueError(f"Tensor para chave '{k}' deve ter shape (B,1).")
        batch_logits.append(tensor)

    try:
        W = torch.cat(batch_logits, dim=1)  # (B, K)
        weights = torch.softmax(W, dim=1).mean(dim=0)  # (K,)
        result = {k: float(weights[i].item()) for i, k in enumerate(keys)}
        logger.info("Pesos de atenção exportados com sucesso: %s", result)
        return result
    except Exception as e:
        logger.exception("Falha ao exportar pesos de atenção: %s", str(e))
        raise
