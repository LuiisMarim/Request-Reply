from __future__ import annotations

from typing import Dict

import torch


def export_attention_weights(
    attention_logits: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Normaliza logits de atenção por fonte e retorna pesos escalares médios.

    Args:
        attention_logits: dicionário {fonte: Tensor(B,1)} com logits por amostra.

    Returns:
        Pesos médios normalizados por fonte (softmax ao longo das fontes).
    """
    if not attention_logits:
        return {}

    keys = list(attention_logits.keys())
    batch_logits = [attention_logits[k] for k in keys]
    W = torch.cat(batch_logits, dim=1)  # (B, K)
    weights = torch.softmax(W, dim=1).mean(dim=0)  # (K,)
    return {k: float(weights[i].item()) for i, k in enumerate(keys)}
