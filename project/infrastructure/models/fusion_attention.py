from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from domain.services.feature_fuser import IFeatureFusion


class _AttentionGate(nn.Module):
    """Bloco de atenção escalar por fonte de feature."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Retorna logit de atenção (escalares por amostra)
        return self.net(x)  # (B, 1)


class FeatureFusionAttention(nn.Module, IFeatureFusion):
    """Fusão de features com atenção entre fontes (estático x temporal x linguístico).

    Espera um dicionário com tensores PyTorch por chave:
      - "aus": (B, D1)
      - "fuzzy": (B, D2)
      - "embeddings": (B, D3)
      - "temporal": (B, D4)

    Concatena entradas reponderadas por pesos de atenção.
    """

    def __init__(self, dims: Dict[str, int], projection_dim: Optional[int] = 256) -> None:
        super().__init__()
        self.order: List[str] = []
        self.attn: nn.ModuleDict = nn.ModuleDict()
        self.proj: Optional[nn.Linear] = None

        for key, dim in dims.items():
            self.order.append(key)
            self.attn[key] = _AttentionGate(in_dim=dim)

        in_total = sum(dims.values())
        if projection_dim is not None:
            self.proj = nn.Linear(in_total, projection_dim)

    def fuse(self, features: Dict[str, Any]) -> Any:
        """Implementa IFeatureFusion.fuse usando o forward abaixo."""
        return self.forward(features)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Calcula pesos de atenção por fonte e reescala cada vetor.
        parts: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []
        for key in self.order:
            x = features[key]  # (B, Dk)
            logit = self.attn[key](x)  # (B, 1)
            weights.append(logit)

        # Normaliza pesos com softmax ao longo das fontes
        W = torch.cat(weights, dim=1)  # (B, K)
        W = torch.softmax(W, dim=1)

        # Aplica pesos e concatena
        for i, key in enumerate(self.order):
            x = features[key]
            w = W[:, i].unsqueeze(1)  # (B,1)
            parts.append(x * w)

        fused = torch.cat(parts, dim=1)  # (B, sum Dk)
        if self.proj is not None:
            fused = self.proj(fused)
        return fused
