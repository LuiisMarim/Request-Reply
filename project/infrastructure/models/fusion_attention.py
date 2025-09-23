from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import logging

from domain.services.feature_fuser import IFeatureFusion

logger = logging.getLogger(__name__)


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
        """
        Retorna logit de atenção (escalares por amostra).

        Args:
            x (torch.Tensor): Tensor de entrada no formato (B, D).

        Returns:
            torch.Tensor: Logits no formato (B, 1).
        """
        if x.ndim != 2:
            raise ValueError(f"O tensor de entrada deve ter 2 dimensões (B, D), recebido {x.shape}")
        return self.net(x)  # (B, 1)


class FeatureFusionAttention(nn.Module, IFeatureFusion):
    """
    Fusão de features com atenção entre múltiplas fontes.

    Espera um dicionário com tensores PyTorch por chave:
      - "aus": (B, D1)
      - "fuzzy": (B, D2)
      - "embeddings": (B, D3)
      - "temporal": (B, D4)

    As entradas são reponderadas por pesos de atenção e concatenadas.
    """

    def __init__(self, dims: Dict[str, int], projection_dim: Optional[int] = 256) -> None:
        """
        Inicializa o módulo de fusão com atenção.

        Args:
            dims (Dict[str, int]): Dicionário contendo as dimensões de cada fonte de feature.
            projection_dim (Optional[int]): Dimensão final de projeção após concatenação. Se None, não projeta.
        """
        super().__init__()
        self.order: List[str] = []
        self.attn: nn.ModuleDict = nn.ModuleDict()
        self.proj: Optional[nn.Linear] = None

        for key, dim in dims.items():
            if not isinstance(key, str) or not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Dimensão inválida para chave '{key}': {dim}")
            self.order.append(key)
            self.attn[key] = _AttentionGate(in_dim=dim)

        in_total = sum(dims.values())
        if projection_dim is not None:
            self.proj = nn.Linear(in_total, projection_dim)

        logger.info("FeatureFusionAttention inicializado com fontes: %s", self.order)

    def fuse(self, features: Dict[str, Any]) -> Any:
        """Implementa IFeatureFusion.fuse chamando o forward."""
        return self.forward(features)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Executa a fusão das features com atenção.

        Args:
            features (Dict[str, torch.Tensor]): Dicionário com tensores de entrada.

        Returns:
            torch.Tensor: Tensor fusionado após aplicação de pesos de atenção e concatenação.
        """
        if not isinstance(features, dict):
            raise ValueError("O parâmetro 'features' deve ser um dicionário.")

        parts: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []

        for key in self.order:
            if key not in features:
                raise ValueError(f"Feature esperada '{key}' não encontrada nas entradas.")
            x = features[key]
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"A feature '{key}' deve ser um torch.Tensor.")
            if x.ndim != 2:
                raise ValueError(f"A feature '{key}' deve ter 2 dimensões (B, D). Recebido: {x.shape}")

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

        logger.debug("Fusão concluída. Saída com shape: %s", fused.shape)
        return fused
