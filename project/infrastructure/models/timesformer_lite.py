from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Codificação posicional senoidal (estática)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TimeSformerLite(nn.Module):
    """TimeSformer-Lite simplificado para sequências curtas de embeddings.

    Entrada:
      x: Tensor de forma (batch, seq_len, embed_dim)

    Saída:
      Tensor (batch, embed_dim) após pooling temporal.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(embed_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.pos(x)
        out = self.encoder(x)  # (B, T, D)
        out = self.norm(out)
        out = out.transpose(1, 2)  # (B, D, T)
        pooled = self.pool(out).squeeze(-1)  # (B, D)
        return pooled
