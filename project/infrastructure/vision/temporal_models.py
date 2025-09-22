from __future__ import annotations

import torch
import torch.nn as nn


class TimeSformerLite(nn.Module):
    """Implementação simplificada de um TimeSformer-Lite."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, embed_dim)."""
        out = self.encoder(x)
        out = out.transpose(1, 2)  # (B, D, T)
        pooled = self.pool(out).squeeze(-1)
        return pooled
