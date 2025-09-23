from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Codificação posicional senoidal (estática)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model e max_len devem ser valores positivos.")
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

        logger.info("SinusoidalPositionalEncoding inicializado (d_model=%d, max_len=%d)", d_model, max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica codificação posicional senoidal a um tensor de entrada.

        Args:
            x (torch.Tensor): Tensor de entrada de forma (B, T, D).

        Returns:
            torch.Tensor: Tensor com codificação posicional adicionada.
        """
        if x.ndim != 3:
            raise ValueError(f"Entrada deve ter 3 dimensões (B, T, D), recebido {x.shape}")
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TimeSformerLite(nn.Module):
    """
    TimeSformer-Lite simplificado para sequências curtas de embeddings.

    Entrada:
        x: Tensor de forma (B, T, D)
    Saída:
        Tensor de forma (B, D) após pooling temporal.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0 or num_layers <= 0:
            raise ValueError("Parâmetros embed_dim, num_heads e num_layers devem ser positivos.")

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

        logger.info(
            "TimeSformerLite inicializado (embed_dim=%d, num_heads=%d, num_layers=%d, dropout=%.2f, max_len=%d)",
            embed_dim,
            num_heads,
            num_layers,
            dropout,
            max_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa forward pass do TimeSformerLite.

        Args:
            x (torch.Tensor): Tensor de entrada no formato (B, T, D).

        Returns:
            torch.Tensor: Tensor de saída no formato (B, D).
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("A entrada deve ser um torch.Tensor.")
        if x.ndim != 3:
            raise ValueError(f"Entrada deve ter 3 dimensões (B, T, D). Recebido: {x.shape}")

        logger.debug("Entrada do TimeSformerLite com shape: %s", x.shape)

        x = self.pos(x)
        out = self.encoder(x)  # (B, T, D)
        out = self.norm(out)
        out = out.transpose(1, 2)  # (B, D, T)
        pooled = self.pool(out).squeeze(-1)  # (B, D)

        logger.debug("Saída do TimeSformerLite com shape: %s", pooled.shape)
        return pooled
