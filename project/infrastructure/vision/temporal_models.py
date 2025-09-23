from __future__ import annotations

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class TimeSformerLite(nn.Module):
    """
    Implementação simplificada de um TimeSformer-Lite.

    Entrada:
        x: Tensor no formato (B, T, D) onde:
            B = batch size
            T = número de frames (seq_len)
            D = dimensão dos embeddings

    Saída:
        Tensor no formato (B, D) após pooling temporal.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 2) -> None:
        if embed_dim <= 0 or num_heads <= 0 or num_layers <= 0:
            logger.error(
                "Parâmetros inválidos para TimeSformerLite: embed_dim=%d, num_heads=%d, num_layers=%d",
                embed_dim, num_heads, num_layers
            )
            raise ValueError("Parâmetros embed_dim, num_heads e num_layers devem ser positivos.")

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

        logger.info(
            "TimeSformerLite inicializado (embed_dim=%d, num_heads=%d, num_layers=%d)",
            embed_dim, num_heads, num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa o forward do TimeSformerLite.

        Args:
            x (torch.Tensor): Tensor de entrada no formato (B, T, D).

        Returns:
            torch.Tensor: Tensor de saída no formato (B, D).

        Raises:
            ValueError: Se x não for tensor 3D no formato esperado.
        """
        if not isinstance(x, torch.Tensor):
            logger.error("Entrada inválida para TimeSformerLite: %s", type(x))
            raise ValueError("A entrada deve ser um torch.Tensor.")

        if x.ndim != 3:
            logger.error("Tensor inválido, esperado 3D (B, T, D), recebido: %s", x.shape)
            raise ValueError("O tensor de entrada deve ter 3 dimensões (B, T, D).")

        logger.debug("Entrada do TimeSformerLite com shape: %s", x.shape)

        out = self.encoder(x)  # (B, T, D)
        out = out.transpose(1, 2)  # (B, D, T)
        pooled = self.pool(out).squeeze(-1)  # (B, D)

        logger.debug("Saída do TimeSformerLite com shape: %s", pooled.shape)
        return pooled
