from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
import logging

from domain.services.embedding_extractor import IEmbeddingExtractor

logger = logging.getLogger(__name__)


class VGG19EmbeddingExtractor(IEmbeddingExtractor):
    """
    Extrator de embeddings baseado em VGG19.

    Permite extrair embeddings de diferentes camadas (fc6/fc7).
    """

    def __init__(self, layer: str = "fc7", device: str = "cpu") -> None:
        """
        Inicializa o extrator VGG19.

        Args:
            layer (str): Camada alvo para extração ("fc6" ou "fc7").
            device (str): Device alvo ("cpu" ou "cuda").
        """
        if layer not in {"fc6", "fc7"}:
            logger.error("Camada inválida para VGG19EmbeddingExtractor: %s", layer)
            raise ValueError("A camada deve ser 'fc6' ou 'fc7'.")
        self.layer = layer
        self.device = device

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.classifier.children())[:-1])
        self.model = vgg.to(self.device).eval()

        logger.info("VGG19EmbeddingExtractor inicializado (layer=%s, device=%s)", self.layer, self.device)

    def extract(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extrai embeddings da imagem fornecida.

        Args:
            image (torch.Tensor): Tensor de entrada no formato (B, C, H, W).

        Returns:
            torch.Tensor: Tensor de embeddings no formato (B, D).

        Raises:
            ValueError: Se a entrada não for válida.
        """
        if not isinstance(image, torch.Tensor):
            logger.error("Entrada inválida para extração: %s", type(image))
            raise ValueError("A entrada deve ser um torch.Tensor.")
        if image.ndim != 4:
            logger.error("Tensor inválido, esperado 4D (B,C,H,W), recebido: %s", image.shape)
            raise ValueError("O tensor de entrada deve ter 4 dimensões (B, C, H, W).")

        with torch.no_grad():
            feats = self.model.features(image.to(self.device))
            pooled = self.model.avgpool(feats)
            flat = torch.flatten(pooled, 1)
            emb = self.features(flat)

        logger.debug("Embeddings extraídos com shape: %s", emb.shape)
        return emb
