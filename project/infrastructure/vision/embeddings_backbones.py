from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from domain.services.embedding_extractor import IEmbeddingExtractor


class VGG19EmbeddingExtractor(IEmbeddingExtractor):
    """Extrator de embeddings baseado em VGG19 (fc6/fc7)."""

    def __init__(self, layer: str = "fc7", device: str = "cpu") -> None:
        vgg = models.vgg19(pretrained=True)
        self.layer = layer
        self.device = device
        self.features = nn.Sequential(*list(vgg.classifier.children())[:-1])
        self.model = vgg.to(self.device).eval()

    def extract(self, image: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            feats = self.model.features(image.to(self.device))
            pooled = self.model.avgpool(feats)
            flat = torch.flatten(pooled, 1)
            emb = self.features(flat)
        return emb
