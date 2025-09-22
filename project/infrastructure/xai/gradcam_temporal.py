from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class TemporalGradCAM:
    """Grad-CAM temporal (genérico) para modelos seq->logit.

    Observação:
      - Este é um esqueleto genérico para redes que aceitam (B, T, D) e geram logits.
      - Requer convenção de registrar gradientes no penúltimo bloco (features temporais).
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model.eval()
        self.target_layer = target_layer
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        if self.target_layer is not None:
            self._hook()

    def _hook(self) -> None:
        assert self.target_layer is not None

        def fwd_hook(_: nn.Module, __, output: torch.Tensor) -> None:
            self._acts = output.detach()

        def bwd_hook(_: nn.Module, grad_in, grad_out) -> None:  # pragma: no cover
            self._grads = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)  # type: ignore[attr-defined]

    @torch.no_grad()
    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> List[float]:
        """Gera importância temporal normalizada (T valores entre 0-1)."""
        x.requires_grad_(True)
        logits = self.model(x)  # (B, C) ou (B,)
        if logits.ndim == 1:
            score = logits
        else:
            idx = class_idx if class_idx is not None else torch.argmax(logits, dim=1)
            score = logits[torch.arange(logits.size(0)), idx]

        score = score.sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self._acts is None or self._grads is None:
            # Fallback simples: variação temporal pela norma das diferenças
            # x: (B,T,D) -> (T,)
            diff = torch.norm(x[:, 1:, :] - x[:, :-1, :], dim=2).mean(dim=0)
            imp = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            return imp.cpu().tolist()

        # Grad-CAM temporal: média espacial (D) dos gradientes * ativações
        # acts/grads: assumidos (B, T, D)
        weights = self._grads.mean(dim=2)  # (B, T)
        cam = (weights * self._acts.mean(dim=2)).mean(dim=0)  # (T,)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().tolist()
