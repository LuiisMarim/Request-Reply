from __future__ import annotations

from typing import Dict, List, Optional
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemporalGradCAM:
    """
    Grad-CAM temporal para modelos seq->logit.

    Estratégia:
        - Registra ativações e gradientes na camada alvo.
        - Calcula importância temporal normalizada (0–1).
        - Se hooks não forem acionados, usa fallback baseado em variação temporal.

    Observações:
        - Requer que o modelo aceite entrada (B, T, D).
        - O target_layer deve produzir tensores (B, T, D).
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model.eval()
        self.target_layer = target_layer
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        if self.target_layer is not None:
            self._hook()
            logger.info("Hook registrado no target_layer: %s", type(self.target_layer).__name__)

    def _hook(self) -> None:
        assert self.target_layer is not None

        def fwd_hook(_: nn.Module, __, output: torch.Tensor) -> None:
            self._acts = output.detach()
            logger.debug("Ativações capturadas com shape: %s", output.shape)

        def bwd_hook(_: nn.Module, grad_in, grad_out) -> None:  # pragma: no cover
            self._grads = grad_out[0].detach()
            logger.debug("Gradientes capturados com shape: %s", grad_out[0].shape)

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)  # type: ignore[attr-defined]

    @torch.no_grad()
    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> List[float]:
        """
        Gera importância temporal normalizada (valores entre 0–1).

        Args:
            x (torch.Tensor): Tensor de entrada no formato (B, T, D).
            class_idx (Optional[int]): Índice da classe alvo. Se None, usa a predição de maior logit.

        Returns:
            List[float]: Lista com importâncias temporais normalizadas.

        Raises:
            ValueError: Se x não for um tensor válido.
            RuntimeError: Se ocorrer falha durante forward/backward.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            logger.error("Entrada inválida para TemporalGradCAM: %s", type(x))
            raise ValueError("x deve ser um torch.Tensor 3D no formato (B, T, D).")

        try:
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
        except Exception as e:
            logger.exception("Falha no forward/backward do modelo: %s", str(e))
            raise RuntimeError(f"Falha durante forward/backward: {str(e)}") from e

        if self._acts is None or self._grads is None:
            logger.warning("Hooks não capturaram gradientes/ativações. Usando fallback.")
            diff = torch.norm(x[:, 1:, :] - x[:, :-1, :], dim=2).mean(dim=0)
            imp = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            return imp.cpu().tolist()

        try:
            weights = self._grads.mean(dim=2)  # (B, T)
            cam = (weights * self._acts.mean(dim=2)).mean(dim=0)  # (T,)
            cam = torch.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            logger.info("Grad-CAM temporal gerado com sucesso.")
            return cam.cpu().tolist()
        except Exception as e:
            logger.exception("Falha ao calcular Grad-CAM temporal: %s", str(e))
            raise RuntimeError(f"Falha ao calcular Grad-CAM temporal: {str(e)}") from e
