from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from domain.entities.metrics import Metrics
from domain.services.classifier import IClassifier


class _MLP(nn.Module):
    """MLP simples para classificação binária (autistic vs non_autistic)."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),  # saída logit binária
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


class TorchMLPClassifier(IClassifier):
    """Wrapper de MLP em PyTorch compatível com IClassifier."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 64,
        device: Optional[str] = None,
        use_amp: bool = True,
    ) -> None:
        self.model = _MLP(in_dim=in_dim, hidden=hidden, dropout=dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and (self.device == "cuda")

        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _to_tensor(self, X: Any, y: Optional[Any] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        y_t = None if y is None else torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device)
        return X_t, y_t

    def _iterate_minibatches(self, X: torch.Tensor, y: torch.Tensor, batch_size: int):
        n = X.size(0)
        for i in range(0, n, batch_size):
            yield X[i : i + batch_size], y[i : i + batch_size]

    def train(self, X: Any, y: Any) -> None:
        self.model.train()
        X_t, y_t = self._to_tensor(X, y)
        for _ in range(self.epochs):
            for xb, yb in self._iterate_minibatches(X_t, y_t, self.batch_size):
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

    @torch.no_grad()
    def predict(self, X: Any) -> Any:
        self.model.eval()
        X_t, _ = self._to_tensor(X)
        logits = self.model(X_t)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().cpu().numpy()
        return preds

    @torch.no_grad()
    def evaluate(self, X: Any, y: Any) -> Metrics:
        self.model.eval()
        X_t, y_t = self._to_tensor(X, y)
        logits = self.model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y_t.cpu().numpy().astype(int)

        y_pred = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = float("nan")
        f1 = f1_score(y_true, y_pred)

        per_group: Dict[str, Dict[str, float]] = {}
        return Metrics(accuracy=acc, auc=auc, f1_score=f1, per_group=per_group)
