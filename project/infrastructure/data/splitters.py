from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def loso_split(samples: List[Dict[str, Any]], subject_key: str = "subject_id") -> List[Tuple[List[int], List[int]]]:
    """Leave-One-Subject-Out split: para cada sujeito, separa treino/teste."""
    subjects = list({s[subject_key] for s in samples if subject_key in s})
    splits = []
    for subj in subjects:
        test_idx = [i for i, s in enumerate(samples) if s.get(subject_key) == subj]
        train_idx = [i for i in range(len(samples)) if i not in test_idx]
        splits.append((train_idx, test_idx))
    return splits


def stratified_kfold_split(X: Any, y: Any, n_splits: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Estratificado por classe."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def kfold_split(X: Any, n_splits: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split padrão K-Fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(X))
