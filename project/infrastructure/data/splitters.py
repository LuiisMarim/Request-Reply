from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import logging

logger = logging.getLogger(__name__)


def loso_split(samples: List[Dict[str, Any]], subject_key: str = "subject_id") -> List[Tuple[List[int], List[int]]]:
    """
    Realiza Leave-One-Subject-Out split.

    Para cada sujeito, separa treino e teste deixando todas as amostras desse
    sujeito como teste.

    Args:
        samples (List[Dict[str, Any]]): Lista de amostras, cada uma contendo metadados.
        subject_key (str): Chave usada para identificar o sujeito.

    Returns:
        List[Tuple[List[int], List[int]]]: Lista de tuplas (train_idx, test_idx).

    Raises:
        ValueError: Se não houver amostras ou se a chave do sujeito não estiver presente.
    """
    if not samples:
        logger.error("Lista de samples vazia fornecida ao loso_split.")
        raise ValueError("A lista de samples não pode estar vazia.")

    if not any(subject_key in s for s in samples):
        logger.error("Chave '%s' não encontrada em nenhuma amostra.", subject_key)
        raise ValueError(f"A chave '{subject_key}' não foi encontrada nas amostras.")

    subjects = list({s[subject_key] for s in samples if subject_key in s})
    splits = []
    for subj in subjects:
        test_idx = [i for i, s in enumerate(samples) if s.get(subject_key) == subj]
        train_idx = [i for i in range(len(samples)) if i not in test_idx]
        splits.append((train_idx, test_idx))
        logger.debug("LOSO split criado para sujeito '%s': %d treino, %d teste.", subj, len(train_idx), len(test_idx))

    logger.info("Total de %d splits LOSO gerados.", len(splits))
    return splits


def stratified_kfold_split(X: Any, y: Any, n_splits: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Realiza split estratificado K-Fold.

    Args:
        X (Any): Features de entrada.
        y (Any): Labels correspondentes.
        n_splits (int): Número de folds.
        seed (int): Semente aleatória para reprodutibilidade.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: Lista de índices de treino e validação.

    Raises:
        ValueError: Se n_splits for menor que 2.
    """
    if n_splits < 2:
        logger.error("n_splits inválido: %d", n_splits)
        raise ValueError("O número de splits deve ser >= 2.")

    logger.info("Executando StratifiedKFold com %d splits e seed=%d.", n_splits, seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def kfold_split(X: Any, n_splits: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Realiza split padrão K-Fold.

    Args:
        X (Any): Features de entrada.
        n_splits (int): Número de folds.
        seed (int): Semente aleatória para reprodutibilidade.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: Lista de índices de treino e validação.

    Raises:
        ValueError: Se n_splits for menor que 2.
    """
    if n_splits < 2:
        logger.error("n_splits inválido: %d", n_splits)
        raise ValueError("O número de splits deve ser >= 2.")

    logger.info("Executando KFold com %d splits e seed=%d.", n_splits, seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(X))
