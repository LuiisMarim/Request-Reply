from __future__ import annotations

import glob
import os
from typing import List, Optional

import pandas as pd

from infrastructure.utils.errors import IOFailure


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Salva DataFrame em Parquet (pyarrow) com diretório criado se necessário."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_parquet(path, index=False)
    except Exception as exc:
        raise IOFailure("Falha ao salvar DataFrame", path=path, error=str(exc))


def load_dataframe(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Carrega DataFrame Parquet; seleciona colunas se fornecidas."""
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise IOFailure("Falha ao carregar DataFrame", path=path, error=str(exc))


def list_files(prefix: str, pattern: str = "*.parquet") -> List[str]:
    """Lista arquivos sob um prefixo com pattern (default: parquet)."""
    base = os.path.join(prefix, pattern)
    return sorted(glob.glob(base))
