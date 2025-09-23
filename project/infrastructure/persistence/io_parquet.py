from __future__ import annotations

import glob
import os
from typing import List, Optional

import pandas as pd
import logging

from infrastructure.utils.errors import IOFailure

logger = logging.getLogger(__name__)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Salva DataFrame em formato Parquet (pyarrow).

    Args:
        df (pd.DataFrame): DataFrame a ser salvo.
        path (str): Caminho de saída do arquivo Parquet.

    Raises:
        IOFailure: Se ocorrer falha no salvamento.
    """
    if not isinstance(path, str) or not path.strip():
        logger.error("Caminho inválido para salvar DataFrame: %s", path)
        raise ValueError("path deve ser uma string não vazia.")
    if df is None or df.empty:
        logger.warning("Tentativa de salvar DataFrame vazio em: %s", path)

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("DataFrame salvo com sucesso em: %s", path)
    except Exception as exc:
        logger.exception("Falha ao salvar DataFrame em %s: %s", path, str(exc))
        raise IOFailure("Falha ao salvar DataFrame", path=path, error=str(exc))


def load_dataframe(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Carrega DataFrame de arquivo Parquet.

    Args:
        path (str): Caminho do arquivo Parquet.
        columns (Optional[List[str]]): Colunas específicas a carregar.

    Returns:
        pd.DataFrame: DataFrame carregado.

    Raises:
        IOFailure: Se ocorrer falha no carregamento.
    """
    if not isinstance(path, str) or not path.strip():
        logger.error("Caminho inválido para carregar DataFrame: %s", path)
        raise ValueError("path deve ser uma string não vazia.")

    try:
        df = pd.read_parquet(path, columns=columns)
        logger.info("DataFrame carregado de %s com %d linhas.", path, len(df))
        return df
    except Exception as exc:
        logger.exception("Falha ao carregar DataFrame de %s: %s", path, str(exc))
        raise IOFailure("Falha ao carregar DataFrame", path=path, error=str(exc))


def list_files(prefix: str, pattern: str = "*.parquet") -> List[str]:
    """
    Lista arquivos sob um prefixo com padrão fornecido.

    Args:
        prefix (str): Diretório base.
        pattern (str): Padrão de arquivos (default: '*.parquet').

    Returns:
        List[str]: Lista de caminhos de arquivos encontrados.
    """
    if not isinstance(prefix, str) or not prefix.strip():
        logger.error("Prefixo inválido para listagem de arquivos: %s", prefix)
        raise ValueError("prefix deve ser uma string não vazia.")

    base = os.path.join(prefix, pattern)
    files = sorted(glob.glob(base))
    logger.info("Listados %d arquivos em %s com padrão '%s'.", len(files), prefix, pattern)
    return files
