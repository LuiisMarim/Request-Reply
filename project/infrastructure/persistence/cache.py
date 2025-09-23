from __future__ import annotations

import os
from typing import Any

from joblib import Memory, dump, load
import logging

from infrastructure.utils.errors import IOFailure

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Cache simples baseado em joblib.Memory + utilitários dump/load.

    - Útil para memorizar funções caras (ex.: extração de embeddings).
    - Também expõe dump/load para objetos arbitrários.
    """

    def __init__(self, cache_dir: str, verbose: int = 0) -> None:
        if not isinstance(cache_dir, str) or not cache_dir.strip():
            logger.error("Diretório de cache inválido: %s", cache_dir)
            raise ValueError("cache_dir deve ser uma string não vazia.")

        os.makedirs(cache_dir, exist_ok=True)
        self.memory = Memory(location=cache_dir, verbose=verbose)
        logger.info("CacheManager inicializado em: %s", cache_dir)

    def cache_fn(self, fn):
        """
        Decora função com cache (memoization).

        Args:
            fn (Callable): Função a ser cacheada.

        Returns:
            Callable: Função decorada com cache.
        """
        if not callable(fn):
            logger.error("Objeto inválido fornecido para cache_fn: %s", type(fn))
            raise ValueError("fn deve ser uma função ou callable.")
        return self.memory.cache(fn)

    def save(self, obj: Any, path: str) -> None:
        """
        Salva objeto arbitrário no cache.

        Args:
            obj (Any): Objeto a ser salvo.
            path (str): Caminho para salvar o objeto.

        Raises:
            IOFailure: Se falhar ao salvar.
        """
        if not isinstance(path, str) or not path.strip():
            logger.error("Caminho inválido para salvar no cache: %s", path)
            raise ValueError("path deve ser uma string não vazia.")

        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            dump(obj, path)
            logger.info("Objeto salvo no cache em: %s", path)
        except Exception as exc:
            logger.exception("Falha ao salvar objeto em cache: %s", str(exc))
            raise IOFailure("Falha ao salvar objeto em cache", path=path, error=str(exc))

    def load(self, path: str) -> Any:
        """
        Carrega objeto previamente salvo no cache.

        Args:
            path (str): Caminho do objeto salvo.

        Returns:
            Any: Objeto carregado.

        Raises:
            IOFailure: Se falhar ao carregar.
        """
        if not isinstance(path, str) or not path.strip():
            logger.error("Caminho inválido para carregar do cache: %s", path)
            raise ValueError("path deve ser uma string não vazia.")

        try:
            obj = load(path)
            logger.info("Objeto carregado do cache em: %s", path)
            return obj
        except Exception as exc:
            logger.exception("Falha ao carregar objeto do cache: %s", str(exc))
            raise IOFailure("Falha ao carregar objeto do cache", path=path, error=str(exc))
