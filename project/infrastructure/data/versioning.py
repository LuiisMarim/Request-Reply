from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def compute_file_hash(path: str, block_size: int = 65536) -> str:
    """
    Computa o hash SHA256 de um arquivo em blocos.

    Args:
        path (str): Caminho do arquivo.
        block_size (int): Tamanho do bloco em bytes para leitura incremental.

    Returns:
        str: Hash SHA256 abreviado (16 primeiros caracteres).

    Raises:
        ValueError: Se o arquivo não existir.
        IOError: Se ocorrer erro de leitura.
    """
    if not os.path.isfile(path):
        logger.error("Arquivo não encontrado para hashing: %s", path)
        raise ValueError(f"Arquivo não encontrado: {path}")

    sha = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                sha.update(block)
        hash_value = sha.hexdigest()[:16]
        logger.debug("Hash computado para %s: %s", path, hash_value)
        return hash_value
    except Exception as e:
        logger.exception("Erro ao calcular hash de %s: %s", path, str(e))
        raise


def save_version_metadata(dataset_dir: str, out_path: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Salva metadados de versionamento do dataset (hashes + timestamp).

    Args:
        dataset_dir (str): Diretório raiz do dataset.
        out_path (str): Caminho do arquivo JSON de saída.
        extra_meta (Optional[Dict[str, Any]]): Metadados adicionais.

    Returns:
        str: Caminho do arquivo JSON gerado.

    Raises:
        ValueError: Se o diretório do dataset não existir.
        IOError: Se ocorrer erro ao salvar o arquivo de metadados.
    """
    if not os.path.isdir(dataset_dir):
        logger.error("Diretório de dataset inválido: %s", dataset_dir)
        raise ValueError(f"Diretório de dataset inválido: {dataset_dir}")

    logger.info("Iniciando versionamento do dataset em: %s", dataset_dir)
    records: Dict[str, Any] = {}

    try:
        for root, _, files in os.walk(dataset_dir):
            for fn in files:
                fpath = os.path.join(root, fn)
                rel = os.path.relpath(fpath, dataset_dir)
                try:
                    records[rel] = compute_file_hash(fpath)
                except Exception as e:
                    logger.warning("Falha ao processar arquivo %s: %s", fpath, str(e))
                    continue

        meta = {
            "dataset_dir": dataset_dir,
            "timestamp": datetime.utcnow().isoformat(),
            "files": records,
            "extra": extra_meta or {},
        }

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Metadados de versionamento salvos em: %s", out_path)
        return out_path

    except Exception as e:
        logger.exception("Erro ao salvar metadados de versionamento: %s", str(e))
        raise
