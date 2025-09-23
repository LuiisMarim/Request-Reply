from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from joblib import dump, load
import logging

from infrastructure.utils.errors import ModelRegistryError

logger = logging.getLogger(__name__)


def _hash_bytes(data: bytes) -> str:
    """Calcula hash SHA256 abreviado de dados binários."""
    return hashlib.sha256(data).hexdigest()[:16]


class ModelRegistry:
    """
    Registro leve de modelos com versionamento por timestamp + hash.

    Estrutura:
    models_dir/
      <model_name>/
        <timestamp>_<hash>/
          model.joblib
          metadata.json
    """

    def __init__(self, models_dir: str) -> None:
        if not isinstance(models_dir, str) or not models_dir.strip():
            logger.error("Diretório inválido para ModelRegistry: %s", models_dir)
            raise ValueError("models_dir deve ser uma string não vazia.")
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info("ModelRegistry inicializado em: %s", self.models_dir)

    def _model_dir(self, name: str) -> str:
        return os.path.join(self.models_dir, name)

    def _artifact_dir(self, name: str, version: str) -> str:
        return os.path.join(self._model_dir(name), version)

    def register(self, model: Any, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Registra um modelo no repositório.

        Args:
            model (Any): Modelo treinado a ser salvo.
            name (str): Nome do modelo.
            metadata (Optional[Dict[str, Any]]): Metadados adicionais.

        Returns:
            str: Versão criada no formato <timestamp>_<hash>.

        Raises:
            ModelRegistryError: Se ocorrer falha ao registrar.
        """
        if model is None:
            logger.error("Tentativa de registrar modelo None.")
            raise ValueError("model não pode ser None.")
        if not isinstance(name, str) or not name.strip():
            logger.error("Nome inválido para modelo: %s", name)
            raise ValueError("name deve ser uma string não vazia.")

        try:
            os.makedirs(self._model_dir(name), exist_ok=True)

            # Dump temporário para gerar hash
            tmp_path = os.path.join(self._model_dir(name), "__tmp_model.joblib")
            dump(model, tmp_path)
            with open(tmp_path, "rb") as f:
                model_bytes = f.read()
            os.remove(tmp_path)

            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            hsh = _hash_bytes(model_bytes)
            version = f"{ts}_{hsh}"

            art_dir = self._artifact_dir(name, version)
            os.makedirs(art_dir, exist_ok=True)
            model_path = os.path.join(art_dir, "model.joblib")
            meta_path = os.path.join(art_dir, "metadata.json")

            dump(model, model_path)
            meta = {
                "name": name,
                "version": version,
                "created_at": ts,
                "hash": hsh,
                "metadata": metadata or {},
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            logger.info("Modelo '%s' registrado com versão %s.", name, version)
            return version
        except Exception as exc:
            logger.exception("Falha ao registrar modelo '%s': %s", name, str(exc))
            raise ModelRegistryError("Falha ao registrar modelo", name=name, error=str(exc))

    def list_versions(self, name: str) -> List[str]:
        """
        Lista versões disponíveis para um modelo.

        Args:
            name (str): Nome do modelo.

        Returns:
            List[str]: Lista de versões ordenadas.

        Raises:
            ModelRegistryError: Se ocorrer falha na listagem.
        """
        if not isinstance(name, str) or not name.strip():
            logger.error("Nome inválido fornecido a list_versions: %s", name)
            raise ValueError("name deve ser uma string não vazia.")

        try:
            path = self._model_dir(name)
            if not os.path.exists(path):
                return []
            versions = [
                d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
            ]
            logger.info("Foram encontradas %d versões para o modelo '%s'.", len(versions), name)
            return sorted(versions)
        except Exception as exc:
            logger.exception("Falha ao listar versões do modelo '%s': %s", name, str(exc))
            raise ModelRegistryError("Falha ao listar versões", name=name, error=str(exc))

    def get_path(self, name: str, version: str) -> str:
        """Retorna o caminho absoluto para o arquivo do modelo especificado."""
        return os.path.join(self._artifact_dir(name, version), "model.joblib")

    def load(self, name: str, version: Optional[str] = None) -> Any:
        """
        Carrega um modelo por versão.

        Args:
            name (str): Nome do modelo.
            version (Optional[str]): Versão específica. Se None, carrega a última.

        Returns:
            Any: Modelo carregado.

        Raises:
            ModelRegistryError: Se falhar ao carregar.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name deve ser uma string não vazia.")

        try:
            if version is None:
                versions = self.list_versions(name)
                if not versions:
                    raise ModelRegistryError("Nenhuma versão encontrada", name=name)
                version = versions[-1]
            model_path = self.get_path(name, version)
            model = load(model_path)
            logger.info("Modelo '%s' carregado com versão %s.", name, version)
            return model
        except ModelRegistryError:
            raise
        except Exception as exc:
            logger.exception("Falha ao carregar modelo '%s' versão %s: %s", name, version, str(exc))
            raise ModelRegistryError(
                "Falha ao carregar modelo", name=name, version=version, error=str(exc)
            )

    def load_metadata(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Carrega metadados de um modelo.

        Args:
            name (str): Nome do modelo.
            version (Optional[str]): Versão específica. Se None, usa a última.

        Returns:
            Dict[str, Any]: Metadados carregados.

        Raises:
            ModelRegistryError: Se falhar ao carregar metadados.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name deve ser uma string não vazia.")

        try:
            if version is None:
                versions = self.list_versions(name)
                if not versions:
                    raise ModelRegistryError("Nenhuma versão encontrada", name=name)
                version = versions[-1]
            meta_path = os.path.join(self._artifact_dir(name, version), "metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            logger.info("Metadados carregados para modelo '%s' versão %s.", name, version)
            return meta
        except ModelRegistryError:
            raise
        except Exception as exc:
            logger.exception("Falha ao carregar metadados do modelo '%s' versão %s: %s", name, version, str(exc))
            raise ModelRegistryError(
                "Falha ao carregar metadados", name=name, version=version, error=str(exc)
            )
