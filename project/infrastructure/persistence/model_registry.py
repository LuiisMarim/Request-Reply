from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from joblib import dump, load

from infrastructure.utils.errors import ModelRegistryError


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class ModelRegistry:
    """Registro leve de modelos com versionamento por timestamp + hash.

    Estrutura:
    models_dir/
      <model_name>/
        <timestamp>_<hash>/
          model.joblib
          metadata.json
    """

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def _model_dir(self, name: str) -> str:
        return os.path.join(self.models_dir, name)

    def _artifact_dir(self, name: str, version: str) -> str:
        return os.path.join(self._model_dir(name), version)

    def register(self, model: Any, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Registra um modelo, retornando a versão criada."""
        try:
            os.makedirs(self._model_dir(name), exist_ok=True)

            # Dump first to bytes-like temp for hashing
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

            return version
        except Exception as exc:
            raise ModelRegistryError("Falha ao registrar modelo", name=name, error=str(exc))

    def list_versions(self, name: str) -> List[str]:
        """Lista versões disponíveis (ordenadas)."""
        try:
            path = self._model_dir(name)
            if not os.path.exists(path):
                return []
            versions = [
                d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
            ]
            return sorted(versions)
        except Exception as exc:
            raise ModelRegistryError("Falha ao listar versões", name=name, error=str(exc))

    def get_path(self, name: str, version: str) -> str:
        return os.path.join(self._artifact_dir(name, version), "model.joblib")

    def load(self, name: str, version: Optional[str] = None) -> Any:
        """Carrega um modelo por versão; se None, carrega a última."""
        try:
            if version is None:
                versions = self.list_versions(name)
                if not versions:
                    raise ModelRegistryError("Nenhuma versão encontrada", name=name)
                version = versions[-1]
            model_path = self.get_path(name, version)
            return load(model_path)
        except ModelRegistryError:
            raise
        except Exception as exc:
            raise ModelRegistryError(
                "Falha ao carregar modelo", name=name, version=version, error=str(exc)
            )

    def load_metadata(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Carrega metadados do modelo."""
        try:
            if version is None:
                versions = self.list_versions(name)
                if not versions:
                    raise ModelRegistryError("Nenhuma versão encontrada", name=name)
                version = versions[-1]
            meta_path = os.path.join(self._artifact_dir(name, version), "metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except ModelRegistryError:
            raise
        except Exception as exc:
            raise ModelRegistryError(
                "Falha ao carregar metadados", name=name, version=version, error=str(exc)
            )
