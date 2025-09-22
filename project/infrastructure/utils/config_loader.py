from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from .errors import ConfigError


def _load_env_file(path: str = ".env") -> None:
    """Carrega variáveis de um arquivo .env simples (KEY=VALUE), sem dependências extras."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
    except Exception as exc:  # pragma: no cover
        # Falha em ler .env não deve derrubar a aplicação.
        pass


def _env_expand(value: Any) -> Any:
    """Expande ${VAR} em strings, recursivamente em dicts/listas."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _env_expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_env_expand(v) for v in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    base_path: str = "config/config.yaml",
    profile_name: str | None = None,
    profiles_dir: str = "config/profiles",
) -> Dict[str, Any]:
    """Carrega configuração base + profile opcional, expande variáveis de ambiente.

    Args:
        base_path: caminho do YAML base.
        profile_name: nome do profile (ex.: 'low', 'medium', 'high').
        profiles_dir: diretório dos perfis.

    Returns:
        Dicionário de configuração expandido.

    Raises:
        ConfigError: se arquivos não existirem ou YAML inválido.
    """
    _load_env_file(".env")

    if not os.path.exists(base_path):
        raise ConfigError(f"Arquivo de configuração não encontrado: {base_path}")

    try:
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        raise ConfigError("Falha ao carregar config base", path=base_path, error=str(exc))

    cfg = base_cfg

    # Descobre profile pelo env/argumento
    profile_name = profile_name or os.getenv("PROFILE")
    if profile_name:
        prof_path = os.path.join(profiles_dir, f"{profile_name}.yaml")
        if not os.path.exists(prof_path):
            raise ConfigError(f"Profile '{profile_name}' não encontrado", path=prof_path)
        try:
            with open(prof_path, "r", encoding="utf-8") as f:
                prof_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, prof_cfg)
        except Exception as exc:
            raise ConfigError(
                f"Falha ao carregar profile '{profile_name}'",
                path=prof_path,
                error=str(exc),
            )

    # Expande env vars (${VAR})
    cfg = _env_expand(cfg)
    return cfg
