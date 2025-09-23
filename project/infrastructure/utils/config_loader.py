from __future__ import annotations

import os
from typing import Any, Dict

import yaml
import logging

from .errors import ConfigError

logger = logging.getLogger(__name__)


def _load_env_file(path: str = ".env") -> None:
    """
    Carrega variáveis de um arquivo .env simples (KEY=VALUE).

    Args:
        path (str): Caminho para o arquivo .env.

    Observação:
        Falhas em leitura do .env não derrubam a aplicação.
    """
    if not os.path.exists(path):
        logger.debug("Arquivo .env não encontrado em: %s", path)
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
        logger.info("Arquivo .env carregado com sucesso de: %s", path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Falha ao carregar arquivo .env (%s): %s", path, str(exc))


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
    """Mescla recursivamente dois dicionários, com override sobre base."""
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
    """
    Carrega configuração base + profile opcional, expandindo variáveis de ambiente.

    Args:
        base_path (str): Caminho do YAML base.
        profile_name (Optional[str]): Nome do profile (ex.: 'low', 'medium', 'high').
        profiles_dir (str): Diretório onde ficam os perfis.

    Returns:
        Dict[str, Any]: Dicionário de configuração expandido.

    Raises:
        ConfigError: Se arquivos não existirem ou YAML for inválido.
    """
    if not isinstance(base_path, str) or not base_path.strip():
        logger.error("Caminho base inválido: %s", base_path)
        raise ValueError("base_path deve ser uma string não vazia.")

    if profile_name is not None and (not isinstance(profile_name, str) or not profile_name.strip()):
        logger.error("Nome de profile inválido: %s", profile_name)
        raise ValueError("profile_name deve ser None ou uma string não vazia.")

    if not isinstance(profiles_dir, str) or not profiles_dir.strip():
        logger.error("Diretório de profiles inválido: %s", profiles_dir)
        raise ValueError("profiles_dir deve ser uma string não vazia.")

    _load_env_file(".env")

    if not os.path.exists(base_path):
        logger.error("Arquivo de configuração base não encontrado: %s", base_path)
        raise ConfigError(f"Arquivo de configuração não encontrado: {base_path}")

    try:
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
        logger.info("Configuração base carregada de: %s", base_path)
    except Exception as exc:
        logger.exception("Falha ao carregar config base (%s): %s", base_path, str(exc))
        raise ConfigError("Falha ao carregar config base", path=base_path, error=str(exc))

    cfg = base_cfg

    # Descobre profile pelo env/argumento
    profile_name = profile_name or os.getenv("PROFILE")
    if profile_name:
        prof_path = os.path.join(profiles_dir, f"{profile_name}.yaml")
        if not os.path.exists(prof_path):
            logger.error("Profile '%s' não encontrado em: %s", profile_name, prof_path)
            raise ConfigError(f"Profile '{profile_name}' não encontrado", path=prof_path)
        try:
            with open(prof_path, "r", encoding="utf-8") as f:
                prof_cfg = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, prof_cfg)
            logger.info("Profile '%s' carregado de: %s", profile_name, prof_path)
        except Exception as exc:
            logger.exception("Falha ao carregar profile '%s' (%s): %s", profile_name, prof_path, str(exc))
            raise ConfigError(
                f"Falha ao carregar profile '{profile_name}'",
                path=prof_path,
                error=str(exc),
            )

    # Expande env vars (${VAR})
    cfg = _env_expand(cfg)
    logger.debug("Configuração final expandida: %s", cfg)
    return cfg
