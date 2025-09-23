from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AppError(Exception):
    """
    Exceção base com contexto estruturado e run_id para correlação.

    Attributes:
        message (str): Mensagem descritiva do erro.
        run_id (Optional[str]): Identificador da execução para rastreabilidade.
        code (str): Código de erro padronizado.
        context (Dict[str, Any]): Contexto adicional associado ao erro.
    """

    message: str
    run_id: Optional[str] = None
    code: str = "APP_ERROR"
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.message, str) or not self.message.strip():
            logger.error("Mensagem inválida para AppError: %s", self.message)
            raise ValueError("message deve ser uma string não vazia.")
        if not isinstance(self.context, dict):
            logger.warning("Contexto inválido para AppError, convertendo para dict: %s", type(self.context))
            self.context = dict(self.context or {})
        logger.error("Exceção %s criada: %s", self.code, self.message)

    def __str__(self) -> str:  # pragma: no cover - representação
        rid = f" run_id={self.run_id}" if self.run_id else ""
        return f"{self.code}: {self.message}{rid}"

    def to_dict(self) -> Dict[str, Any]:
        """Converte a exceção para um dicionário serializável."""
        return {
            "code": self.code,
            "message": self.message,
            "run_id": self.run_id,
            "context": self.context,
        }


class DataValidationError(AppError):
    """Erro lançado em falhas de validação de dados."""

    def __init__(self, message: str, run_id: Optional[str] = None, **ctx: Any) -> None:
        super().__init__(message=message, run_id=run_id, code="DATA_VALIDATION", context=ctx)


class ConfigError(AppError):
    """Erro lançado em falhas de configuração."""

    def __init__(self, message: str, run_id: Optional[str] = None, **ctx: Any) -> None:
        super().__init__(message=message, run_id=run_id, code="CONFIG_ERROR", context=ctx)


class IOFailure(AppError):
    """Erro lançado em falhas de entrada/saída (I/O)."""

    def __init__(self, message: str, run_id: Optional[str] = None, **ctx: Any) -> None:
        super().__init__(message=message, run_id=run_id, code="IO_FAILURE", context=ctx)


class ModelRegistryError(AppError):
    """Erro lançado em falhas relacionadas ao registro de modelos."""

    def __init__(self, message: str, run_id: Optional[str] = None, **ctx: Any) -> None:
        super().__init__(message=message, run_id=run_id, code="MODEL_REGISTRY", context=ctx)
