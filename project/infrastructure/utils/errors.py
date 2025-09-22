from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AppError(Exception):
    """Exceção base com contexto estruturado e run_id para correlação."""

    message: str
    run_id: Optional[str] = None
    code: str = "APP_ERROR"
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - representação
        rid = f" run_id={self.run_id}" if self.run_id else ""
        return f"{self.code}: {self.message}{rid}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "run_id": self.run_id,
            "context": self.context,
        }


class DataValidationError(AppError):
    def __init__(self, message: str, run_id: Optional[str] = None, **ctx: Any) -> None:
        super().__init__(message=message, run_id=run_id, code="DATA_VALIDATION", context=ctx)


class ConfigError(AppError):
    def __init__(self, message: str, run_id: Optional[str] = None, **ctx: Any) -> None:
        super().__init__(message=message, run_id=run_id, code="CONFIG_ERROR", context=ctx)


class IOFailure(AppError):
    def __init__(self, message: str, run_id: Optional[str] = None, **ctx: Any) -> None:
        super().__init__(message=message, run_id=run_id, code="IO_FAILURE", context=ctx)


class ModelRegistryError(AppError):
    def __init__(self, message: str, run_id: Optional[str] = None, **ctx: Any) -> None:
        super().__init__(message=message, run_id=run_id, code="MODEL_REGISTRY", context=ctx)
