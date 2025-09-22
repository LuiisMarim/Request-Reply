from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ValidateDataRequest:
    """Request DTO para validação de dados."""

    input_dir: str
    output_dir: str
    report_path: Optional[str] = None


@dataclass(frozen=True)
class ValidateDataResponse:
    """Response DTO para validação de dados."""

    success: bool
    report_path: str
