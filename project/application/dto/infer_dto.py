from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InferRequest:
    """Request DTO para inferência em novos dados."""

    input_dir: str
    models_dir: str
    output_dir: str
    profile: str = "medium"


@dataclass(frozen=True)
class InferResponse:
    """Response DTO para inferência."""

    success: bool
    results_path: str
    report_path: Optional[str] = None
