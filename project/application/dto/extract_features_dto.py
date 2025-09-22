from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExtractFeaturesRequest:
    """Request DTO para extração de features."""

    input_dir: str
    output_dir: str
    profile: str = "medium"


@dataclass(frozen=True)
class ExtractFeaturesResponse:
    """Response DTO para extração de features."""

    success: bool
    features_dir: str
    log_path: Optional[str] = None
