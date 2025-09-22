from application.dto.extract_features_dto import ExtractFeaturesRequest, ExtractFeaturesResponse
from typing import Protocol


class IFeatureExtractor(Protocol):
    """Protocolo para extrator de features multimodais."""

    def extract(self, input_dir: str, output_dir: str, profile: str) -> bool:
        ...


class ExtractFeaturesUseCase:
    """Caso de uso: extração de features (AUs, microexpressões, embeddings)."""

    def __init__(self, extractor: IFeatureExtractor):
        self._extractor = extractor

    def execute(self, request: ExtractFeaturesRequest) -> ExtractFeaturesResponse:
        success = self._extractor.extract(
            input_dir=request.input_dir,
            output_dir=request.output_dir,
            profile=request.profile,
        )
        return ExtractFeaturesResponse(
            success=success,
            features_dir=request.output_dir,
        )
