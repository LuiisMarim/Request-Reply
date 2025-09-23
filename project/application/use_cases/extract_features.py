import logging
from application.dto.extract_features_dto import ExtractFeaturesRequest, ExtractFeaturesResponse
from typing import Protocol

logger = logging.getLogger(__name__)


class IFeatureExtractor(Protocol):
    """Protocolo para extrator de features multimodais."""

    def extract(self, input_dir: str, output_dir: str, profile: str) -> bool:
        """
        Executa a extração de features multimodais.

        Args:
            input_dir (str): Caminho do diretório de entrada com dados brutos.
            output_dir (str): Caminho do diretório de saída para salvar as features.
            profile (str): Perfil de execução (ex.: 'medium', 'fast', 'accurate').

        Returns:
            bool: True se a extração foi concluída com sucesso, False caso contrário.
        """
        ...


class ExtractFeaturesUseCase:
    """
    Caso de uso: extração de features (AUs, microexpressões, embeddings).

    Responsável por orquestrar a extração de features a partir de dados brutos,
    utilizando um extrator que implemente o protocolo `IFeatureExtractor`.
    """

    def __init__(self, extractor: IFeatureExtractor):
        if not isinstance(extractor, IFeatureExtractor):
            logger.error("Extrator inválido fornecido: %s", type(extractor))
            raise ValueError("O parâmetro 'extractor' deve implementar IFeatureExtractor.")
        self._extractor = extractor

    def execute(self, request: ExtractFeaturesRequest) -> ExtractFeaturesResponse:
        """
        Executa o caso de uso de extração de features.

        Args:
            request (ExtractFeaturesRequest): DTO contendo diretórios de entrada, saída e perfil de execução.

        Returns:
            ExtractFeaturesResponse: DTO com status de sucesso e diretório de saída das features.
        """
        if not isinstance(request, ExtractFeaturesRequest):
            logger.error("Tipo inválido para request: %s", type(request))
            raise ValueError("O parâmetro 'request' deve ser uma instância de ExtractFeaturesRequest.")

        logger.info(
            "Iniciando extração de features do diretório '%s' para '%s' com perfil '%s'.",
            request.input_dir,
            request.output_dir,
            request.profile,
        )

        try:
            success = self._extractor.extract(
                input_dir=request.input_dir,
                output_dir=request.output_dir,
                profile=request.profile,
            )

            if success:
                logger.info("Extração de features concluída com sucesso. Saída em: %s", request.output_dir)
            else:
                logger.warning("Extração de features concluída com falhas.")

            return ExtractFeaturesResponse(
                success=success,
                features_dir=request.output_dir,
            )

        except Exception as e:
            logger.exception("Erro durante a execução da extração de features: %s", str(e))
            return ExtractFeaturesResponse(
                success=False,
                features_dir=request.output_dir,
                log_path=None,
            )
