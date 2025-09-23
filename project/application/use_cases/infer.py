import os
import logging
import pandas as pd
from application.dto.infer_dto import InferRequest, InferResponse
from domain.services.classifier import IClassifier

logger = logging.getLogger(__name__)


class InferUseCase:
    """
    Caso de uso: inferência em novos dados.

    Responsável por carregar as features de entrada, executar predições utilizando
    um classificador, e salvar os resultados e relatórios no diretório de saída.
    """

    def __init__(self, classifier: IClassifier):
        if not isinstance(classifier, IClassifier):
            logger.error("Classificador inválido fornecido: %s", type(classifier))
            raise ValueError("O parâmetro 'classifier' deve implementar IClassifier.")
        self._classifier = classifier

    def execute(self, request: InferRequest) -> InferResponse:
        """
        Executa a inferência sobre os dados fornecidos.

        Args:
            request (InferRequest): DTO contendo diretórios de entrada, saída e perfil.

        Returns:
            InferResponse: DTO contendo status, caminho dos resultados e relatório.
        """
        if not isinstance(request, InferRequest):
            logger.error("Tipo inválido para request: %s", type(request))
            raise ValueError("O parâmetro 'request' deve ser uma instância de InferRequest.")

        logger.info("Iniciando inferência com dados em '%s'.", request.input_dir)

        try:
            X = self._load_features(request.input_dir)

            logger.debug("Features carregadas: %d amostras.", len(X))

            preds = self._classifier.predict(X)

            results_path = os.path.join(request.output_dir, "results.csv")
            report_path = os.path.join(request.output_dir, "inference_report.html")

            # Salvar predições em CSV
            pd.DataFrame(preds, columns=["prediction"]).to_csv(results_path, index=False)
            logger.info("Resultados salvos em: %s", results_path)

            # Relatório pode ser gerado posteriormente (placeholder aqui para garantir path)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("<html><body><h1>Relatório de Inferência</h1></body></html>")
            logger.info("Relatório gerado em: %s", report_path)

            return InferResponse(
                success=True,
                results_path=results_path,
                report_path=report_path,
            )

        except Exception as e:
            logger.exception("Erro durante execução da inferência: %s", str(e))
            return InferResponse(
                success=False,
                results_path="",
                report_path=None,
            )

    def _load_features(self, input_dir: str):
        """
        Carrega features do disco a partir de arquivos CSV no diretório de entrada.

        Args:
            input_dir (str): Diretório contendo os arquivos de features.

        Returns:
            pd.DataFrame: DataFrame contendo as features carregadas.
        """
        if not os.path.isdir(input_dir):
            logger.error("Diretório de entrada inválido: %s", input_dir)
            raise ValueError(f"Diretório de entrada inválido: {input_dir}")

        try:
            # Procura pelo primeiro arquivo CSV no diretório
            for file_name in os.listdir(input_dir):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(input_dir, file_name)
                    logger.info("Carregando features de: %s", file_path)
                    return pd.read_csv(file_path)

            logger.error("Nenhum arquivo CSV encontrado no diretório: %s", input_dir)
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {input_dir}")

        except Exception as e:
            logger.exception("Erro ao carregar features de %s: %s", input_dir, str(e))
            raise
