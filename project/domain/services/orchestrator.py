from abc import ABC, abstractmethod
from typing import Any


class IPipelineOrchestrator(ABC):
    """
    Interface para orquestração de pipeline fim-a-fim.

    Esta interface define o contrato para implementações que coordenam
    todas as etapas do pipeline de dados, desde pré-processamento até
    inferência, avaliação e geração de relatórios.
    """

    @abstractmethod
    def run(self, config: Any) -> None:
        """
        Executa o pipeline completo de acordo com a configuração fornecida.

        Args:
            config (Any): Estrutura de configuração contendo parâmetros necessários
                          (ex.: caminhos de entrada/saída, hiperparâmetros, flags de auditoria).

        Returns:
            None

        Raises:
            ValueError: Se a configuração for inválida ou incompleta.
            RuntimeError: Se ocorrer falha em alguma etapa do pipeline.
        """
        raise NotImplementedError
