from abc import ABC, abstractmethod
from typing import Any, Dict


class IXAIExplainer(ABC):
    """
    Interface para explicabilidade de modelos (ex.: SHAP, Grad-CAM, attention maps).

    Esta interface define o contrato para implementação de explicadores responsáveis
    por gerar explicações locais e globais sobre predições de modelos.
    """

    @abstractmethod
    def explain(self, X: Any, model: Any) -> Dict[str, Any]:
        """
        Gera explicações locais e/ou globais para a entrada e modelo fornecidos.

        Args:
            X (Any): Conjunto de dados de entrada para explicação (ex.: DataFrame, array NumPy).
            model (Any): Modelo treinado para o qual as explicações serão geradas.

        Returns:
            Dict[str, Any]: Estrutura contendo explicações (ex.: valores SHAP, mapas de calor).

        Raises:
            ValueError: Se os dados de entrada ou o modelo forem inválidos.
            RuntimeError: Se ocorrer erro durante a geração das explicações.
        """
        raise NotImplementedError
