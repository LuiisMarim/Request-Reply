import logging
from enum import Enum

logger = logging.getLogger(__name__)


class Label(Enum):
    """
    Representa o rótulo clínico da amostra.

    Valores possíveis:
        AUTISTIC: Indica amostra com diagnóstico de autismo.
        NON_AUTISTIC: Indica amostra sem diagnóstico de autismo.
    """

    AUTISTIC = "autistic"
    NON_AUTISTIC = "non_autistic"

    def __str__(self) -> str:
        """
        Retorna o valor em string do rótulo.

        Returns:
            str: Valor associado ao rótulo.
        """
        return self.value

    @staticmethod
    def from_str(label_str: str) -> "Label":
        """
        Converte uma string em um objeto `Label`.

        Args:
            label_str (str): String representando o rótulo.

        Returns:
            Label: Instância correspondente do enum.

        Raises:
            ValueError: Se a string não corresponder a nenhum rótulo válido.
        """
        if not isinstance(label_str, str):
            logger.error("Tipo inválido para conversão de Label: %s", type(label_str))
            raise ValueError("O parâmetro 'label_str' deve ser uma string.")

        normalized = label_str.strip().lower()
        for label in Label:
            if label.value == normalized:
                logger.info("Conversão bem-sucedida de string '%s' para Label '%s'.", label_str, label)
                return label

        logger.error("String inválida para Label: %s", label_str)
        raise ValueError(f"Rótulo inválido: {label_str}. Valores aceitos: {[l.value for l in Label]}")
