from enum import Enum


class Label(Enum):
    """Representa o rótulo clínico da amostra."""

    AUTISTIC = "autistic"
    NON_AUTISTIC = "non_autistic"

    def __str__(self) -> str:
        return self.value
