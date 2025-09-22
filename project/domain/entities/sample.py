from dataclasses import dataclass
from typing import Dict, Optional
from .label import Label


@dataclass(frozen=True)
class Sample:
    """Representa uma amostra de imagem ou sequência de imagens."""

    id: str
    file_path: str
    label: Optional[Label]
    demographics: Dict[str, str]  # idade, gênero, etnia (se disponível)
    metadata: Dict[str, str]  # dimensões, canais, qualidade, etc.

    def is_labeled(self) -> bool:
        """Verifica se a amostra possui rótulo definido."""
        return self.label is not None
