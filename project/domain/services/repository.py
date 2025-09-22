from abc import ABC, abstractmethod
from typing import Any, List


class IDataRepository(ABC):
    """Interface para repositórios de dados (persistência e cache)."""

    @abstractmethod
    def save(self, obj: Any, path: str) -> None:
        """Salva objeto em path (parquet/csv/modelos/etc.)."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> Any:
        """Carrega objeto de path."""
        raise NotImplementedError

    @abstractmethod
    def list(self, prefix: str) -> List[str]:
        """Lista arquivos/artefatos sob prefixo."""
        raise NotImplementedError
