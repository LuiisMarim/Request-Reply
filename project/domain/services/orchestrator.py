from abc import ABC, abstractmethod
from typing import Any


class IPipelineOrchestrator(ABC):
    """Interface para orquestrar pipeline fim-a-fim."""

    @abstractmethod
    def run(self, config: Any) -> None:
        """Executa pipeline completo segundo configuração."""
        raise NotImplementedError
