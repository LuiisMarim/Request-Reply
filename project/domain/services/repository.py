from abc import ABC, abstractmethod
from typing import Any, List


class IDataRepository(ABC):
    """
    Interface para repositórios de dados (persistência e cache).

    Define o contrato para implementações responsáveis por salvar,
    carregar e listar objetos em diferentes formatos (parquet, CSV, modelos, etc.).
    """

    @abstractmethod
    def save(self, obj: Any, path: str) -> None:
        """
        Salva um objeto em disco.

        Args:
            obj (Any): Objeto a ser persistido (ex.: DataFrame, modelo treinado).
            path (str): Caminho completo para salvar o objeto.

        Returns:
            None

        Raises:
            ValueError: Se o caminho for inválido.
            IOError: Se ocorrer falha durante a escrita no disco.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> Any:
        """
        Carrega um objeto a partir do disco.

        Args:
            path (str): Caminho completo do arquivo a ser carregado.

        Returns:
            Any: Objeto carregado (ex.: DataFrame, modelo treinado).

        Raises:
            ValueError: Se o caminho for inválido.
            FileNotFoundError: Se o arquivo não for encontrado.
            IOError: Se ocorrer falha durante a leitura.
        """
        raise NotImplementedError

    @abstractmethod
    def list(self, prefix: str) -> List[str]:
        """
        Lista os arquivos ou artefatos salvos sob um prefixo.

        Args:
            prefix (str): Diretório ou prefixo para busca.

        Returns:
            List[str]: Lista de caminhos de arquivos encontrados.

        Raises:
            ValueError: Se o prefixo for inválido.
            IOError: Se ocorrer falha durante a listagem de arquivos.
        """
        raise NotImplementedError
