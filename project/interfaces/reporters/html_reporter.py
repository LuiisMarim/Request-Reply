from __future__ import annotations

import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class HTMLReporter:
    """Gera relatórios HTML simples e autoexplicativos."""

    def __init__(self, out_dir: str = "artifacts/reports") -> None:
        if not isinstance(out_dir, str) or not out_dir.strip():
            logger.error("Diretório inválido para HTMLReporter: %s", out_dir)
            raise ValueError("out_dir deve ser uma string não vazia.")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        logger.info("HTMLReporter inicializado em: %s", self.out_dir)

    def write(self, data: Dict[str, Any], name: str, title: str = "Report") -> str:
        """
        Salva relatório em formato HTML.

        Args:
            data (Dict[str, Any]): Dados a incluir no relatório.
            name (str): Nome base do arquivo (sem extensão).
            title (str): Título exibido no relatório.

        Returns:
            str: Caminho do arquivo HTML gerado.

        Raises:
            ValueError: Se data for inválido.
            RuntimeError: Se falhar a escrita do arquivo.
        """
        if not isinstance(data, dict) or not data:
            logger.error("Dados inválidos fornecidos ao HTMLReporter: %s", data)
            raise ValueError("data deve ser um dicionário não vazio.")

        if not isinstance(name, str) or not name.strip():
            logger.error("Nome inválido para relatório HTML: %s", name)
            raise ValueError("name deve ser uma string não vazia.")

        path = os.path.join(self.out_dir, f"{name}.html")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("<html><head><meta charset='utf-8'>")
                f.write(f"<title>{title}</title></head><body>")
                f.write(f"<h1>{title}</h1><ul>")
                for k, v in data.items():
                    f.write(f"<li><b>{k}</b>: {v}</li>")
                f.write("</ul>")
                f.write(
                    "<p style='font-size:12px;color:#666'>Nota ética: Ferramenta de apoio à decisão; "
                    "não substitui avaliação multiprofissional. Requer aprovação ética e conformidade "
                    "LGPD/GDPR. Uso sob supervisão de profissional de saúde.</p>"
                )
                f.write("</body></html>")
            logger.info("Relatório HTML salvo em: %s", path)
            return path
        except Exception as e:
            logger.exception("Falha ao salvar relatório HTML em %s: %s", path, str(e))
            raise RuntimeError(f"Falha ao salvar relatório HTML: {str(e)}") from e
