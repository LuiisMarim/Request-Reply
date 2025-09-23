from __future__ import annotations

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Dict, Any

from application.use_cases.validate_data import ValidateDataUseCase
from application.use_cases.extract_features import ExtractFeaturesUseCase
from application.use_cases.train import TrainUseCase
from application.use_cases.infer import InferUseCase

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TEA Biomarker API",
    description="API para pipeline de análise de expressões faciais em TEA",
    version="0.1.0",
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Endpoint raiz de saúde do serviço."""
    logger.info("Requisição recebida em GET /")
    return {"status": "ok", "message": "TEA Biomarker API running"}


@app.post("/validate")
async def validate(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Valida dataset enviado pelo usuário.

    Args:
        file (UploadFile): Arquivo compactado contendo imagens ou dataset.

    Returns:
        Dict[str, Any]: Resultado da validação.
    """
    if not file:
        logger.error("Nenhum arquivo fornecido para validação.")
        raise HTTPException(status_code=400, detail="Nenhum arquivo fornecido.")

    try:
        logger.info("Iniciando validação do arquivo: %s", file.filename)
        uc = ValidateDataUseCase()
        result = uc.execute(file)  # assume que o UseCase implementa método execute
        logger.info("Validação concluída para arquivo: %s", file.filename)
        return {"status": "success", "file": file.filename, "result": result}
    except Exception as e:
        logger.exception("Erro durante validação do arquivo %s: %s", file.filename, str(e))
        raise HTTPException(status_code=500, detail=f"Erro na validação: {str(e)}")


@app.post("/extract")
async def extract() -> Dict[str, Any]:
    """Extrai features do dataset previamente validado."""
    try:
        logger.info("Iniciando extração de features.")
        uc = ExtractFeaturesUseCase()
        result = uc.execute()
        logger.info("Extração de features concluída.")
        return {"status": "success", "task": "extract_features", "result": result}
    except Exception as e:
        logger.exception("Erro durante extração de features: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Erro na extração: {str(e)}")


@app.post("/train")
async def train() -> Dict[str, Any]:
    """Treina modelos usando as features extraídas."""
    try:
        logger.info("Iniciando treinamento de modelo.")
        uc = TrainUseCase()
        result = uc.execute()
        logger.info("Treinamento concluído.")
        return {"status": "success", "task": "train", "result": result}
    except Exception as e:
        logger.exception("Erro durante treinamento: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")


@app.post("/infer")
async def infer() -> Dict[str, Any]:
    """Executa inferência em dados fornecidos usando modelo treinado."""
    try:
        logger.info("Iniciando inferência.")
        uc = InferUseCase()
        result = uc.execute()
        logger.info("Inferência concluída.")
        return {"status": "success", "task": "infer", "result": result}
    except Exception as e:
        logger.exception("Erro durante inferência: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {str(e)}")
