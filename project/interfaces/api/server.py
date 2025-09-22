from __future__ import annotations

from fastapi import FastAPI, UploadFile
from typing import Dict

from application.use_cases.validate_data import ValidateDataUseCase
from application.use_cases.extract_features import ExtractFeaturesUseCase
from application.use_cases.train import TrainUseCase
from application.use_cases.infer import InferUseCase


app = FastAPI(
    title="TEA Biomarker API",
    description="API para pipeline de análise de expressões faciais em TEA",
    version="0.1.0",
)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok", "message": "TEA Biomarker API running"}


@app.post("/validate")
async def validate(file: UploadFile) -> Dict[str, str]:
    # Placeholder: delegar para use case
    uc = ValidateDataUseCase()
    # Implementação futura: ler e validar arquivo
    return {"status": "pending", "file": file.filename}


@app.post("/extract")
async def extract() -> Dict[str, str]:
    uc = ExtractFeaturesUseCase()
    return {"status": "pending", "task": "extract_features"}


@app.post("/train")
async def train() -> Dict[str, str]:
    uc = TrainUseCase()
    return {"status": "pending", "task": "train"}


@app.post("/infer")
async def infer() -> Dict[str, str]:
    uc = InferUseCase()
    return {"status": "pending", "task": "infer"}
