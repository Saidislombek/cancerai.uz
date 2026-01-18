from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import run_inference
from app.model_registry import ModelRegistry

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Cervical Cytology AI", version="1.0.0")

registry = ModelRegistry()

# Важно: абсолютный путь к папке static (чтобы работало при любом cwd)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "time": int(time.time())}


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    models = [
        {
            "model_id": m.model_id,
            "display_name": m.display_name,
            "expected_file": str(registry.resolve_path(m.model_id)),
        }
        for m in registry.list_specs()
    ]
    return {"models": models}


@app.post("/api/admin/clear-cache")
def clear_cache() -> Dict[str, Any]:
    registry.clear_cache()
    return {"status": "ok", "message": "Model cache cleared."}


@app.post("/api/predict")
async def predict(
    model_id: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    try:
        image_bytes = await file.read()
        loaded = registry.get(model_id)
        result = run_inference(model_id, loaded, image_bytes)

        return JSONResponse(
            {
                "model_id": result.model_id,
                "predicted_label": result.predicted_label,
                "predicted_probability": result.predicted_probability,
                "probabilities": result.probabilities,
                "latency_ms": result.latency_ms,
                "input_shape": list(result.input_shape),
                "filename": file.filename,
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )