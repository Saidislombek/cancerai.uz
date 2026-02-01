from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import run_inference
from app.model_registry import ModelRegistry

BASE_DIR = Path(__file__).resolve().parent.parent

# Новый фронтенд
PAGES_DIR = BASE_DIR / "pages"
ASSETS_DIR = BASE_DIR / "assets"

app = FastAPI(title="Cervical Cytology AI", version="1.0.0")
registry = ModelRegistry()


def _safe_page_name(page: str) -> bool:
    # защита от path traversal и странных имен
    if not page:
        return False
    if ".." in page or "/" in page or "\\" in page:
        return False
    # Доп. ограничение: только буквы/цифры/подчёркивание/дефис
    return all(ch.isalnum() or ch in ("_", "-") for ch in page)


def _file_or_404(path: Path, detail: str = "Not found") -> FileResponse:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=detail)
    # Можно добавить cache-control при желании через headers
    return FileResponse(str(path))


# Монтируем ассеты (CSS/JS/img/vendor)
if not ASSETS_DIR.exists():
    raise RuntimeError(f"ASSETS_DIR not found: {ASSETS_DIR}")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/")
def index() -> FileResponse:
    return _file_or_404(PAGES_DIR / "index.html", detail="index.html not found in /pages")


@app.get("/{page}.html")
def html_page(page: str) -> FileResponse:
    if not _safe_page_name(page):
        raise HTTPException(status_code=404, detail="Not found")
    return _file_or_404(PAGES_DIR / f"{page}.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "time": int(time.time())}


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    models = []
    for m in registry.list_specs():
        try:
            expected = str(registry.resolve_path(m.model_id))
        except Exception:
            expected = ""
        models.append(
            {
                "model_id": m.model_id,
                "display_name": m.display_name,
                "expected_file": expected,
            }
        )
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
    # базовая валидация
    if not model_id:
        raise HTTPException(status_code=422, detail="model_id is required")
    if file is None:
        raise HTTPException(status_code=422, detail="file is required")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

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

    except HTTPException as e:
        # пробрасываем понятные ошибки как есть
        raise e
    except Exception as e:
        # не валим сервер — отдаём нормальную ошибку клиенту
        return JSONResponse(status_code=400, content={"error": str(e)})