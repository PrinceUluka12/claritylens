# backend/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code BEFORE yield  →  runs once at server startup
    Code AFTER yield   →  runs once at server shutdown

    This replaces the old @app.on_event("startup") pattern.
    We load all ML models here so they stay alive in app.state
    for the entire lifetime of the server process.
    """
    logger.info("ClarityLens starting up...")

    # Make sure storage folders exist on disk
    settings.storage_uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_extracted_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storage dirs ready")

    # --- ML model loading (these lines get uncommented phase by phase) ---
    # from ml.inference.classifier import ClassifierInference
    # app.state.classifier = ClassifierInference(settings.ml_classifier_path)
    #
    # from ml.inference.ner import NERInference
    # app.state.ner = NERInference(settings.ml_ner_path)
    #
    # from ml.inference.qa import QAInference
    # app.state.qa = QAInference(settings.ml_qa_path)

    logger.info("ML models not yet trained — stubs active")

    yield  # <-- server is live and handling requests here

    # Shutdown: Python will garbage-collect models, but we log it clearly
    logger.info("Shutting down — releasing model memory")


# ---- Create the app ----
app = FastAPI(
    title="ClarityLens",
    description="Local ML pipeline for contract risk analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# ---- CORS ----
# The React dev server runs on port 5173 (Vite default).
# Without this, the browser blocks its own requests to port 8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Health check ----
# Always the first route you add to any API.
# The React frontend will ping this on load to show a connection indicator.
@app.get("/health", tags=["system"])
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "models_loaded": False,   # becomes True after Phase 9
    }


# ---- Routers (uncommented one by one as phases complete) ----
# app.include_router(ingest.router,    prefix="/api/ingest")
# app.include_router(classify.router,  prefix="/api/classify")
# app.include_router(ner.router,       prefix="/api/ner")
# app.include_router(summarize.router, prefix="/api/summarize")
# app.include_router(qa.router,        prefix="/api/qa")