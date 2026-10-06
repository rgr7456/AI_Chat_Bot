"""FastAPI entrypoint for the HRMS face service.

Models are loaded once at startup (not at import). The pgvector pool is opened
on startup and closed on shutdown.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.pool import init_pool, close_pool
from app.ml.engine import engine
from app.controller.face_routes import router as face_router
from app.controller.ocr_routes import router as ocr_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("face.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading face models...")
    engine.load()
    engine.warmup()
    log.info("Models loaded. Opening pgvector pool...")
    init_pool()
    log.info("Startup complete.")
    yield
    close_pool()
    log.info("Shutdown complete.")


app = FastAPI(title="HRMS Face Service", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(face_router, prefix="/face", tags=["Face"])
app.include_router(ocr_router, prefix="/ocr", tags=["OCR / KYC"])


@app.get("/")
async def root():
    return {"service": "hrms-face", "status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
