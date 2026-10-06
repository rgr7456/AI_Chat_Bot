"""HTTP API for KYC document OCR. Tenant-scoped like the face routes.

Endpoints
  POST /ocr/extract   extract structured fields (doc_type optional -> auto-detect)
  POST /ocr/read      generic OCR: every detected line + confidence
  GET  /ocr/health    readiness probe

OCR-only: extraction is returned to the caller; authority verification and
persistence are the HRMS side's job. Aadhaar numbers are masked by default
(OCR_AADHAAR_MASK) to respect UIDAI storage rules.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.auth import require_tenant
from app.service import ocr_service as svc
from app.service.ocr_service import OcrError

log = logging.getLogger("face.ocr.routes")
router = APIRouter()

_ALLOWED = {"pan", "aadhaar", "bank"}


def _handle(fn, *args):
    try:
        return fn(*args)
    except (OcrError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        log.exception("unexpected error in %s", getattr(fn, "__name__", fn))
        raise HTTPException(status_code=500, detail="internal_error")


@router.get("/health")
async def health():
    from app.ml.engine import engine
    ready = engine.ocr is not None
    return {"status": "ok" if ready else "unavailable", "ocr_loaded": ready}


@router.post("/extract")
async def extract(
    image: UploadFile = File(...),
    doc_type: Optional[str] = Form(None),
    tenant_id: str = Depends(require_tenant),
):
    if doc_type and doc_type.lower() not in _ALLOWED:
        raise HTTPException(
            status_code=422,
            detail=f"doc_type must be one of {sorted(_ALLOWED)} or omitted for auto-detect",
        )
    raw = await image.read()
    return await run_in_threadpool(_handle, svc.extract, raw, doc_type)


@router.post("/read")
async def read(
    image: UploadFile = File(...),
    tenant_id: str = Depends(require_tenant),
):
    raw = await image.read()
    return await run_in_threadpool(_handle, svc.read_text, raw)
