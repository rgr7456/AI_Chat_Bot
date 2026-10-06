"""OCR orchestration: decode -> OCR -> (auto-detect) -> parse structured fields.

Stateless and DB-free by design: this service extracts document fields and hands
them back. Business validation and persistence are the caller's responsibility
(the HRMS side), consistent with the "verify-only" posture of the face service.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.core.config import settings
from app.ml import doc_parsers
from app.ml.engine import engine, decode_image

log = logging.getLogger("face.ocr.service")


class OcrError(Exception):
    """Expected, client-facing OCR condition -> HTTP 422."""


def _require_engine():
    if engine.ocr is None:
        raise OcrError(
            "OCR engine is not available. Install rapidocr-onnxruntime and restart."
        )
    return engine.ocr


def read_text(image_bytes: bytes) -> Dict:
    """Generic OCR: return every detected line + confidence, no document parsing."""
    ocr = _require_engine()
    img = decode_image(image_bytes)
    result = ocr.read(img)
    return {
        "lines": [{"text": l.text, "confidence": round(l.confidence, 4)} for l in result.lines],
        "text": result.text,
        "mean_confidence": round(result.mean_confidence, 4),
    }


def extract(image_bytes: bytes, doc_type: Optional[str] = None) -> Dict:
    """Extract structured fields for an Indian KYC document.

    `doc_type` is one of pan/aadhaar/bank; when omitted it is auto-detected from
    the OCR text. Returns the parsed fields plus `raw_lines` and confidence so
    the caller can apply its own validation.
    """
    ocr = _require_engine()
    img = decode_image(image_bytes)
    result = ocr.read(img)
    lines: List[str] = [l.text for l in result.lines]

    resolved = (doc_type or "").lower() or doc_parsers.detect_doc_type(result.text)
    if not resolved:
        raise OcrError(
            "Could not determine document type from the image. "
            "Pass doc_type=pan|aadhaar|bank explicitly, or retake a clearer photo."
        )
    if resolved not in doc_parsers.PARSERS:
        raise OcrError(f"unsupported document type: {resolved!r}")

    try:
        fields = doc_parsers.parse(
            resolved, lines, mask_aadhaar=settings.OCR_AADHAAR_MASK
        )
    except ValueError as e:
        raise OcrError(str(e))

    return {
        "doc_type": resolved,
        "auto_detected": doc_type in (None, ""),
        "fields": fields,
        "mean_confidence": round(result.mean_confidence, 4),
        "raw_lines": lines,
    }
