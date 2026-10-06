"""OCR text extraction with PaddleOCR PP-OCRv4 models, run via ONNX Runtime.

We use `rapidocr-onnxruntime`, which packages PaddleOCR's detection + angle +
recognition models as ONNX and runs them purely on onnxruntime (Apache-2.0,
no PyTorch/Paddle at runtime) -- matching this service's "OpenCV + ONNX only"
stack. The rest of the app depends only on the thin `OcrEngine` wrapper below,
so the backend can be swapped without touching the parsers or service layer.

Returns word/line boxes with text + confidence. Angle classification is on so
rotated/skewed ID photos are handled.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("face.ocr")

# Cap the longest edge before OCR. Full-res phone photos of ID cards (3-4k px)
# make the detection model allocate huge buffers and can OOM a small instance;
# 1600px keeps IDs/PAN/Aadhaar text crisp while bounding memory and latency.
_MAX_OCR_SIDE = 1600


def _downscale_for_ocr(image: np.ndarray, max_side: int = _MAX_OCR_SIDE) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(
        image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
    )


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: Tuple[Tuple[float, float], ...]  # 4 polygon points, clockwise


@dataclass
class OcrResult:
    lines: List[OcrLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        """All detected text joined top-to-bottom, one line per detection."""
        return "\n".join(l.text for l in self.lines)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return float(sum(l.confidence for l in self.lines) / len(self.lines))


class OcrEngine:
    """Thin wrapper over RapidOCR (PaddleOCR PP-OCRv4 ONNX)."""

    def __init__(self, use_angle_cls: bool = True, text_score: float = 0.5):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:  # pragma: no cover
            raise FileNotFoundError(
                "rapidocr-onnxruntime is not installed. "
                "Add it to requirements.txt and `pip install rapidocr-onnxruntime`."
            ) from e
        # RapidOCR downloads/bundles the PP-OCR ONNX models on first construction.
        self._engine = RapidOCR()
        self.use_angle_cls = use_angle_cls
        self.text_score = text_score

    def read(self, image: np.ndarray) -> OcrResult:
        """Run OCR on a BGR OpenCV image and return detected lines."""
        image = _downscale_for_ocr(image)
        raw, _elapse = self._engine(
            image, use_cls=self.use_angle_cls, text_score=self.text_score
        )
        result = OcrResult()
        if not raw:
            return result
        for item in raw:
            # RapidOCR yields [box(4x2), text, score]
            box, text, score = item[0], item[1], float(item[2])
            text = (text or "").strip()
            if not text:
                continue
            pts = tuple((float(p[0]), float(p[1])) for p in box)
            result.lines.append(OcrLine(text=text, confidence=score, box=pts))
        return result

    def warmup(self) -> None:
        try:
            self.read(np.zeros((64, 160, 3), dtype=np.uint8))
        except Exception:  # pragma: no cover
            pass
