"""Loads and holds the ML models once (at app startup), not at import time."""
from __future__ import annotations

import numpy as np
import cv2

from app.core.config import settings
from app.ml.detector import Detector
from app.ml.recognizer import Recognizer
from app.ml.liveness import PassiveLiveness
from app.ml.challenge import ChallengeStore
from app.ml.ocr import OcrEngine


class Engine:
    def __init__(self):
        self.detector: Detector | None = None
        self.recognizer: Recognizer | None = None
        self.liveness: PassiveLiveness | None = None
        self.challenges: ChallengeStore | None = None
        self.ocr: OcrEngine | None = None

    def load(self) -> None:
        self.detector = Detector(settings.YUNET_MODEL, settings.DETECT_SCORE_THRESHOLD)
        self.recognizer = Recognizer(settings.SFACE_MODEL)
        # Passive liveness is optional at boot: if the Silent-Face ONNX models
        # are not present yet, we run without it (active challenge still applies).
        try:
            self.liveness = PassiveLiveness(
                [settings.ANTISPOOF_MODEL_1, settings.ANTISPOOF_MODEL_2],
                threshold=settings.LIVENESS_PASSIVE_THRESHOLD,
            )
        except FileNotFoundError as e:
            import logging
            logging.getLogger("face.engine").warning(
                "Passive liveness disabled (anti-spoof models missing): %s", e
            )
            self.liveness = None
        self.challenges = ChallengeStore(
            ttl_seconds=settings.CHALLENGE_TTL_SECONDS,
            turn_threshold=settings.LIVENESS_TURN_THRESHOLD,
            mirror=settings.LIVENESS_MIRROR,
        )
        # OCR is optional at boot: if rapidocr-onnxruntime is not installed the
        # face endpoints still run; /ocr/* return a clear 422 until it is added.
        if settings.OCR_ENABLED:
            try:
                self.ocr = OcrEngine(
                    use_angle_cls=settings.OCR_USE_ANGLE_CLS,
                    text_score=settings.OCR_TEXT_SCORE,
                )
            except FileNotFoundError as e:
                import logging
                logging.getLogger("face.engine").warning(
                    "OCR disabled (rapidocr-onnxruntime missing): %s", e
                )
                self.ocr = None

    def warmup(self) -> None:
        """Run the models once on a dummy frame so the first real request is fast."""
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        try:
            self.detector.detect(dummy)
        except Exception:
            pass
        if self.ocr is not None:
            self.ocr.warmup()


engine = Engine()


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes to a BGR OpenCV image."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes")
    return img
