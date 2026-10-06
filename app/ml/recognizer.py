"""Face recognition with OpenCV SFace (Apache-2.0, commercial-safe).

Produces a 128-dim embedding. We L2-normalize so cosine similarity is a
simple dot product and pgvector's cosine distance (<=>) is well-behaved.
"""
import cv2
import numpy as np

from app.ml.detector import Face


class Recognizer:
    def __init__(self, model_path: str):
        self._rec = cv2.FaceRecognizerSF_create(model_path, "")

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        """Align the detected face and return a normalized 128-d float32 vector."""
        aligned = self._rec.alignCrop(image, face.row)
        feat = self._rec.feature(aligned)          # shape (1, 128)
        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec)) + 1e-10
        return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)
