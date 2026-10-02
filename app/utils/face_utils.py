import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
from typing import Tuple, Optional

detector = MTCNN()
embedder = FaceNet()


def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode image bytes to an OpenCV BGR image."""
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes")
    return img


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image around its center by angle degrees."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated


def align_face(face: np.ndarray, keypoints: dict) -> np.ndarray:
    """Align face using eye keypoints returned by MTCNN.

    keypoints expected keys: 'left_eye', 'right_eye', etc.
    This rotates the face so the eye-line is horizontal.
    """
    if not keypoints or 'left_eye' not in keypoints or 'right_eye' not in keypoints:
        return face

    left = keypoints['left_eye']
    right = keypoints['right_eye']
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    if dx == 0:
        angle = 0.0
    else:
        angle = np.degrees(np.arctan2(dy, dx))
    # rotate by negative angle to align eyes horizontally
    aligned = _rotate_image(face, -angle)
    return aligned


def preprocess_face(face: np.ndarray, target_size: Tuple[int, int] = (160, 160), keypoints: Optional[dict] = None) -> np.ndarray:
    """Convert color, optionally align, resize and return RGB float32 image.

    Returns an image suitable for the embedding model.
    """
    if face.size == 0:
        raise ValueError("Empty face image")

    # Align first (uses coordinates relative to the face crop)
    if keypoints:
        face = align_face(face, keypoints)

    # Convert BGR (OpenCV) to RGB expected by many models
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    face_resized = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # Ensure float32 type (FaceNet wrapper will accept this)
    face_resized = face_resized.astype(np.float32)
    return face_resized


def _debug_log_embedding(tag: str, emb: np.ndarray) -> None:
    try:
        emb = np.asarray(emb)
        print(f"[DEBUG] {tag} shape={emb.shape} dtype={emb.dtype} norm={np.linalg.norm(emb):.6f}")
        print(f"[DEBUG] {tag} first10 = {emb.flatten()[:10].tolist()}")
    except Exception:
        pass


def extract_face_embedding(image: np.ndarray, target_size: Tuple[int, int] = (160, 160), debug: bool = False) -> np.ndarray:
    """Detect, preprocess and extract a normalized embedding for the first detected face.

    Returns a 1D float32 numpy array of length equal to the embedder output (512 by default).
    """
    results = detector.detect_faces(image)
    if not results:
        raise ValueError("No face detected")

    r = results[0]
    x, y, width, height = r['box']

    # Clamp coordinates to image bounds
    x = max(0, x)
    y = max(0, y)
    x2 = min(image.shape[1], x + width)
    y2 = min(image.shape[0], y + height)

    face = image[y:y2, x:x2]
    if face.size == 0:
        raise ValueError("Invalid face crop from image")

    keypoints = r.get('keypoints') or None
    face_pre = preprocess_face(face, target_size=target_size, keypoints=keypoints)

    # FaceNet wrapper expects a list of RGB images
    embedding = embedder.embeddings([face_pre])[0].astype(np.float32)

    # Normalize embedding vector (L2)
    norm = np.linalg.norm(embedding) + 1e-10
    embedding = embedding / norm

    if debug:
        _debug_log_embedding("embedding", embedding)

    return embedding

