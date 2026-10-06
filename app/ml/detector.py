"""Face detection with OpenCV YuNet (MIT licensed, commercial-safe).

YuNet returns, per face, a 15-value row:
    [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y,
     rm_x, rm_y, lm_x, lm_y, score]
The raw row is what SFace's alignCrop expects, so we keep it around.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Face:
    row: np.ndarray            # raw 15-value YuNet row (float32), used by SFace.alignCrop
    box: tuple                 # (x, y, w, h)
    score: float
    right_eye: tuple
    left_eye: tuple
    nose: tuple
    right_mouth: tuple
    left_mouth: tuple


class Detector:
    def __init__(self, model_path: str, score_threshold: float = 0.8):
        # Input size is reset per-image in detect().
        self._det = cv2.FaceDetectorYN_create(
            model_path, "", (320, 320), score_threshold, 0.3, 5000
        )
        self.score_threshold = score_threshold

    def detect(self, image: np.ndarray) -> List[Face]:
        h, w = image.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(image)
        out: List[Face] = []
        if faces is None:
            return out
        for row in faces:
            row = row.astype(np.float32)
            score = float(row[14])
            if score < self.score_threshold:
                continue
            x, y, bw, bh = row[0:4]
            out.append(
                Face(
                    row=row,
                    box=(float(x), float(y), float(bw), float(bh)),
                    score=score,
                    right_eye=(float(row[4]), float(row[5])),
                    left_eye=(float(row[6]), float(row[7])),
                    nose=(float(row[8]), float(row[9])),
                    right_mouth=(float(row[10]), float(row[11])),
                    left_mouth=(float(row[12]), float(row[13])),
                )
            )
        # Largest face first (closest to camera).
        out.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        return out

    def detect_one(self, image: np.ndarray) -> Optional[Face]:
        faces = self.detect(image)
        return faces[0] if faces else None
