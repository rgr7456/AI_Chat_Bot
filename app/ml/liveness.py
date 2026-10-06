"""Passive liveness / anti-spoofing with Silent-Face (MiniFASNet) ONNX models.

Silent-Face (MiniVision, Apache-2.0) ships two small models evaluated at
different crop scales; their softmax outputs are averaged. Class index 1 is
"real". Preprocessing mirrors the reference repo: crop the face region expanded
by the model's scale, resize to the model's input size, scale to [0,1], CHW.

See models/README.md for how to obtain / export the .onnx files.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


def _parse_model_name(path: str) -> Tuple[int, int, Optional[float]]:
    """From '2.7_80x80_MiniFASNetV2.onnx' -> (h=80, w=80, scale=2.7)."""
    name = os.path.basename(path)
    info = name.split(".onnx")[0].split("_")[:-1]  # drop the 'MiniFASNet..' part
    h_str, w_str = info[-1].split("x")
    scale = None if info[0] == "org" else float(info[0])
    return int(h_str), int(w_str), scale


def _crop(img: np.ndarray, bbox: Tuple[int, int, int, int], scale: Optional[float],
          out_w: int, out_h: int) -> np.ndarray:
    if scale is None:
        return cv2.resize(img, (out_w, out_h))
    src_h, src_w = img.shape[:2]
    x, y, bw, bh = bbox
    scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
    new_w, new_h = bw * scale, bh * scale
    cx, cy = x + bw / 2.0, y + bh / 2.0
    lx, ly = cx - new_w / 2.0, cy - new_h / 2.0
    rx, ry = cx + new_w / 2.0, cy + new_h / 2.0
    if lx < 0:
        rx -= lx; lx = 0
    if ly < 0:
        ry -= ly; ly = 0
    if rx > src_w - 1:
        lx -= (rx - src_w + 1); rx = src_w - 1
    if ry > src_h - 1:
        ly -= (ry - src_h + 1); ry = src_h - 1
    crop = img[int(ly):int(ry) + 1, int(lx):int(rx) + 1]
    return cv2.resize(crop, (out_w, out_h))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-10)


@dataclass
class LivenessResult:
    is_real: bool
    score: float  # probability of "real" in [0,1]


class PassiveLiveness:
    def __init__(self, model_paths: List[str], threshold: float = 0.70):
        self.threshold = threshold
        self._models = []
        for p in model_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Anti-spoofing model not found: {p}")
            sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            h, w, scale = _parse_model_name(p)
            self._models.append((sess, sess.get_inputs()[0].name, h, w, scale))

    def score(self, image: np.ndarray, bbox: Tuple[float, float, float, float]) -> LivenessResult:
        ibbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        accum = np.zeros(3, dtype=np.float32)
        for sess, in_name, h, w, scale in self._models:
            patch = _crop(image, ibbox, scale, w, h)
            blob = patch.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # 1xCxHxW
            logits = sess.run(None, {in_name: blob})[0].reshape(-1)
            accum += _softmax(logits)
        accum /= max(len(self._models), 1)
        label = int(np.argmax(accum))
        real_score = float(accum[1])
        return LivenessResult(is_real=(label == 1 and real_score >= self.threshold),
                              score=real_score)
