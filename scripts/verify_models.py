"""Phase-1 proof: run detection + SFace + Silent-Face on local images.

No database needed. Confirms the models load, embeddings are 128-dim, and shows
same-person vs different-person cosine similarity plus liveness scores.

Usage:
    python scripts/verify_models.py staff_images/*.jpg staff_images/**/*.jpg
"""
import glob
import itertools
import sys

import cv2

# Ensure 'app' is importable when run from repo root.
sys.path.insert(0, ".")
from app.core.config import settings  # noqa: E402
from app.ml.detector import Detector  # noqa: E402
from app.ml.recognizer import Recognizer, cosine_similarity  # noqa: E402
from app.ml.liveness import PassiveLiveness  # noqa: E402


def main(paths):
    det = Detector(settings.YUNET_MODEL, settings.DETECT_SCORE_THRESHOLD)
    rec = Recognizer(settings.SFACE_MODEL)
    try:
        live = PassiveLiveness([settings.ANTISPOOF_MODEL_1, settings.ANTISPOOF_MODEL_2],
                               settings.LIVENESS_PASSIVE_THRESHOLD)
    except FileNotFoundError as e:
        print(f"[liveness skipped] {e}")
        live = None

    embeddings = {}
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  ! could not read {p}")
            continue
        face = det.detect_one(img)
        if face is None:
            print(f"  ! no face in {p}")
            continue
        emb = rec.embed(img, face)
        embeddings[p] = emb
        msg = f"  {p}: dim={emb.shape[0]} detScore={face.score:.2f}"
        if live is not None:
            r = live.score(img, face.box)
            msg += f" liveness={r.score:.2f} ({'REAL' if r.is_real else 'SPOOF'})"
        print(msg)

    print(f"\nEmbedding dimension: {next(iter(embeddings.values())).shape[0] if embeddings else 'n/a'}"
          f"  (expected {128})")
    print(f"Match threshold (cosine >= match): {settings.FACE_MATCH_COSINE}\n")
    print("Pairwise cosine similarity:")
    for a, b in itertools.combinations(embeddings, 2):
        sim = cosine_similarity(embeddings[a], embeddings[b])
        verdict = "SAME" if sim >= settings.FACE_MATCH_COSINE else "different"
        print(f"  {sim:+.3f}  {verdict:9}  {a.split('/')[-1]}  vs  {b.split('/')[-1]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    files = []
    for a in args:
        files.extend(glob.glob(a, recursive=True))
    if not files:
        files = glob.glob("staff_images/**/*.*", recursive=True)
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print("No images found. Pass image paths or put some in staff_images/.")
        sys.exit(1)
    main(files)
