"""
Calibration script to compute distances and recommend a threshold.
Usage: python tools/calibrate_threshold.py --data_dir <path_to_labeled_images>
Data layout: data_dir/<person_id>/*.jpg

The script will compute embeddings using your current FaceNet pipeline and report ROC/AUC and a suggested threshold.
"""
import argparse
import os
import itertools
import numpy as np
try:
    from sklearn.metrics import roc_curve, auc
except Exception:
    raise ImportError(
        "scikit-learn is required for calibration. Install it with: pip install scikit-learn"
    )

from app.utils.face_utils import read_image_bytes, extract_face_embedding


def load_embeddings(data_dir):
    mapping = {}
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    for person in os.listdir(data_dir):
        pdir = os.path.join(data_dir, person)
        if not os.path.isdir(pdir):
            continue
        mapping[person] = []
        for fname in os.listdir(pdir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(pdir, fname)
            with open(path, "rb") as f:
                b = f.read()
            try:
                emb = extract_face_embedding(read_image_bytes(b))
                mapping[person].append(emb)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    return mapping


def build_pairs(mapping):
    positives = []
    negatives = []
    persons = list(mapping.keys())
    for p in persons:
        embs = mapping[p]
        # all positive pairs within person
        for a, b in itertools.combinations(embs, 2):
            positives.append((a, b))
    # negative pairs: sample random across persons
    for a, b in itertools.combinations(persons, 2):
        for ea in mapping[a]:
            for eb in mapping[b]:
                negatives.append((ea, eb))
    return positives, negatives


def compute_distances(pairs):
    dists = []
    for a, b in pairs:
        a = np.asarray(a)
        b = np.asarray(b)
        sim = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))
        dists.append(1.0 - sim)
    return np.array(dists)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Labeled images folder (subfolders per identity)")
    args = parser.parse_args()

    mapping = load_embeddings(args.data_dir)
    positives, negatives = build_pairs(mapping)
    if len(positives) == 0 or len(negatives) == 0:
        print("Not enough data to compute calibration.")
        return

    pos_d = compute_distances(positives)
    neg_d = compute_distances(negatives)

    y = np.concatenate([np.ones_like(pos_d), np.zeros_like(neg_d)])
    scores = np.concatenate([ -pos_d, -neg_d ])  # higher score = more similar

    fpr, tpr, thresholds = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    # find EER (where FPR ~= 1-TPR)
    fnr = 1 - tpr
    eer_thresh = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    print(f"EER threshold (score space): {eer_thresh:.6f}, EER: {eer:.4f}")

    # Convert score threshold back to distance threshold: score = -distance
    dist_thresh = -eer_thresh
    print(f"Suggested distance threshold (approx EER): {dist_thresh:.6f}")

    # Optionally print some statistics
    print(f"Positive distances: mean={pos_d.mean():.4f} std={pos_d.std():.4f}")
    print(f"Negative distances: mean={neg_d.mean():.4f} std={neg_d.std():.4f}")


if __name__ == '__main__':
    main()
