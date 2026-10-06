#!/usr/bin/env python3
"""Phase-1 OCR proof, no DB / API needed.

Runs the OCR engine on each image and prints the detected text, the
auto-detected document type, and the parsed fields.

    python scripts/verify_ocr.py path/to/pan.jpg path/to/aadhaar.jpg
    python scripts/verify_ocr.py --type bank path/to/cheque.jpg

Run from the repo root so the `app` package resolves.
"""
import argparse
import json
import sys

import cv2

from app.ml.ocr import OcrEngine
from app.ml import doc_parsers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--type", dest="doc_type", default=None,
                    choices=["pan", "aadhaar", "bank"],
                    help="force a document type instead of auto-detecting")
    ap.add_argument("--no-mask", action="store_true",
                    help="return the full Aadhaar number (debug only)")
    args = ap.parse_args()

    print("Loading OCR engine (PaddleOCR PP-OCRv4 via ONNX Runtime)...")
    ocr = OcrEngine()

    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"\n[{path}] !! could not read image")
            continue
        result = ocr.read(img)
        lines = [l.text for l in result.lines]
        dtype = args.doc_type or doc_parsers.detect_doc_type(result.text)

        print(f"\n===== {path} =====")
        print(f"detected type : {dtype or 'UNKNOWN'}")
        print(f"mean conf     : {result.mean_confidence:.3f}")
        print("text:")
        for l in result.lines:
            print(f"  [{l.confidence:.2f}] {l.text}")
        if dtype:
            fields = doc_parsers.parse(dtype, lines, mask_aadhaar=not args.no_mask)
            print("fields:")
            print(json.dumps(fields, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
