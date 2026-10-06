# Model files

These are **not** committed (see `.gitignore`). Download them into this folder.

## 1. Detection + Recognition (OpenCV Zoo) — commercial-safe

```bash
cd models

# YuNet face detector (MIT)
curl -L -o face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# SFace recognizer (Apache-2.0) -> 128-dim embeddings
curl -L -o face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

## 2. Passive liveness — Silent-Face / MiniFASNet (Apache-2.0)

The original repo ships PyTorch weights; we run them via ONNX. Export once:

```bash
# Clone the reference repo (Apache-2.0)
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git /tmp/silent-face

# Weights live in /tmp/silent-face/resources/anti_spoof_models/:
#   2.7_80x80_MiniFASNetV2.pth
#   4_0_0_80x80_MiniFASNetV1SE.pth

# Export to ONNX (needs torch in a scratch venv):
python scripts/export_silent_face_onnx.py \
  --repo /tmp/silent-face \
  --out  models
```

That produces:
- `models/2.7_80x80_MiniFASNetV2.onnx`
- `models/4_0_0_80x80_MiniFASNetV1SE.onnx`

The filenames encode the crop scale (`2.7`, `4`) and input size (`80x80`); the
service parses them, so **keep the names unchanged**.

## Licensing note
YuNet (MIT), SFace (Apache-2.0) and Silent-Face (Apache-2.0) are all usable in a
commercial product. Keep this README with the models for your license audit trail.
