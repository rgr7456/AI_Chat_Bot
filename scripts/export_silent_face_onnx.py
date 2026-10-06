"""Export Silent-Face (MiniFASNet) PyTorch weights to ONNX.

Run in a scratch venv that has torch installed:
    pip install torch
    python scripts/export_silent_face_onnx.py --repo /tmp/silent-face --out models

It reuses the model definitions from the cloned Silent-Face repo, so point
--repo at that clone.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to cloned Silent-Face-Anti-Spoofing repo")
    ap.add_argument("--out", default="models", help="output dir for .onnx files")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import torch
    from src.anti_spoof_predict import AntiSpoofPredict  # noqa
    from src.model_lib.MiniFASNet import (  # noqa
        MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE,
    )
    from src.utility import parse_model_name  # noqa

    kernel_size = (80 // 16, 80 // 16)  # conv6_kernel for 80x80 inputs
    builders = {
        "MiniFASNetV1": MiniFASNetV1,
        "MiniFASNetV2": MiniFASNetV2,
        "MiniFASNetV1SE": MiniFASNetV1SE,
        "MiniFASNetV2SE": MiniFASNetV2SE,
    }

    weights_dir = os.path.join(args.repo, "resources", "anti_spoof_models")
    os.makedirs(args.out, exist_ok=True)

    for fname in os.listdir(weights_dir):
        if not fname.endswith(".pth"):
            continue
        h, w, model_type, scale = parse_model_name(fname)
        model = builders[model_type](conv6_kernel=kernel_size).eval()
        state = torch.load(os.path.join(weights_dir, fname), map_location="cpu")
        # Strip DataParallel 'module.' prefixes if present.
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        model.load_state_dict(state)

        onnx_path = os.path.join(args.out, fname.replace(".pth", ".onnx"))
        dummy = torch.randn(1, 3, h, w)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=11,
        )
        print(f"exported {onnx_path}  (scale={scale}, {h}x{w})")


if __name__ == "__main__":
    main()
