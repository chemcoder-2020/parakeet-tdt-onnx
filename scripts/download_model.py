#!/usr/bin/env python3
"""Download parakeet-tdt-0.6b-v3 ONNX model files from Hugging Face.

    python scripts/download_model.py --quantization int8     # ~670 MB (default)
    python scripts/download_model.py --quantization float32  # ~2.5 GB
    python scripts/download_model.py --quantization both
"""
from __future__ import annotations

import argparse

REPO_ID = "istupakov/parakeet-tdt-0.6b-v3-onnx"
COMMON = ["config.json", "vocab.txt", "nemo128.onnx", "README.md"]
INT8 = ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"]
FLOAT32 = ["encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quantization", choices=["int8", "float32", "both"], default="int8")
    ap.add_argument("--dir", default="models", help="destination directory (default: ./models)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download  # pip install 'parakeet-tdt-onnx[hub]'

    patterns = list(COMMON)
    if args.quantization in ("int8", "both"):
        patterns += INT8
    if args.quantization in ("float32", "both"):
        patterns += FLOAT32

    path = snapshot_download(REPO_ID, local_dir=args.dir, allow_patterns=patterns)
    print(f"downloaded {args.quantization} files to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
