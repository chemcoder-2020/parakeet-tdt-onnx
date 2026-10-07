"""Command-line interface for parakeet-tdt-onnx."""
from __future__ import annotations

import argparse
import json

from .model import Transcriber


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="parakeet-tdt-onnx",
        description="NVIDIA parakeet-tdt-0.6b-v3 (multilingual) speech-to-text on ONNX Runtime.")
    ap.add_argument("audio", help="audio file (any format ffmpeg can read)")
    ap.add_argument("--model-dir", default="models",
                    help="model directory (default: ./models)")
    ap.add_argument("--quantization", choices=["float32", "int8"], default="float32",
                    help="model files to load (default: float32)")
    ap.add_argument("--threads", type=int, default=None,
                    help="intra-op threads for onnxruntime (default: onnxruntime default)")
    ap.add_argument("--timestamps", action="store_true",
                    help="include per-token timestamps (seconds) in the result")
    ap.add_argument("--json", action="store_true", help="print full result as JSON")
    args = ap.parse_args(argv)

    transcriber = Transcriber(args.model_dir, quantization=args.quantization,
                              threads=args.threads)
    result = transcriber.transcribe(args.audio, with_timestamps=args.timestamps)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["text"])
        rt = result["audio_seconds"] / result["wall"] if result["wall"] else 0.0
        print(f"\n[audio {result['audio_seconds']:.2f}s | wall {result['wall']:.2f}s"
              f" | ~{rt:.1f}x realtime]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
