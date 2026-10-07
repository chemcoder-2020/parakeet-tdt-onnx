#!/usr/bin/env python3
"""CPU benchmark: warmed min-of-N wall times, optionally across thread counts.

    python scripts/benchmark.py                        # int8, ORT default threads
    python scripts/benchmark.py --quantization both --threads 1,2,4,8
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from parakeet_tdt_onnx import Transcriber


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quantization", choices=["int8", "float32", "both"], default="int8")
    ap.add_argument("--threads", default="", help="comma-separated intra-op counts "
                    "(default: ORT default only)")
    ap.add_argument("--model-dir", default=str(REPO / "models"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    thread_counts = [int(x) for x in args.threads.split(",") if x.strip()] or [None]
    quants = ["int8", "float32"] if args.quantization == "both" else [args.quantization]
    wavs = sorted((REPO / "testdata").glob("*.wav"))

    print(f"machine: {platform.machine()} | {platform.platform()} | "
          f"onnxruntime {rt.__version__}")
    results = []
    for quant in quants:
        for threads in thread_counts:
            label = f"{quant} {threads or 'default'}t"
            try:
                t0 = time.perf_counter()
                transcriber = Transcriber(args.model_dir, quantization=quant, threads=threads)
                load_s = time.perf_counter() - t0
            except Exception as e:  # missing files etc.
                print(f"{label:24} SKIP ({e})")
                continue
            transcriber.transcribe(str(wavs[0]))  # warm
            for wav in wavs:
                ts = []
                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    transcriber.transcribe(str(wav))
                    ts.append(time.perf_counter() - t0)
                wall = min(ts)
                results.append({"quantization": quant, "threads": threads,
                                "file": wav.name, "wall": round(wall, 4),
                                "load_seconds": round(load_s, 2)})
                print(f"{label:24} {wav.name:<34} {wall:.3f} s", flush=True)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nsaved {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
