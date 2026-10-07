#!/usr/bin/env python3
"""Verify this port against the recorded reference transcripts on all test clips.

Runs every testdata/*.wav through this port with the chosen quantization and
diffs the transcript against testdata/ref_<quant>_*.json — refs recorded with
onnx-asr's independent decode loop fed this port's features (see
scripts/record_reference.py for the exact provenance). Prints warmed wall times;
exit code 0 = all identical, 1 = any mismatch.

    python scripts/verify.py                 # int8 (default; ~670 MB download)
    python scripts/verify.py --quantization float32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from parakeet_tdt_onnx import Transcriber


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quantization", choices=["int8", "float32"], default="int8")
    ap.add_argument("--model-dir", default=str(REPO / "models"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--runs", type=int, default=3, help="timed runs per file (default 3)")
    args = ap.parse_args()

    transcriber = Transcriber(args.model_dir, quantization=args.quantization,
                              threads=args.threads)
    wavs = sorted((REPO / "testdata").glob("*.wav"))
    if not wavs:
        print("no testdata/*.wav found")
        return 1

    import onnxruntime as rt
    print(f"onnxruntime {rt.__version__} | quantization: {args.quantization} | "
          f"threads: {args.threads or 'ORT default'} | runs per file: {args.runs}")
    transcriber.transcribe(str(wavs[0]))  # warm-up
    print("warm-up done\n")

    ok = 0
    for wav in wavs:
        ref = json.loads(
            (REPO / "testdata" / f"ref_{args.quantization}_{wav.stem}.json").read_text())
        ts = []
        result = None
        for _ in range(args.runs):
            t0 = time.perf_counter()
            result = transcriber.transcribe(str(wav))
            ts.append(time.perf_counter() - t0)
        same = result["text"] == ref["text"]
        ok += same
        print(f"{'PASS' if same else 'FAIL'}  {wav.name:<34} {min(ts):.3f} s")
        if not same:
            print(f"      ours: {result['text'][:120]!r}")
            print(f"      ref : {ref['text'][:120]!r}")

    print(f"\nidentical transcripts: {ok}/{len(wavs)}")
    return 0 if ok == len(wavs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
