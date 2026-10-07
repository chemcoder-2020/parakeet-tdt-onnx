#!/usr/bin/env python3
"""Record the reference transcripts that scripts/verify.py checks against.

Provenance: this port's mel graph (models/nemo128.onnx, the file shipped in the
model repo) computes features; onnx-asr's *independent* decode loop then
transcribes them, with its preprocessor replaced by identity so both sides
decode the same feature stream. The recorded refs are therefore a byte-parity
target for the decode loop / tokenizer / assembly of this port.

Requires the dev-only oracle:  pip install onnx-asr
(never a runtime dependency of this package).

    python scripts/record_reference.py --quantization int8
    python scripts/record_reference.py --quantization float32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import onnx_asr  # noqa: E402
from onnx_asr.utils import _select_channel, read_wav  # noqa: E402

from parakeet_tdt_onnx import Transcriber  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quantization", choices=["int8", "float32"], default="int8")
    ap.add_argument("--model-dir", default=str(REPO / "models"))
    args = ap.parse_args()

    # onnx-asr selects files by quantization suffix; float32 == the unsuffixed default
    oracle_quant = None if args.quantization == "float32" else args.quantization
    model = onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3", args.model_dir, quantization=oracle_quant,
        providers=["CPUExecutionProvider"],
        preprocessor_config={"use_numpy_preprocessors": False, "use_conv_preprocessors": False},
    )
    asr = model.asr
    asr._preprocessor = lambda waveforms, waveforms_lens: (waveforms, waveforms_lens)

    transcriber = Transcriber(args.model_dir, quantization=args.quantization)
    for f in sorted((REPO / "testdata").glob("*.wav")):
        wav = _select_channel(read_wav(str(f))[0], None).astype(np.float32)
        features = transcriber._preprocess(wav)
        text = next(asr.recognize_batch(features, np.array([features.shape[2]], np.int64))).text
        ref = {
            "text": text,
            "oracle": (f"onnx-asr decode loop fed this port's nemo128.onnx features "
                       f"({args.quantization}); recorded by scripts/record_reference.py"),
        }
        (REPO / "testdata" / f"ref_{args.quantization}_{f.stem}.json").write_text(
            json.dumps(ref, indent=2, ensure_ascii=False))
        print(f"{f.name}: {text[:90]!r}", flush=True)
    print("\nrefs written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
