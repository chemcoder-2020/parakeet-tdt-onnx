#!/usr/bin/env python3
"""Parity probe: feature drift + borderline-token matrix across mel frontends.

Dev-only tool (requires `pip install onnx-asr` for the numpy preprocessor and
its two bundled ONNX preprocessor graphs). Generates the numbers quoted in
docs/parity-matrix.md:

  1. feature drift of each frontend vs the numpy preprocessor (frames, the
     frontend's own features_lens, max/mean |diff|);
  2. full transcripts on the two decision-boundary clips (mary, alice) from
     the same int8 encoder + greedy loop for every frontend;
  3. a length check: decoding with features_lens +/- 1 to test whether the
     model-repo graph's off-by-one length could explain token flips.

    python scripts/parity_probe.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import onnx_asr
import onnxruntime as rt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy  # noqa: E402
from onnx_asr.utils import _select_channel, read_wav  # noqa: E402

from parakeet_tdt_onnx.tokenizer import Vocab  # noqa: E402

# onnx-asr ships its own preprocessor graphs inside the package data dir
PKG = Path(onnx_asr.__file__).parent / "preprocessors" / "data"
MODELS = REPO / "models"

CLIPS = ["mary_had_lamb", "en-Alice_woman", "bcn_weather"]
BOUNDARY_CLIPS = ["mary_had_lamb", "en-Alice_woman"]


def load(wav_path: Path) -> np.ndarray:
    return _select_channel(read_wav(str(wav_path))[0], None).astype(np.float32)


def feature_frontends(wav: np.ndarray, quant: str = "int8") -> dict[str, tuple[np.ndarray, int]]:
    out: dict[str, tuple[np.ndarray, int]] = {}
    np_pre = NemoPreprocessorNumpy("nemo128")
    feats, lens = np_pre(wav[None, :], np.array([len(wav)], np.int64))
    out["onnx-asr numpy preprocessor"] = (feats, int(lens[0]))
    for label, path in [
        ("onnx-asr bundled nemo128.onnx", PKG / "nemo128.onnx"),
        ("onnx-asr bundled nemo128_conv.onnx", PKG / "nemo128_conv.onnx"),
        ("model repo nemo128.onnx (this port)", MODELS / "nemo128.onnx"),
    ]:
        sess = rt.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feats, lens = sess.run(["features", "features_lens"],
                               {"waveforms": wav[None, :],
                                "waveforms_lens": np.array([len(wav)], np.int64)})
        out[label] = (feats, int(lens[0]))
    return out


def main() -> int:
    suffix = ".int8"
    vocab = Vocab.from_file(MODELS / "vocab.txt")
    blank, vsize = vocab.blank_id, vocab.size
    enc = rt.InferenceSession(str(MODELS / f"encoder-model{suffix}.onnx"),
                              providers=["CPUExecutionProvider"])
    dec = rt.InferenceSession(str(MODELS / f"decoder_joint-model{suffix}.onnx"),
                              providers=["CPUExecutionProvider"])

    def decode(feats: np.ndarray, flen: int) -> str:
        """This port's greedy TDT loop (kept inline so the probe is standalone)."""
        outputs, olens = enc.run(["outputs", "encoded_lengths"],
                                 {"audio_signal": feats, "length": np.asarray([flen], np.int64)})
        enc_out, enc_len = outputs.transpose(0, 2, 1)[0], int(olens[0])
        state = (np.zeros((2, 1, 640), np.float32), np.zeros((2, 1, 640), np.float32))
        tokens, t, emitted = [], 0, 0
        while t < enc_len:
            prev = tokens[-1] if tokens else blank
            logits, s1, s2 = dec.run(["outputs", "output_states_1", "output_states_2"], {
                "encoder_outputs": enc_out[t][None, :, None].astype(np.float32),
                "targets": [[prev]], "target_length": [1],
                "input_states_1": state[0], "input_states_2": state[1]})
            logits = logits.reshape(-1)
            token, duration = int(logits[:vsize].argmax()), int(logits[vsize:].argmax())
            if token != blank:
                state = (s1, s2)
                tokens.append(token)
                emitted += 1
            if duration > 0:
                t += duration
                emitted = 0
            elif token == blank or emitted == 10:
                t += 1
                emitted = 0
        return vocab.decode(tokens)

    print("== 1. feature drift vs the numpy preprocessor (int8 graph set) ==")
    for clip in CLIPS:
        wav = load(REPO / "testdata" / f"{clip}.wav")
        fronts = feature_frontends(wav)
        base = fronts["onnx-asr numpy preprocessor"][0]
        print(f"-- {clip}")
        for label, (feats, flen) in fronts.items():
            if label.startswith("onnx-asr numpy"):
                print(f"   {label:38} frames={feats.shape[2]} len={flen}")
            else:
                diff = np.abs(feats - base)
                print(f"   {label:38} frames={feats.shape[2]} len={flen} "
                      f"max|d|={diff.max():.3e} mean|d|={diff.mean():.3e}")

    print("\n== 2. borderline tokens by frontend (int8) ==")
    for clip in BOUNDARY_CLIPS:
        wav = load(REPO / "testdata" / f"{clip}.wav")
        print(f"-- {clip}")
        for label, (feats, flen) in feature_frontends(wav).items():
            text = decode(feats, flen)
            print(f"   [{hashlib.sha1(text.encode()).hexdigest()[:8]}] {label}")
            print(f"        {text}")

    print("\n== 3. does features_lens +/- 1 change anything? (mary, alice) ==")
    for clip in BOUNDARY_CLIPS:
        wav = load(REPO / "testdata" / f"{clip}.wav")
        fronts = feature_frontends(wav)
        for label in ("onnx-asr numpy preprocessor", "model repo nemo128.onnx (this port)"):
            feats, flen = fronts[label]
            a, b = decode(feats, flen), decode(feats, flen + 1 if "numpy" in label else flen - 1)
            tag = "len L+1" if "numpy" in label else "len L-1"
            print(f"   {clip:20} {label:38} len={flen} vs {tag}: "
                  f"{'IDENTICAL' if a == b else 'DIFFERENT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
