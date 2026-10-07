"""TDT greedy decoding over the three ONNX graphs.

Pipeline: waveform -> nemo128 (log-mel) -> encoder -> per-frame decoder+joint
(TDT greedy loop, max 10 symbols per frame, duration skips 0..4).

The loop mirrors NeMo's GreedyTDTInfer semantics exactly as implemented by
onnx-asr (the verification oracle): the LSTM predictor state is committed only
on non-blank emissions, and the duration branch advances time even when the
token emitted in that same call is blank.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as rt

from .audio import SAMPLE_RATE, load_audio
from .tokenizer import Vocab

# "float32" -> encoder-model.onnx / decoder_joint-model.onnx
# "int8"    -> encoder-model.int8.onnx / decoder_joint-model.int8.onnx
_SUFFIX = {"float32": "", "int8": ".int8"}
_FRAME_SECONDS = 0.08  # 8x subsampling of a 10 ms mel hop


def _np_of(type_str: str) -> type:
    return {"tensor(float)": np.float32, "tensor(int64)": np.int64,
            "tensor(int32)": np.int32}[type_str]


class Transcriber:
    """Load the ONNX graphs and transcribe 16 kHz audio.

    Args:
        model_dir: directory with nemo128.onnx, encoder-model[.int8].onnx,
            decoder_joint-model[.int8].onnx, vocab.txt, config.json.
        quantization: "float32" (default) or "int8" (smaller, faster; see README
            for measured quality notes).
        threads: intra-op threads for onnxruntime (None = ORT default).
        providers: execution providers list (default: CPU only).
        max_symbols: TDT max symbols per frame step (NeMo default 10).
    """

    def __init__(
        self,
        model_dir: str | Path,
        quantization: str = "float32",
        threads: int | None = None,
        providers: Sequence[str] | None = None,
        max_symbols: int = 10,
    ) -> None:
        self.model_dir = Path(model_dir)
        if quantization not in _SUFFIX:
            raise ValueError(f"quantization must be one of {sorted(_SUFFIX)}")
        suffix = _SUFFIX[quantization]

        opts = rt.SessionOptions()
        if threads is not None:
            opts.intra_op_num_threads = threads
        kw: dict = {"sess_options": opts}
        if providers is not None:
            kw["providers"] = list(providers)

        self.pre = rt.InferenceSession(str(self.model_dir / "nemo128.onnx"), **kw)
        self.enc = rt.InferenceSession(
            str(self.model_dir / f"encoder-model{suffix}.onnx"), **kw)
        self.dec = rt.InferenceSession(
            str(self.model_dir / f"decoder_joint-model{suffix}.onnx"), **kw)
        self.vocab = Vocab.from_file(self.model_dir / "vocab.txt")
        self.max_symbols = max_symbols

        dec_in = {i.name: i for i in self.dec.get_inputs()}
        self._state_shape = (dec_in["input_states_1"].shape[0], 1,
                             dec_in["input_states_1"].shape[2])
        self._enc_len_dtype = _np_of({i.name: i.type for i in self.enc.get_inputs()}["length"])
        self._dec_in_dtype = {
            n: _np_of(dec_in[n].type) for n in ("targets", "target_length")}

    # -- graph wrappers ---------------------------------------------------

    def _preprocess(self, waveform: np.ndarray) -> np.ndarray:
        features, _ = self.pre.run(["features", "features_lens"], {
            "waveforms": waveform[None, :].astype(np.float32),
            "waveforms_lens": np.array([waveform.shape[0]], np.int64)})
        return features

    def _encode(self, features: np.ndarray) -> tuple[np.ndarray, int]:
        outputs, lengths = self.enc.run(["outputs", "encoded_lengths"], {
            "audio_signal": features,
            "length": np.array([features.shape[2]], self._enc_len_dtype)})
        return outputs.transpose(0, 2, 1)[0], int(lengths[0])  # [T', 1024]

    def _decoder_step(self, token: int, states: tuple[np.ndarray, np.ndarray],
                      frame: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        outputs, s1, s2 = self.dec.run(["outputs", "output_states_1", "output_states_2"], {
            "encoder_outputs": frame[None, :, None].astype(np.float32),
            "targets": np.array([[token]], self._dec_in_dtype["targets"]),
            "target_length": np.array([1], self._dec_in_dtype["target_length"]),
            "input_states_1": states[0], "input_states_2": states[1]})
        return outputs.reshape(-1), (s1, s2)

    # -- TDT greedy loop --------------------------------------------------

    def _greedy_decode(self, encoded: np.ndarray, encoded_len: int) -> tuple[list[int], list[int]]:
        assert encoded_len <= encoded.shape[0]
        states = (np.zeros(self._state_shape, np.float32),
                  np.zeros(self._state_shape, np.float32))
        tokens: list[int] = []
        timestamps: list[int] = []
        t = 0
        emitted = 0
        while t < encoded_len:
            prev = tokens[-1] if tokens else self.vocab.blank_id
            logits, next_states = self._decoder_step(prev, states, encoded[t])
            token = int(logits[: self.vocab.size].argmax())
            duration = int(logits[self.vocab.size :].argmax())

            if token != self.vocab.blank_id:
                states = next_states
                tokens.append(token)
                timestamps.append(t)
                emitted += 1

            if duration > 0:
                t += duration
                emitted = 0
            elif token == self.vocab.blank_id or emitted == self.max_symbols:
                t += 1
                emitted = 0
        return tokens, timestamps

    # -- public API -------------------------------------------------------

    def transcribe(self, path: str | Path, with_timestamps: bool = False) -> dict:
        """Transcribe an audio file. Returns a dict with text / tokens / wall."""
        waveform = load_audio(path)
        t0 = time.perf_counter()
        tokens, timestamps = self._greedy_decode(*self._encode(self._preprocess(waveform)))
        wall = time.perf_counter() - t0
        result = {
            "text": self.vocab.decode(tokens),
            "tokens": tokens,
            "wall": wall,
            "audio_seconds": waveform.shape[0] / SAMPLE_RATE,
        }
        if with_timestamps:
            result["timestamps"] = [round(t * _FRAME_SECONDS, 2) for t in timestamps]
        return result
