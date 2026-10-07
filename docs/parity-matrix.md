# Parity matrix: mel frontends and borderline tokens

Exact outputs behind the README's *Parity notes*. Everything below was produced on
**2026-10-06**, Apple M1 (16 GB), macOS 26.6.2, Python 3.11.15, **onnxruntime 1.30.0**,
**onnx-asr 0.12.0** (used for the numpy preprocessor and its two bundled ONNX preprocessor
graphs). All decoding used the **int8** encoder/decoder with this port's greedy loop.
Reproduce with `python scripts/parity_probe.py`.

## Why this document exists

"Byte-identical to the reference runtime" is only meaningful once the *mel frontend* is
pinned, because it turns out several legitimately different frontends are in circulation:

1. **onnx-asr's numpy preprocessor** — a NumPy reimplementation of NeMo's log-mel, silently
   selected by onnx-asr whenever CPU is the first execution provider (its default on Macs).
2. **onnx-asr's bundled `nemo128.onnx`** — the ONNX mel graph shipped *inside the pip
   package*, used when the numpy path is disabled.
3. **onnx-asr's bundled `nemo128_conv.onnx`** — a convolutional variant, used when numpy is
   disabled and the first provider is not CUDA/TensorRT.
4. **the model repo's `nemo128.onnx`** — a different revision of the mel graph, shipped in
   `istupakov/parakeet-tdt-0.6b-v3-onnx` itself (139,764 bytes vs the package's 138,824) and
   what most raw-ONNX consumers, including this port, will use.

They are all "correct enough" to drive the model — but they are not the same numbers.

## 1. Feature drift (each frontend vs the numpy one)

`frames` = number of 128-bin mel frames returned; `len` = the frontend's *own* reported
valid length; `max|d|` / `mean|d|` = absolute difference of the feature tensors vs numpy.

| clip | frontend | frames | len | max│d│ | mean│d│ |
| --- | --- | --- | --- | --- | --- |
| mary_had_lamb | numpy | 1595 | 1594 | — | — |
| | bundled nemo128 | 1595 | 1594 | 8.636e-03 | 7.165e-05 |
| | bundled nemo128_conv | 1595 | 1594 | 2.630e-03 | 5.852e-05 |
| | **model repo (this port)** | 1595 | **1595** | 3.405 | 3.435e-03 |
| en-Alice_woman | numpy | 928 | 927 | — | — |
| | bundled nemo128 | 928 | 927 | 4.357e-04 | 1.398e-06 |
| | bundled nemo128_conv | 928 | 927 | 2.564e-04 | 1.105e-06 |
| | **model repo (this port)** | 928 | **928** | 9.195 | 2.745e-03 |
| bcn_weather | numpy | 1105 | 1104 | — | — |
| | bundled nemo128 | 1105 | 1104 | 7.939e-05 | 2.905e-06 |
| | bundled nemo128_conv | 1105 | 1104 | 1.048e-04 | 2.871e-06 |
| | **model repo (this port)** | 1105 | **1105** | 1.063 | 1.338e-03 |

Two observations:

- onnx-asr's numpy implementation and its bundled graph agree to ~1e-4…1e-6 (rounding-level),
  while the model repo's graph differs from both by ~1e-3 **mean** — a systematic difference
  (likely an export-revision difference in normalization), raising the question of which the
  encoder "expects". Both work: transcripts are fully coherent with either, and the six-clip
  spot checks are unaffected (see below).
- The model repo's graph reports *all* frames as valid (`len == frames`), while every
  onnx-asr frontend reports `frames - 1`.

## 2. Borderline tokens by frontend

The two hard clips; int8; identical encoder, decoder and greedy loop — only the mel frontend
differs. `[sha1-8]` prefixes the text hash.

**mary_had_lamb**

| frontend | transcript |
| --- | --- |
| numpy | `[773d24e4]` Mary had a little **man** speak for quite a floor, and everywhere that Mary went, the **man** would sure go. |
| bundled nemo128 | `[ca0b5116]` … and everywhere that Mary went, the **mail** would sure go. |
| bundled nemo128_conv | `[269ba591]` … speak for quite a **small** and everywhere that Mary went, the **ma'am** would sure go. |
| model repo (this port) | `[773d24e4]` Mary had a little **man** speak for quite a floor, and everywhere that Mary went, the **man** would sure go. |

**en-Alice_woman**

| frontend | transcript |
| --- | --- |
| numpy | `[23181be0]` So just to clarify, we've had **19 to 20** year olds on our podcast so far. And if you don't mind me asking, how old are you? |
| bundled nemo128 | `[23181be0]` … we've had **19 to 20** year olds … |
| bundled nemo128_conv | `[79f2da27]` … we've had **nineteen to twenty** year olds … |
| model repo (this port) | `[79f2da27]` … we've had **nineteen to twenty** year olds … |

The variants even disagree **with each other**: on mary, onnx-asr's own three frontends
produce three different tails (*man* / *mail* / *ma'am*); on alice its numpy and bundled
frontends say *19 to 20* while the conv one says *nineteen to twenty*. This port's model-repo
frontend lands with numpy on mary and with the conv variant on alice — i.e. there is no
"canonical" answer to align to at these two decision boundaries; they flip on any feature
perturbation at the ~1e-3 level.

## 3. The `features_lens` off-by-one is not the cause

Because the model repo's graph also reports one *more* valid frame than the onnx-asr family,
we tested the length field directly: decoding with the frontend's own length vs `len ± 1`
changes nothing on either clip (numpy features at L vs L+1: identical; model-repo features at
L vs L-1: identical). The flips track the **feature values**, not the length accounting.

## 4. What this port claims, precisely

- Against a **shared feature stream** — this port's mel graph as input to onnx-asr's
  independent decode loop — transcripts are **byte-identical on all six clips, int8 and
  fp32** (`scripts/verify.py`; refs recorded by `scripts/record_reference.py`).
- Against **onnx-asr's default-full-pipeline runs**, expect agreement on 4/6 clips on this
  machine; the two disagreements are exactly the boundary clips above, where onnx-asr's own
  frontends disagree among themselves at the same rate.
- This port pins the frontend shipped in the model repo (`nemo128.onnx`) — the file any
  consumer of the HF repo receives — rather than any pip-package-internal variant.
