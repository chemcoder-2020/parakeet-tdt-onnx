# parakeet-tdt-onnx

Run **NVIDIA parakeet-tdt-0.6b-v3** — a ~600M-parameter multilingual FastConformer-TDT speech
recognizer (25 languages) — on **plain ONNX Runtime**: no NeMo, no PyTorch, and no `onnx-asr`
at runtime. Three ONNX graphs from
[`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
plus ~250 lines of Python implementing the TDT greedy decode loop.

- **Verified byte-for-byte** against `onnx-asr`'s reference decode loop on all six test
  clips, for both int8 and fp32 (details in [Verified results](#verified-results) and
  [docs/parity-matrix.md](docs/parity-matrix.md)).
- **CPU-first**: int8 runs at **~28× realtime** on an Apple M1 (11 s clip → 0.39 s; fp32 ≈ 14×).
  Default ONNX Runtime threading is already optimal on this machine, and the CoreML execution
  provider measured *slower* than CPU (see [Speed](#speed-cpu-first)).
- **Small on disk**: the int8 model set is ~670 MB (vs ~2.6 GB fp32) and loads in ~0.9 s
  (fp32: ~2.8 s).

## Quick start

```bash
git clone https://github.com/chemcoder-2020/parakeet-tdt-onnx && cd parakeet-tdt-onnx
uv venv && uv pip install -e ".[hub]"        # or: python -m venv .venv && pip install -e ".[hub]"

python scripts/download_model.py             # int8 files (~670 MB) into ./models
parakeet-tdt-onnx testdata/bcn_weather.wav --quantization int8
```
```
Yesterday it was uh thirty five degrees in Barcelona, but today uh the temperature will go down to minus uh twenty degrees.

[audio 11.04s | wall 0.39s | ~28.0x realtime]
```

For full precision:

```bash
python scripts/download_model.py --quantization float32   # ~2.6 GB
parakeet-tdt-onnx testdata/bcn_weather.wav
```
```
Yesterday it was 35 degrees in Barcelona, but today the temperature will go down to minus 20 degrees.

[audio 11.04s | wall 0.80s | ~13.8x realtime]
```

Python API:

```python
from parakeet_tdt_onnx import Transcriber

tr = Transcriber("models", quantization="int8")
print(tr.transcribe("testdata/mary_had_lamb.wav")["text"])
```

Input can be any audio format ffmpeg reads (soundfile fallback for 16 kHz wavs). The CLI's
default quantization is `float32` (accuracy first); `scripts/verify.py` defaults to `int8`
(smaller download). Add `--timestamps` for per-token times, `--json` for the full result,
`--threads N` to override ONNX Runtime's thread count.

## Verified results

Every documented output below is reproducible; `scripts/verify.py` re-runs all six clips and
diffs the transcripts against recorded references, exiting non-zero on any mismatch:

```bash
python scripts/verify.py                      # int8, the default
python scripts/verify.py --quantization float32
```

Expected (Apple M1, onnxruntime 1.30; walls have ±10% run-to-run noise):

```
onnxruntime 1.30.0 | quantization: int8 | threads: ORT default | runs per file: 3
warm-up done

PASS  bcn_weather.wav                    0.435 s
PASS  en-Alice_woman.wav                 0.350 s
PASS  f2641_0_throatclearing.wav         0.170 s
PASS  fleur_es_sample.wav                0.305 s
PASS  librispeech_mr_quilter.wav         0.255 s
PASS  mary_had_lamb.wav                  0.616 s

identical transcripts: 6/6
```

(`float32` prints the same six PASS lines at roughly 1.6–2× the walls, e.g. bcn 0.704 s,
mary 1.036 s.)

**What exactly is byte-identical, and to what?** The references themselves are recorded by
`scripts/record_reference.py`: this port's mel graph computes the features, and then
*onnx-asr's independent decode loop* (its preprocessor replaced with identity) transcribes
them. Both sides therefore run the identical encoder/decoder graphs on identical feature
arrays, and the byte-compare isolates exactly what a reimplementation can get wrong — the
decode loop, tokenizer, and text assembly. There are no differences, for either quantization,
on any of the six clips. Full-pipeline comparisons against onnx-asr's built-in frontends, and
why *any* two mel frontends can flip borderline tokens on two of these clips, are in
[docs/parity-matrix.md](docs/parity-matrix.md).

The full-precision model's transcripts also match the
[`parakeet-redux-torch`](https://github.com/chemcoder-2020/parakeet-redux-torch) port (a
compressed ternary derivative of the same base model) on the well-behaved clips — e.g.
bcn_weather comes out character-for-character identical — and both models hear the same gist
on the hard clips.

## int8 vs float32: measured quality

Quantization is not free, so we diffed the transcripts rather than assuming. On the six
clips, int8 changed the text on 3 of 6 versus fp32 — two cosmetic, one material:

- **bcn_weather** — fp32 emits digits ("35 degrees … 20 degrees"), int8 spells them out and
  adds filler ("uh thirty five … uh twenty degrees").
- **fleur (Spanish)** — casing only: "en la Tierra" (fp32) vs "en la tierra" (int8).
- **mary_had_lamb** — material. fp32 transcribes the spoken preamble in full ("The uh first
  words I spoke in the original phonograph, a little piece of blacks and poetry…"), while the
  int8 encoder misses it and garbles the rest ("Mary had a little man speak for quite a
  floor…").

Rule of thumb: int8 for speed and disk, fp32 when accuracy matters. Both are verified
byte-identical to the reference runtime for their own quantization — the caveat above is
about int8's effect on the *model*, not on this port's fidelity.

## Speed (CPU first)

Apple M1, 16 GB RAM, onnxruntime 1.30.0, warmed min-of-3 per file, model load excluded
(int8 loads in ~0.9 s, fp32 ~2.8 s):

| clip | int8 | float32 |
| --- | --- | --- |
| bcn_weather.wav (11.0 s audio) | **0.409 s** | 0.815 s |
| en-Alice_woman.wav | 0.348 s | 0.716 s |
| f2641_0_throatclearing.wav | 0.158 s | 0.260 s |
| fleur_es_sample.wav | 0.304 s | 0.588 s |
| librispeech_mr_quilter.wav | 0.247 s | 0.469 s |
| mary_had_lamb.wav (16.5 s audio) | 0.598 s | 1.193 s |

Thread behavior (int8, bcn_weather / mary_had_lamb) — ONNX Runtime's MLAS kernels scale the
opposite way from naive per-op threading; the default already picks the sweet spot:

| threads | int8 | float32 |
| --- | --- | --- |
| 1 | 0.877 / 1.246 s | 2.203 / 3.143 s |
| 2 | 0.551 / 0.762 s | 1.182 / 1.698 s |
| 4 | **0.390 / 0.575 s** | **0.764 / 1.156 s** |
| 8 | 0.635 / 0.862 s | — |
| ORT default (= 4 here) | 0.409 / 0.598 s | 0.815 / 1.193 s |

Notes from the same sessions:

- **CoreML EP: tested and declined.** On this machine the CoreML execution provider (which
  onnx-asr silently enables by default) partitions the graph and measured *slower* than
  plain CPU (0.56 s vs ~0.39 s on bcn_weather). This port is CPU-only by default; pass
  `providers=["CoreMLExecutionProvider", ...]` to `Transcriber` if you want to try it.
- **Where it sits.** Same machine and clips, for reference: `parakeet-redux-torch`
  (compressed ternary, CPU, 1 thread) 0.63 / 0.86 s, and moondream's Photon-kernel runtime
  (CPU) 0.35 / 0.49 s on bcn/mary. This port's int8 (0.41 / 0.60 s) is in the same class
  with nothing but stock onnxruntime — and unlike the redux comparison, these are different
  checkpoints (full-precision base model vs compressed derivative), so treat the numbers as
  context, not a ranking.
- The whole stack is pure Python + onnxruntime + numpy, so it is OS/arch independent; all
  numbers here are from an Apple M1 and have not been re-measured on x86/CUDA.

## Reproduce the results

```bash
# 1. one-command verification (exit 0 = byte-identical to the recorded references)
python scripts/verify.py                                  # int8
python scripts/verify.py --quantization float32

# 2. benchmark: quantizations and thread counts (writes JSON with --json-out)
python scripts/benchmark.py --quantization int8 --threads "1,2,4,8"
python scripts/benchmark.py --quantization float32

# 3. re-record the references from scratch (dev-only oracle; requires onnx-asr)
uv pip install onnx-asr
python scripts/record_reference.py --quantization int8
python scripts/record_reference.py --quantization float32

# 4. the parity investigation behind docs/parity-matrix.md (also requires onnx-asr)
python scripts/parity_probe.py
```

## Serving it: OpenAI-compatible API (third-party, verified on this machine)

If you want an HTTP endpoint rather than the CLI/library — for Open WebUI, any OpenAI SDK
client, or a shared transcription box — the mature third-party server built on the **same
engine** is
[`groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai`](https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai):
FastAPI, OpenAI-compatible `POST /v1/audio/transcriptions`, int8 CPU default, Silero-VAD
chunking with a parallel inference pool for long files. We ran it CPU-only on the same M1
used for this repo's benchmarks and cross-checked it against this repo's references:

| workload (CPU, int8) | wall | effective |
| --- | --- | --- |
| bcn_weather (11.0 s), `PARAKEET_ORT_INTRA_THREADS=4` | **0.35–0.38 s** | ~30× realtime |
| bcn_weather (11.0 s), its auto-detected 8 threads | 0.53–0.72 s | ~19× |
| mary_had_lamb (16.0 s), 4 threads | 0.51–0.54 s | ~30× |
| 263 s file — 5 VAD chunks, parallel pool | **9.3 s** (4 threads) / 10.7 s (8 threads) | ~28× / ~25× |
| 4 × concurrent 11 s clips | 1.7 s wall | 25.6× aggregate |

Spot check: on bcn_weather and mary_had_lamb its int8 transcripts are **byte-identical** to
this repo's recorded references. Verdict: genuinely fast on CPU — and expectedly so, because
it is the same engine (istupakov int8 graphs + onnxruntime CPU); its added value is serving
ergonomics (HTTP API, VAD chunk fan-out for long audio, WAV fast path), not a different
inference path. On short single clips the two are a wash (~0.35–0.4 s on bcn_weather); long
files benefit from its chunk parallelism.

### Run it CPU-only

Its shipped `requirements.txt` pins `onnxruntime-gpu` + `tensorrt-cu12` and its default
profile is GPU/TensorRT — unsuitable for Macs and CPU-only boxes. This is the recipe we
verified (Python 3.11; installs in ~2 min):

```bash
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai && cd parakeet-tdt-0.6b-v3-fastapi-openai
python -m venv .venv && source .venv/bin/activate     # or: uv venv .venv
pip install "onnx-asr[hub]==0.12.0" onnxruntime "fastapi>=0.115" "uvicorn[standard]>=0.30" \
            "python-multipart>=0.0.9" "silero-vad>=6.0.0" "psutil>=5.0" "numpy<2"

# fastest CPU profile we measured (Apple Silicon — see notes for other CPUs):
PARAKEET_USE_GPU=false \
PARAKEET_DEFAULT_MODEL=parakeet-tdt-0.6b-v3 \
PARAKEET_ORT_INTRA_THREADS=4 \
python server.py                                     # serves on http://127.0.0.1:5092
```

Then, OpenAI style:

```bash
curl -s -F "file=@testdata/bcn_weather.wav" -F "model=parakeet-tdt-0.6b-v3" \
     -F "response_format=text" http://127.0.0.1:5092/v1/audio/transcriptions
```

Timing it exactly the way the table above was measured (warmed, curl wall — expect
~0.35–0.38 s for this clip with the launch command above):

```bash
for i in 1 2 3; do curl -s -o /dev/null -w "run$i: %{time_total}s\n" \
    -F "file=@testdata/bcn_weather.wav" -F "model=parakeet-tdt-0.6b-v3" \
    -F "response_format=text" http://127.0.0.1:5092/v1/audio/transcriptions; done
```

Tuning notes from our run:

- `onnx-asr[hub]` does **not** pull an ONNX Runtime wheel — install `onnxruntime`
  explicitly (their requirements do this via the GPU build; on CPU you want the plain one).
- The default is `PARAKEET_USE_GPU=true` (TensorRT); CPU boxes must set
  `PARAKEET_USE_GPU=false`.
- **Apple Silicon:** it sizes ORT threads to detected physical cores (8 on an M1), but 4 is
  ~35% faster there — that's why the launch block above sets
  `PARAKEET_ORT_INTRA_THREADS=4` (same 4-vs-8 finding as this repo's thread sweep: 0.390 s
  vs 0.635 s on bcn_weather). On other CPUs leave it unset — auto uses physical cores,
  which is the profile the project benchmarked on x86.
- Files longer than 75 s are split at VAD pause midpoints and inferred by its parallel pool:
  `PARAKEET_CHUNK_TARGET_SEC` (60 s default on CPU), `PARAKEET_INFER_WORKERS` (4). The first
  long request also loads the Silero VAD model (~4 s one-time).
- At the time of writing that repo contained no LICENSE file (its README shows an MIT
  badge) — confirm terms for your use case. It is an independent project; issues with it
  belong upstream.

## How it works

```
waveform (16 kHz mono)
  └─ nemo128.onnx            log-mel, 128 bins (the export's preprocessor graph)
      └─ encoder-model.onnx  FastConformer encoder, 8× subsampling
          └─ decoder_joint-model.onnx   run once per frame: token logits + duration logits
              └─ greedy TDT loop (parakeet_tdt_onnx/model.py)
                  └─ vocab.txt decode (metaspace → text)
```

The TDT loop mirrors NeMo's `GreedyTDTInfer` exactly as implemented by onnx-asr: at most 10
symbols per frame, the duration branch advances time even when the emitted token is blank,
and the predictor LSTM state is committed only on non-blank emissions. Timestamps come from
the frame index at each emission (80 ms per encoder frame).

## Layout

```
parakeet_tdt_onnx/    package: model.py (graphs + greedy loop), tokenizer.py, audio.py, cli.py
scripts/              download_model.py, verify.py, benchmark.py, record_reference.py, parity_probe.py
testdata/             six test clips (16 kHz wav) + recorded references + provenance
tests/                pytest end-to-end checks (skip unless ./models exists)
docs/parity-matrix.md evidence for the borderline-token analysis
```

## Requirements

- Python ≥ 3.10 (tested on 3.11), `onnxruntime` ≥ 1.17 (tested 1.30), `numpy`.
- ffmpeg on PATH for non-wav input (or the `soundfile` extra for 16 kHz wavs).
- `huggingface_hub` (the `[hub]` extra) only for `scripts/download_model.py`.

## Limitations

- Verification is scoped to the six public clips in `testdata/`; it is not a WER benchmark.
- Two of the six clips sit on greedy-decision boundaries, where *any* mel frontend change
  (including onnx-asr's own internal variants) can flip a token; see docs/parity-matrix.md.
- int8 degrades quality on hard audio (measured above); use fp32 when it matters.
- Model architecture documentation lives with the upstream projects (NVIDIA NeMo, the
  istupakov export); this repo is an independent, unaffiliated runner.

## License & attribution

Code: Apache-2.0. Model files are downloaded from
[`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
(CC-BY-4.0, NVIDIA parakeet-tdt-0.6b-v3) and are not redistributed here. See
[NOTICE](NOTICE) for full attribution.
