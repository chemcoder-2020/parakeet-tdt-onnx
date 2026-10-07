# testdata

Six public test clips used for verification:

| file | source | notes |
| --- | --- | --- |
| `bcn_weather.wav` | dummy-audio-samples | weather sentence; used in official model cards |
| `librispeech_mr_quilter.wav` | LibriSpeech dev-clean (via dummy-audio-samples) | "mister Quilter is the apostle…" |
| `fleur_es_sample.wav` | FLEURS Spanish (via dummy-audio-samples) | multilingual check |
| `en-Alice_woman.wav` | dummy-audio-samples | conversational English |
| `mary_had_lamb.wav` | dummy-audio-samples | hard fixture (borderline greedy decisions) |
| `f2641_0_throatclearing.wav` | dummy-audio-samples | non-speech edge case |

All files come from [`hf-internal-testing/dummy-audio-samples`](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples)
and were converted with ffmpeg to 16 kHz mono PCM16 wav — the exact bytes fed
to the model on every run here.

`ref_int8_*.json` / `ref_float32_*.json` are the byte-parity targets checked by
`scripts/verify.py`. Provenance: onnx-asr's independent decode loop fed *this
port's* mel features (so the comparison isolates the decode loop, tokenizer and
assembly); reproduce with `scripts/record_reference.py`. See the README's
"Parity notes" for how the mel graphs vary across onnx-asr versions/variants.
