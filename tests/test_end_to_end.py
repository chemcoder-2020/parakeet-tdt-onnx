"""End-to-end checks. Model-dependent tests skip unless ./models is downloaded."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

needs_models = pytest.mark.skipif(
    not (MODELS / "vocab.txt").exists() or not (MODELS / "nemo128.onnx").exists(),
    reason="models/ not downloaded (python scripts/download_model.py)")


@needs_models
@pytest.mark.parametrize("stem", ["bcn_weather", "f2641_0_throatclearing"])
def test_int8_matches_reference(stem):
    int8 = MODELS / "encoder-model.int8.onnx"
    if not int8.exists():
        pytest.skip("int8 model files not downloaded")
    from parakeet_tdt_onnx import Transcriber

    transcriber = Transcriber(MODELS, quantization="int8")
    result = transcriber.transcribe(REPO / "testdata" / f"{stem}.wav")
    ref = json.loads((REPO / "testdata" / f"ref_int8_{stem}.json").read_text())
    assert result["text"] == ref["text"]
