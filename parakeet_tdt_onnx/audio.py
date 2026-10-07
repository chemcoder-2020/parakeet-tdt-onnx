"""Audio loading to 16 kHz mono float32 (via ffmpeg; soundfile as fallback)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file (any format ffmpeg knows) to mono float32 in [-1, 1]."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [ffmpeg, "-nostdin", "-threads", "0", "-i", str(path),
               "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
               "-ar", str(sample_rate), "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0 and proc.stdout:
            wav = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
            return wav

    # fallback: soundfile (only if sample rate already matches)
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr != sample_rate:
            raise RuntimeError(
                f"soundfile got {sr} Hz; need {sample_rate} Hz and ffmpeg is unavailable")
        return data.mean(axis=1)
    except ImportError:
        raise RuntimeError("ffmpeg not found on PATH and soundfile is not installed")
