"""parakeet-tdt-onnx — NVIDIA parakeet-tdt-0.6b-v3 speech-to-text via ONNX Runtime.

Self-contained: three ONNX graphs (mel preprocessor, encoder, decoder+joint)
from istupakov/parakeet-tdt-0.6b-v3-onnx plus a Python greedy TDT loop.
"""
from .audio import load_audio
from .model import Transcriber

__version__ = "0.1.0"
__all__ = ["Transcriber", "load_audio", "__version__"]
