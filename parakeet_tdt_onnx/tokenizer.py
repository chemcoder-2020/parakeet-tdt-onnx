"""vocab.txt (metaspace tokenizer) loading and text decoding."""
from __future__ import annotations

import re
from pathlib import Path

# Space fixup kept identical to onnx-asr's DECODE_SPACE_PATTERN (MIT) so decoded
# text is byte-for-byte what the reference package produces from the same tokens.
_SPACE_FIXUP = re.compile(r"\A\s|\s\B|(\s)\b")


class Vocab:
    """id -> token mapping from the export's vocab.txt ("token id" per line)."""

    def __init__(self, tokens: dict[int, str]) -> None:
        self.tokens = tokens
        self.size = len(tokens)
        self.blank_id = next(i for i, t in tokens.items() if t == "<blk>")

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocab":
        tokens: dict[int, str] = {}
        with Path(path).open("rt", encoding="utf-8") as f:
            for line in f:
                token, _, id_str = line.rstrip("\n").rpartition(" ")
                tokens[int(id_str)] = token.replace("\u2581", " ")
        return cls(tokens)

    def decode(self, ids) -> str:
        """Join tokens and normalize spaces (leading / doubled / pre-punct)."""
        text = "".join(self.tokens[i] for i in ids)
        return _SPACE_FIXUP.sub(lambda m: " " if m.group(1) else "", text)
