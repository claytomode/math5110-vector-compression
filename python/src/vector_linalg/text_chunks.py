"""Text chunk container and helpers shared by corpus loaders."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source_file: str
    text: str


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_is_usable(text: str, *, min_words: int = 12, min_alnum_ratio: float = 0.38) -> bool:
    """Drop near-empty fragments, TOC dot-leaders, and non-prose regions."""
    text = _normalize_whitespace(text)
    if len(text) < 60:
        return False
    if text.count("....") >= 8:
        return False
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(words) < min_words:
        return False
    alnum = len(re.findall(r"[A-Za-z0-9]", text))
    return alnum / len(text) >= min_alnum_ratio


def chunk_text(text: str, *, chunk_chars: int, overlap: int) -> list[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks
