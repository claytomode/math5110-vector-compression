"""Extract and chunk text from PDF files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source_file: str
    text: str


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    try:
        pages = [_normalize_whitespace(page.get_text("text")) for page in doc]
    finally:
        doc.close()
    return "\n\n".join(p for p in pages if p)


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


def build_pdf_chunks(
    pdf_paths: list[Path],
    *,
    chunk_chars: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    out: list[TextChunk] = []
    for path in sorted(pdf_paths):
        stem = path.stem
        raw = extract_pdf_text(path)
        pieces = chunk_text(raw, chunk_chars=chunk_chars, overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            out.append(
                TextChunk(
                    chunk_id=f"{stem}_c{i:03d}",
                    source_file=path.name,
                    text=piece,
                )
            )
    return out
