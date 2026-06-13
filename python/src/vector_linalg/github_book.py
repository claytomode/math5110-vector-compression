"""Fetch MATH 5110 Quarto book chapters from Professor He Wang's public GitHub."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from vector_linalg.pdf_corpus import TextChunk, chunk_is_usable, chunk_text


@dataclass(frozen=True)
class GithubBookConfig:
    owner: str
    repo: str
    branch: str
    chapters_dir: str


def _strip_qmd(raw: str) -> str:
    """Quarto .qmd -> plain text suitable for embedding."""
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            raw = raw[end + 3 :]
    raw = re.sub(r"^:::.*$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\{[^`]*\}.*?```", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
    raw = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", raw)
    raw = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
    raw = raw.replace("$", " ")
    return re.sub(r"\s+", " ", raw).strip()


def _list_chapter_files(client: httpx.Client, cfg: GithubBookConfig) -> list[str]:
    url = (
        f"https://api.github.com/repos/{cfg.owner}/{cfg.repo}"
        f"/contents/{cfg.chapters_dir}?ref={cfg.branch}"
    )
    resp = client.get(url, headers={"Accept": "application/vnd.github+json"})
    resp.raise_for_status()
    names = [
        item["name"]
        for item in resp.json()
        if item.get("type") == "file" and str(item["name"]).endswith(".qmd")
    ]
    return sorted(names)


def _fetch_qmd(client: httpx.Client, cfg: GithubBookConfig, filename: str) -> str:
    url = (
        f"https://raw.githubusercontent.com/{cfg.owner}/{cfg.repo}"
        f"/{cfg.branch}/{cfg.chapters_dir}/{filename}"
    )
    resp = client.get(url)
    resp.raise_for_status()
    return resp.text


def _chunk_chapter(
    filename: str,
    text: str,
    *,
    chunk_chars: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    stem = Path(filename).stem
    sections = re.split(r"(?=## )", text)
    if not sections or (len(sections) == 1 and not sections[0].startswith("##")):
        sections = [text]

    out: list[TextChunk] = []
    idx = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue
        title_match = re.match(r"^##\s+(.+?)(?:\s|$)", section)
        section_title = title_match.group(1) if title_match else "section"
        for piece in chunk_text(section, chunk_chars=chunk_chars, overlap=chunk_overlap):
            if not chunk_is_usable(piece):
                continue
            out.append(
                TextChunk(
                    chunk_id=f"{stem}_c{idx:03d}",
                    source_file=f"{stem} :: {section_title}",
                    text=piece,
                )
            )
            idx += 1
    return out


def build_github_book_chunks(
    cfg: GithubBookConfig,
    *,
    chunk_chars: int,
    chunk_overlap: int,
    cache_dir: Path,
    refresh: bool = False,
) -> list[TextChunk]:
    """
    Pull chapter .qmd sources from GitHub (no PDF/OCR).

    Book: https://github.com/wanghemath/Book-AdvancedLinearAlgebraAI
    Site may 404; source repo is the canonical text.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    if not refresh and manifest_path.exists():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [
            TextChunk(chunk_id=c["chunk_id"], source_file=c["source_file"], text=c["text"])
            for c in raw
        ]

    print(f"Fetching Quarto chapters from {cfg.owner}/{cfg.repo}...")
    chunks: list[TextChunk] = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        files = _list_chapter_files(client, cfg)
        for name in files:
            try:
                qmd = _fetch_qmd(client, cfg, name)
            except httpx.HTTPError as exc:
                print(f"  skip {name}: {exc}")
                continue
            text = _strip_qmd(qmd)
            if not text:
                continue
            pieces = _chunk_chapter(
                name,
                text,
                chunk_chars=chunk_chars,
                chunk_overlap=chunk_overlap,
            )
            chunks.extend(pieces)
            print(f"  {name}: {len(pieces)} chunks")

    if not chunks:
        raise RuntimeError("No chunks from GitHub book source.")

    manifest_path.write_text(
        json.dumps(
            [{"chunk_id": c.chunk_id, "source_file": c.source_file, "text": c.text} for c in chunks],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  -> {len(chunks)} chunks from {len(files)} chapters")
    return chunks
