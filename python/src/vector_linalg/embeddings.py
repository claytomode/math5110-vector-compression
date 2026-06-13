"""Load or cache token embedding vectors (GloVe)."""

from __future__ import annotations

import json
import zipfile
from io import TextIOWrapper
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from vector_linalg.config import ProjectConfig


def _parse_glove_line(line: str, dim: int) -> tuple[str, np.ndarray] | None:
    parts = line.strip().split()
    if len(parts) != dim + 1:
        return None
    word = parts[0]
    vec = np.asarray(parts[1:], dtype=np.float64)
    return word, vec


def _stream_glove_vectors(
    text_stream: TextIOWrapper,
    wanted: set[str],
    dim: int,
    max_lines: int,
) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    for i, line in enumerate(text_stream):
        if i >= max_lines or len(found) == len(wanted):
            break
        parsed = _parse_glove_line(line, dim)
        if parsed is None:
            continue
        word, vec = parsed
        if word in wanted and word not in found:
            found[word] = vec
    return found


def _download_glove_archive(cfg: ProjectConfig) -> Path:
    cache = cfg.data_dir / "glove.6B.zip"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache
    print(f"Downloading GloVe archive (~862 MB, one-time) to {cache} ...")
    with httpx.stream("GET", cfg.glove_url, follow_redirects=True, timeout=900.0) as resp:
        resp.raise_for_status()
        with cache.open("wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    return cache


def _glove_inner_name(cfg: ProjectConfig, zf: zipfile.ZipFile) -> str:
    preferred = f"glove.6B.{cfg.glove_dim}d.txt"
    if preferred in zf.namelist():
        return preferred
    matches = [n for n in zf.namelist() if n.endswith(f"{cfg.glove_dim}d.txt")]
    if not matches:
        raise RuntimeError(f"No {cfg.glove_dim}d GloVe file inside archive")
    return matches[0]


def fetch_token_embeddings(cfg: ProjectConfig, *, refresh: bool = False) -> pl.DataFrame:
    if cfg.embeddings_cache.exists() and not refresh:
        return pl.read_parquet(cfg.embeddings_cache)

    wanted = set(cfg.tokens)
    zip_path = _download_glove_archive(cfg)
    found: dict[str, np.ndarray] = {}

    with zipfile.ZipFile(zip_path) as zf:
        inner = _glove_inner_name(cfg, zf)
        with zf.open(inner) as raw:
            text = TextIOWrapper(raw, encoding="utf-8", errors="replace")
            found = _stream_glove_vectors(text, wanted, cfg.glove_dim, cfg.max_glove_scan_lines)

    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} tokens in GloVe scan: {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
        )

    rows = [{"token": t, **{f"d{i}": float(found[t][i]) for i in range(cfg.glove_dim)}} for t in cfg.tokens]
    df = pl.DataFrame(rows)
    cfg.embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cfg.embeddings_cache)

    meta = {
        "source": "GloVe 6B",
        "citation": "Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation.",
        "url": cfg.glove_url,
        "dim": cfg.glove_dim,
        "n_tokens": len(cfg.tokens),
        "file": str(cfg.embeddings_cache),
    }
    cfg.metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


def embedding_matrix(df: pl.DataFrame, dim: int) -> tuple[list[str], np.ndarray]:
    tokens = df["token"].to_list()
    cols = [f"d{i}" for i in range(dim)]
    matrix = df.select(cols).to_numpy()
    return tokens, matrix
