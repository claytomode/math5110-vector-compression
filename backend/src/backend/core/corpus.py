"""Load and cache the RAG corpus index for API requests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from vector_linalg.config import ProjectConfig, load_config
from vector_linalg.rag import (
    RagBundle,
    fetch_rag_embeddings,
    list_compression_methods,
    storage_report,
)

COMPARE_METHODS = ("full_precision", "jl_128", "sign_1bit", "scalar_8bit")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def get_corpus_state() -> tuple[ProjectConfig, RagBundle, list[str], list[dict]]:
    cfg = load_config(_repo_root() / "python" / "config.yaml")
    bundle = fetch_rag_embeddings(cfg)
    methods = list_compression_methods(cfg, bundle.chunk_matrix.shape[1])
    storage = storage_report(bundle, cfg)
    return cfg, bundle, methods, storage


def compare_methods(available: list[str]) -> list[str]:
    return [m for m in COMPARE_METHODS if m in available]
