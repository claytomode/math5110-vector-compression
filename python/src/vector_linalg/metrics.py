"""Evaluate distance preservation and nearest-neighbor recall."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from vector_linalg.compression import CompressedVectors, compress_jl, cosine_scores, reconstruct_vectors, scores_from_compressed


@dataclass(frozen=True)
class MethodResult:
    method: str
    bits_per_dim: float
    mean_distance_rel_error: float
    recall_at_k: float
    compression_ratio: float


def _pairwise_distances(vectors: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    diffs = vectors[pairs[:, 0]] - vectors[pairs[:, 1]]
    return np.linalg.norm(diffs, axis=1)


def distance_distortion(
    original: np.ndarray,
    compressed: np.ndarray,
    pairs: np.ndarray,
) -> float:
    d_true = _pairwise_distances(original, pairs)
    d_comp = _pairwise_distances(compressed, pairs)
    rel = np.abs(d_comp - d_true) / np.maximum(d_true, 1e-12)
    return float(rel.mean())


def recall_at_k(
    keys: np.ndarray,
    queries: np.ndarray,
    query_indices: np.ndarray,
    *,
    k: int,
    score_fn,
) -> float:
    hits = 0
    for qi in query_indices:
        q = queries[qi]
        true_scores = cosine_scores(keys, q)
        true_top = set(np.argsort(-true_scores)[:k].tolist())

        approx_scores = score_fn(qi, q)
        approx_top = set(np.argsort(-approx_scores)[:k].tolist())
        hits += len(true_top & approx_top) / k
    return hits / len(query_indices)


def rag_recall_vs_full_at_k(
    chunk_keys: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    score_fn,
) -> float:
    """Overlap between compressed top-k and full-precision cosine top-k (full = truth)."""
    if len(queries) == 0:
        return 0.0
    overlap = 0.0
    for i, q in enumerate(queries):
        true_scores = cosine_scores(chunk_keys, q)
        true_top = set(np.argsort(-true_scores)[:k].tolist())
        approx_scores = score_fn(i, q)
        approx_top = set(np.argsort(-approx_scores)[:k].tolist())
        overlap += len(true_top & approx_top) / k
    return overlap / len(queries)


def full_precision_top_k(
    chunk_keys: np.ndarray,
    query: np.ndarray,
    *,
    k: int,
) -> list[int]:
    scores = cosine_scores(chunk_keys, query)
    return np.argsort(-scores)[:k].tolist()


def evaluate_rag_method(
    keys: np.ndarray,
    compressed: CompressedVectors,
    auto_queries: np.ndarray,
    *,
    pairs: np.ndarray,
    recall_k: int,
    full_bits: float = 32.0,
) -> MethodResult:
    recon = reconstruct_vectors(keys, compressed)

    def score_fn(_qi: int, q: np.ndarray) -> np.ndarray:
        return scores_from_compressed(keys, q, compressed)

    recall = (
        rag_recall_vs_full_at_k(keys, auto_queries, k=recall_k, score_fn=score_fn)
        if len(auto_queries) > 0
        else 0.0
    )

    return MethodResult(
        method=compressed.name,
        bits_per_dim=compressed.bits_per_dim,
        mean_distance_rel_error=distance_distortion(keys, recon, pairs),
        recall_at_k=recall,
        compression_ratio=full_bits / max(compressed.bits_per_dim, 1e-9),
    )


def evaluate_method(
    keys: np.ndarray,
    compressed: CompressedVectors,
    *,
    pairs: np.ndarray,
    query_indices: np.ndarray,
    recall_k: int,
    full_bits: float = 32.0,
) -> MethodResult:
    recon = reconstruct_vectors(keys, compressed)

    def score_fn(_qi: int, q: np.ndarray) -> np.ndarray:
        return scores_from_compressed(keys, q, compressed)

    return MethodResult(
        method=compressed.name,
        bits_per_dim=compressed.bits_per_dim,
        mean_distance_rel_error=distance_distortion(keys, recon, pairs),
        recall_at_k=recall_at_k(keys, keys, query_indices, k=recall_k, score_fn=score_fn),
        compression_ratio=full_bits / max(compressed.bits_per_dim, 1e-9),
    )


def build_jl_compressed(
    keys: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> CompressedVectors:
    comp, r = compress_jl(keys, k, rng)
    return CompressedVectors(
        name=comp.name,
        keys=comp.keys,
        bits_per_dim=comp.bits_per_dim,
        metadata={**comp.metadata, "r": r},
    )


def results_table(results: list[MethodResult], recall_k: int) -> pl.DataFrame:
    col = f"recall_at_{recall_k}"
    rows = [
        {
            "method": r.method,
            "bits_per_dim": r.bits_per_dim,
            "compression_ratio": r.compression_ratio,
            "mean_distance_rel_error": r.mean_distance_rel_error,
            col: r.recall_at_k,
        }
        for r in results
    ]
    return pl.DataFrame(rows).sort(col, descending=True)
