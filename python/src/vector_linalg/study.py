"""Run all compression methods and evaluate on token embeddings."""

from __future__ import annotations

import numpy as np

from vector_linalg.compression import (
    CompressedVectors,
    compress_rank_k,
    compress_scalar,
    compress_sign,
)
from vector_linalg.config import ProjectConfig
from vector_linalg.metrics import MethodResult, build_jl_compressed, evaluate_method


def run_compression_study(
    keys: np.ndarray,
    cfg: ProjectConfig,
) -> list[MethodResult]:
    rng = np.random.default_rng(cfg.random_seed)
    n = keys.shape[0]
    pairs = rng.integers(0, n, size=(cfg.n_distance_pairs, 2))
    query_indices = rng.choice(n, size=min(cfg.n_query_tokens, n), replace=False)

    compressed_list: list[CompressedVectors] = []

    baseline = CompressedVectors(
        name="full_precision",
        keys=keys.astype(np.float32),
        bits_per_dim=32.0,
        metadata={"method": "none"},
    )
    compressed_list.append(baseline)

    for k in cfg.compression.jl_dims:
        if k < keys.shape[1]:
            compressed_list.append(build_jl_compressed(keys, k, rng))

    for k in cfg.compression.rank_k:
        compressed_list.append(compress_rank_k(keys, k))

    compressed_list.append(compress_sign(keys))

    for bits in cfg.compression.scalar_bits:
        compressed_list.append(compress_scalar(keys, bits))

    results: list[MethodResult] = []
    for comp in compressed_list:
        results.append(
            evaluate_method(
                keys,
                comp,
                pairs=pairs,
                query_indices=query_indices,
                recall_k=cfg.recall_k,
            )
        )
    return results
