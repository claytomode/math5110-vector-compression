"""Vector compression: JL sketch, spectral truncation, sign and scalar quantization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompressedVectors:
    name: str
    keys: np.ndarray
    bits_per_dim: float
    metadata: dict


def jl_matrix(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian Johnson-Lindenstrauss map R in R^{k x d}, scaled by 1/sqrt(k)."""
    return rng.standard_normal((k, d)) / np.sqrt(k)


def compress_jl(keys: np.ndarray, k: int, rng: np.random.Generator) -> tuple[CompressedVectors, np.ndarray]:
    """Sketch keys with y = R x. Store R for query-side projection."""
    n, d = keys.shape
    r = jl_matrix(d, k, rng)
    sketched = keys @ r.T
    return (
        CompressedVectors(
            name=f"jl_{k}",
            keys=sketched.astype(np.float32),
            bits_per_dim=32.0 * k / d,
            metadata={"method": "johnson_lindenstrauss", "k": k},
        ),
        r,
    )


def query_jl(query: np.ndarray, r: np.ndarray) -> np.ndarray:
    return r @ query


def compress_rank_k(keys: np.ndarray, k: int) -> CompressedVectors:
    """Spectral truncation: keep top-k right singular vectors (PolarQuant-style direction)."""
    n, d = keys.shape
    k = min(k, min(n, d))
    _, _, vt = np.linalg.svd(keys, full_matrices=False)
    basis = vt[:k, :]
    coeffs = keys @ basis.T
    reconstructed = coeffs @ basis
    storage_bits = (k * d + n * k) * 32
    return CompressedVectors(
        name=f"rank_{k}",
        keys=reconstructed.astype(np.float32),
        bits_per_dim=storage_bits / (n * d),
        metadata={"method": "rank_k_svd", "k": k, "basis": basis},
    )


def compress_sign(keys: np.ndarray) -> CompressedVectors:
    """1-bit sign quantization per coordinate (QJL-style storage)."""
    n, d = keys.shape
    signs = np.sign(keys)
    signs[signs == 0.0] = 1.0
    norms = np.linalg.norm(keys, axis=1)
    storage_bits = n * d * 1 + n * 32
    return CompressedVectors(
        name="sign_1bit",
        keys=signs.astype(np.int8),
        bits_per_dim=storage_bits / (n * d),
        metadata={"method": "sign_quantization", "norms": norms},
    )


def query_sign_dot(query: np.ndarray, compressed: CompressedVectors) -> np.ndarray:
    """Unbiased-style dot product estimator using full-precision query + sign keys."""
    signs = compressed.keys.astype(np.float64)
    norms = compressed.metadata["norms"]
    d = query.shape[0]
    raw = signs @ query
    return norms * raw / np.sqrt(d)


def compress_scalar(keys: np.ndarray, bits: int) -> CompressedVectors:
    """Uniform scalar quantization per dimension."""
    lo = keys.min(axis=0)
    hi = keys.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    levels = 2**bits - 1
    norm = (keys - lo) / span
    codes = np.round(norm * levels).astype(np.uint8)
    recon = lo + (codes / levels) * span
    return CompressedVectors(
        name=f"scalar_{bits}bit",
        keys=recon.astype(np.float32),
        bits_per_dim=float(bits),
        metadata={"method": "scalar_uniform", "bits": bits, "lo": lo, "hi": hi},
    )


def cosine_scores(keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-12)
    k_norm = np.linalg.norm(keys, axis=1, keepdims=True)
    k_unit = keys / np.maximum(k_norm, 1e-12)
    return k_unit @ q


def scores_from_compressed(keys: np.ndarray, query: np.ndarray, compressed: CompressedVectors) -> np.ndarray:
    method = compressed.metadata["method"]
    if method == "johnson_lindenstrauss":
        r = compressed.metadata.get("r")
        if r is None:
            raise ValueError("JL compression missing projection matrix")
        q_sk = query_jl(query, r)
        return cosine_scores(compressed.keys, q_sk)
    if method == "sign_quantization":
        return query_sign_dot(query, compressed)
    return cosine_scores(compressed.keys, query)
