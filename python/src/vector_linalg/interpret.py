"""Console summaries."""

from __future__ import annotations

import polars as pl

from vector_linalg.metrics import MethodResult


def _print_df(df: pl.DataFrame) -> None:
    with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=80):
        print(df)


def print_results(results: list[MethodResult], recall_k: int) -> None:
    col = f"recall_at_{recall_k}"
    rows = [
        {
            "method": r.method,
            "bits_per_dim": round(r.bits_per_dim, 2),
            "compression_ratio": round(r.compression_ratio, 2),
            "distance_rel_err": round(r.mean_distance_rel_error, 4),
            col: round(r.recall_at_k, 4),
        }
        for r in sorted(results, key=lambda x: -x.recall_at_k)
    ]
    _print_df(pl.DataFrame(rows))


def print_takeaways(results: list[MethodResult], recall_k: int, n_tokens: int, dim: int) -> None:
    best = max(results, key=lambda r: r.recall_at_k)
    print("\n=== Takeaways ===")
    print(f"  Universe: {n_tokens} tokens, d={dim} (GloVe word vectors as token embeddings).")
    print(f"  Task: preserve nearest-neighbor geometry under compression.")
    print(f"  Best recall@{recall_k}: {best.method} ({best.recall_at_k:.3f}) at ~{best.bits_per_dim:.1f} bits/dim.")
    print("  Survey link: JL + sign quantization + spectral truncation mirror TurboQuant-style pipelines.")
    print("  (Not LLM KV-cache inference — same linear-algebra tools on token vectors.)")
