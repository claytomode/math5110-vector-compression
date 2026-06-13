#!/usr/bin/env python3
"""Regenerate token embedding data and compression figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from vector_linalg.config import load_config
from vector_linalg.embeddings import embedding_matrix, fetch_token_embeddings
from vector_linalg.interpret import print_results, print_takeaways
from vector_linalg.plots import generate_all_figures
from vector_linalg.study import run_compression_study


def main() -> None:
    cfg = load_config()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading token embeddings (GloVe)...")
    df = fetch_token_embeddings(cfg)
    tokens, keys = embedding_matrix(df, cfg.glove_dim)
    print(f"  {len(tokens)} tokens x d={cfg.glove_dim}")

    print("\nCompression study (JL, rank-k, sign, scalar)...")
    results = run_compression_study(keys, cfg)
    print_results(results, cfg.recall_k)
    print_takeaways(results, cfg.recall_k, len(tokens), cfg.glove_dim)

    print("\nWriting figures...")
    paths = generate_all_figures(results, keys, tokens, cfg.figures_dir, cfg.recall_k)
    for p in paths:
        print(f"  {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
