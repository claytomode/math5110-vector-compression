#!/usr/bin/env python3
"""Regenerate token embedding data and compression figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from vector_linalg.config import load_config
from vector_linalg.embeddings import embedding_matrix, fetch_token_embeddings
from vector_linalg.interpret import print_rag_results, print_results, print_takeaways
from vector_linalg.plots import generate_all_figures, generate_rag_figures
from vector_linalg.rag import fetch_rag_embeddings, run_rag_compression_study
from vector_linalg.study import run_compression_study


def main() -> None:
    cfg = load_config()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading token embeddings...")
    df = fetch_token_embeddings(cfg)
    tokens, keys = embedding_matrix(df)
    dim = keys.shape[1]
    print(f"  {len(tokens)} tokens x d={dim}")

    print("\nCompression study (JL, rank-k, sign, scalar)...")
    results = run_compression_study(keys, cfg)
    print_results(results, cfg.recall_k)
    print_takeaways(results, cfg.recall_k, len(tokens), dim, model=cfg.embeddings.model)

    print("\nWriting figures...")
    paths = generate_all_figures(results, keys, tokens, cfg.figures_dir, cfg.recall_k)
    for p in paths:
        print(f"  {p}")

    if cfg.rag.enabled:
        print("\nRAG retrieval benchmark (compress chunk index, score labeled queries)...")
        rag_bundle = fetch_rag_embeddings(cfg)
        print(f"  {len(rag_bundle.chunk_ids)} chunks, {len(rag_bundle.query_texts)} labeled queries")
        rag_results = run_rag_compression_study(rag_bundle, cfg)
        print_rag_results(rag_results, cfg.rag.recall_k)
        rag_paths = generate_rag_figures(
            results,
            rag_results,
            cfg.figures_rag_dir,
            token_k=cfg.recall_k,
            rag_k=cfg.rag.recall_k,
        )
        print("\nWriting RAG figures...")
        for p in rag_paths:
            print(f"  {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
