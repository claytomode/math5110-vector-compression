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
from vector_linalg.rag import fetch_rag_embeddings, run_rag_compression_study, storage_report
from vector_linalg.study import run_compression_study

HEADLINE_METHODS = frozenset({
    "full_precision", "jl_128", "sign_1bit", "scalar_2bit", "scalar_3bit", "scalar_4bit", "scalar_8bit",
    "turboquant_2bit", "turboquant_3bit", "turboquant_4bit", "turboquant_8bit", "rank_64",
})


def _export_presentation_results(cfg, token_results, rag_results, rag_bundle) -> Path:
    import json

    def row(r):
        return {
            "method": r.method,
            "overlap_or_recall": round(r.recall_at_k, 4),
            "bits_per_dim": round(r.bits_per_dim, 2),
            "compression_ratio": round(r.compression_ratio, 2),
            "distance_rel_err": round(r.mean_distance_rel_error, 4),
        }

    payload = {
        "token_study": {
            "recall_k": cfg.recall_k,
            "methods": [row(r) for r in token_results if r.method in HEADLINE_METHODS],
        },
        "rag_study": {
            "recall_k": cfg.rag.recall_k,
            "n_chunks": len(rag_bundle.chunk_ids),
            "n_eval_queries": len(rag_bundle.auto_query_texts),
            "ground_truth": "full_precision_cosine_top_k",
            "methods": [row(r) for r in rag_results if r.method in HEADLINE_METHODS],
            "index_storage": storage_report(rag_bundle, cfg),
        },
    }
    out = cfg.data_dir / "presentation_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


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
        print("\nRAG retrieval benchmark (auto queries; full-precision top-k = truth)...")
        rag_bundle = fetch_rag_embeddings(cfg)
        print(f"  {len(rag_bundle.chunk_ids)} chunks, {len(rag_bundle.auto_query_texts)} eval queries")
        rag_results = run_rag_compression_study(rag_bundle, cfg)
        print_rag_results(rag_results, cfg.rag.recall_k)
        print(f"  Top-k drift report: {cfg.rag_drift_report_path}")
        rag_paths = generate_rag_figures(
            results,
            rag_results,
            cfg.figures_rag_dir,
            token_k=cfg.recall_k,
            rag_k=cfg.rag.recall_k,
            drift_path=cfg.rag_drift_report_path,
        )
        print("\nWriting RAG figures...")
        for p in rag_paths:
            print(f"  {p}")

        summary = _export_presentation_results(cfg, results, rag_results, rag_bundle)
        print(f"  Presentation summary: {summary}")

    print("\nDone.")


if __name__ == "__main__":
    main()
