#!/usr/bin/env python3
"""Write presentation_results.json with headline benchmark numbers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from vector_linalg.config import load_config
from vector_linalg.embeddings import embedding_matrix, fetch_token_embeddings
from vector_linalg.metrics import MethodResult
from vector_linalg.rag import fetch_rag_embeddings, run_rag_compression_study, storage_report
from vector_linalg.study import run_compression_study

HEADLINE_METHODS = (
    "full_precision",
    "jl_128",
    "sign_1bit",
    "scalar_2bit",
    "scalar_3bit",
    "scalar_4bit",
    "scalar_8bit",
    "turboquant_2bit",
    "turboquant_3bit",
    "turboquant_4bit",
    "turboquant_8bit",
    "rank_64",
)


def _row(r: MethodResult) -> dict:
    return {
        "method": r.method,
        "overlap_or_recall": round(r.recall_at_k, 4),
        "bits_per_dim": round(r.bits_per_dim, 2),
        "compression_ratio": round(r.compression_ratio, 2),
        "distance_rel_err": round(r.mean_distance_rel_error, 4),
    }


def main() -> None:
    cfg = load_config()
    df = fetch_token_embeddings(cfg)
    tokens, keys = embedding_matrix(df)
    token_results = run_compression_study(keys, cfg)

    payload: dict = {
        "token_study": {
            "n_tokens": len(tokens),
            "dim": keys.shape[1],
            "recall_k": cfg.recall_k,
            "methods": [_row(r) for r in token_results if r.method in HEADLINE_METHODS],
        }
    }

    if cfg.rag.enabled:
        bundle = fetch_rag_embeddings(cfg)
        rag_results = run_rag_compression_study(bundle, cfg)
        payload["rag_study"] = {
            "n_chunks": len(bundle.chunk_ids),
            "n_eval_queries": len(bundle.auto_query_texts),
            "recall_k": cfg.rag.recall_k,
            "ground_truth": "full_precision_cosine_top_k",
            "methods": [_row(r) for r in rag_results if r.method in HEADLINE_METHODS],
            "index_storage": storage_report(bundle, cfg),
        }

    out = cfg.data_dir / "presentation_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
