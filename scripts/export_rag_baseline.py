#!/usr/bin/env python3
"""Export full-precision rank-1 labels and eval manifest for auto RAG queries."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from vector_linalg.config import load_config
from vector_linalg.rag import (
    export_full_baseline_queries_yaml,
    fetch_rag_embeddings,
    build_rag_compressed_list,
    write_rag_topk_drift_report,
)


def main() -> None:
    cfg = load_config()
    bundle = fetch_rag_embeddings(cfg)
    yaml_path = export_full_baseline_queries_yaml(cfg, bundle)
    print(f"Wrote {yaml_path}")
    print(f"Wrote {cfg.rag_eval_manifest_path}")
    print(f"  {len(bundle.auto_query_texts)} auto queries, recall_k={cfg.rag.recall_k}")
    compressed = build_rag_compressed_list(bundle.chunk_matrix, cfg)
    drift = write_rag_topk_drift_report(cfg, bundle, compressed)
    print(f"Wrote {drift}")


if __name__ == "__main__":
    main()
