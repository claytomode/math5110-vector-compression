#!/usr/bin/env python3
"""Search MATH 5110 course PDF chunks by natural-language question."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from vector_linalg.config import load_config
from vector_linalg.rag import fetch_rag_embeddings, search_corpus


def main() -> None:
    if len(sys.argv) < 2:
        query = "What is the singular value decomposition?"
        print(f"No query given; using demo: {query!r}\n")
    else:
        query = " ".join(sys.argv[1:])

    cfg = load_config()
    bundle = fetch_rag_embeddings(cfg)
    print(f"Index: {len(bundle.chunk_ids)} chunks from Canvas PDFs\n")
    print(f"Query: {query}\n")

    for hit in search_corpus(query, bundle, cfg, top_k=5):
        source = hit.chunk_id.rsplit("_", 1)[0]
        print(f"#{hit.rank}  score={hit.score:.3f}  {source}")
        snippet = hit.text[:500].replace("\n", " ")
        try:
            print(snippet)
        except UnicodeEncodeError:
            print(snippet.encode("ascii", errors="replace").decode("ascii"))
        if len(hit.text) > 500:
            print("...")
        print()


if __name__ == "__main__":
    main()
