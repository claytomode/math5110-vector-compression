#!/usr/bin/env python3
"""List RAG chunk ids after Canvas PDF sync (for labeling rag_queries.yaml)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from vector_linalg.canvas import build_canvas_chunks
from vector_linalg.config import load_config
from vector_linalg.pdf_corpus import build_pdf_chunks


def main() -> None:
    cfg = load_config()
    if cfg.rag.source == "canvas":
        chunks = build_canvas_chunks(
            course_id=cfg.rag.canvas.course_id,
            chunk_chars=cfg.rag.chunk_chars,
            chunk_overlap=cfg.rag.chunk_overlap,
            refresh=True,
            dest_dir=cfg.canvas_pdf_dir,
        )
    else:
        pdf_dir = cfg.canvas_pdf_dir
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs in {pdf_dir}. Set rag.source: canvas or add PDFs.")
            return
        chunks = build_pdf_chunks(
            pdfs,
            chunk_chars=cfg.rag.chunk_chars,
            chunk_overlap=cfg.rag.chunk_overlap,
        )
    rows = [
        {
            "chunk_id": c.chunk_id,
            "source_file": c.source_file,
            "preview": c.text[:160] + ("..." if len(c.text) > 160 else ""),
        }
        for c in chunks
    ]
    out = cfg.data_dir / "chunk_manifest.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} chunks to {out}\n")
    for row in rows[:20]:
        print(f"{row['chunk_id']}")
        print(f"  {row['source_file']}: {row['preview']}\n")
    if len(rows) > 20:
        print(f"... and {len(rows) - 20} more")


if __name__ == "__main__":
    main()
