#!/usr/bin/env python3
"""Gradio UI: search course index + compare compressed index sizes."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

import gradio as gr
import polars as pl

from vector_linalg.config import load_config
from vector_linalg.rag import (
    fetch_rag_embeddings,
    list_compression_methods,
    search_corpus,
    storage_report,
)

COMPARE_METHODS = ("full_precision", "jl_128", "sign_1bit", "scalar_8bit")


@lru_cache(maxsize=1)
def _load():
    cfg = load_config()
    bundle = fetch_rag_embeddings(cfg)
    methods = list_compression_methods(cfg, bundle.chunk_matrix.shape[1])
    storage_df = pl.DataFrame(storage_report(bundle, cfg)).sort("index_bytes")
    corpus = cfg.rag.source
    n_chunks = len(bundle.chunk_ids)
    dim = bundle.chunk_matrix.shape[1]
    return cfg, bundle, methods, storage_df, corpus, n_chunks, dim


def _format_hits(hits) -> str:
    if not hits:
        return "_No results._"
    parts: list[str] = []
    for hit in hits:
        preview = hit.text[:700].replace("\n", " ")
        if len(hit.text) > 700:
            preview += "..."
        parts.append(
            f"### #{hit.rank} — score {hit.score:.3f}\n"
            f"**`{hit.chunk_id}`**\n\n{preview}"
        )
    return "\n\n---\n\n".join(parts)


def run_search(query: str, method: str, top_k: int, compare: bool) -> tuple[str, str]:
    query = (query or "").strip()
    if not query:
        return "_Enter a question._", ""

    cfg, bundle, methods, storage_df, corpus, n_chunks, dim = _load()
    if method not in methods:
        method = "full_precision"

    if compare:
        blocks: list[str] = []
        for m in COMPARE_METHODS:
            if m not in methods:
                continue
            hits = search_corpus(query, bundle, cfg, top_k=int(top_k), method=m)
            blocks.append(f"## Index: `{m}`\n\n{_format_hits(hits)}")
        body = "\n\n".join(blocks)
        subtitle = f"Compared {len(blocks)} compression indexes for the same query."
    else:
        hits = search_corpus(query, bundle, cfg, top_k=int(top_k), method=method)
        body = _format_hits(hits)
        subtitle = f"Single index: `{method}`."

    header = (
        f"**Corpus:** {corpus} · **{n_chunks}** chunks · **d={dim}**  \n"
        f"{subtitle}"
    )
    return header, body


def storage_table_md() -> str:
    _, _, _, storage_df, corpus, n_chunks, dim = _load()
    lines = [
        f"**Corpus:** `{corpus}` · **{n_chunks}** vectors · **d={dim}**",
        "",
        "| Method | Index size | MB | Compression vs float32 | bits/dim |",
        "|--------|------------|-----|------------------------|----------|",
    ]
    for row in storage_df.iter_rows(named=True):
        size = f"{row['index_bytes']:,} B"
        lines.append(
            f"| {row['method']} | {size} | {row['index_mb']} | "
            f"{row['compression_ratio']}× | {row['bits_per_dim']} |"
        )
    lines.append("")
    lines.append(
        "_Sizes are approximate in-memory index storage (vectors + small metadata like JL matrix or rank-k basis)._"
    )
    return "\n".join(lines)


def main() -> None:
    cfg, _, methods, _, corpus, n_chunks, _ = _load()

    with gr.Blocks(title="MATH 5110 vector search") as demo:
        gr.Markdown(
            "# Course material search + compressed index sizes\n"
            f"Search the **{corpus}** corpus ({n_chunks} chunks). "
            "Toggle compare mode to see what different compressed indexes retrieve."
        )
        with gr.Row():
            storage = gr.Markdown(storage_table_md())

        with gr.Row():
            query = gr.Textbox(
                label="Question",
                placeholder="What is the singular value decomposition?",
                lines=2,
            )
        with gr.Row():
            method = gr.Dropdown(methods, value="full_precision", label="Index type")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
            compare = gr.Checkbox(
                label="Compare indexes (full, jl_128, sign_1bit, scalar_8bit)",
                value=False,
            )
        search_btn = gr.Button("Search", variant="primary")

        meta = gr.Markdown()
        results = gr.Markdown()

        search_btn.click(
            run_search,
            inputs=[query, method, top_k, compare],
            outputs=[meta, results],
        )
        query.submit(
            run_search,
            inputs=[query, method, top_k, compare],
            outputs=[meta, results],
        )

        gr.Markdown(
            f"Run locally: `uv run python scripts/search_ui.py` · "
            f"RAG source: `{cfg.rag.source}`"
        )

    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
