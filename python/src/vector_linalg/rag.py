"""RAG retrieval benchmark under vector compression."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import polars as pl
import yaml

from vector_linalg.canvas import build_canvas_chunks
from vector_linalg.compression import (
    CompressedVectors,
    compress_rank_k,
    compress_scalar,
    compress_sign,
)
from vector_linalg.config import ProjectConfig
from vector_linalg.embeddings import embed_texts, embedding_matrix
from vector_linalg.metrics import MethodResult, build_jl_compressed, evaluate_rag_method


@dataclass(frozen=True)
class RagBundle:
    chunk_ids: list[str]
    chunk_texts: list[str]
    chunk_matrix: np.ndarray
    query_texts: list[str]
    query_matrix: np.ndarray
    gold_chunk_indices: list[int]


def load_yaml_corpus(path) -> tuple[list[str], list[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    chunks = raw["chunks"]
    return [c["id"] for c in chunks], [str(c["text"]).strip() for c in chunks]


def load_queries(
    path,
    chunk_ids: list[str],
    chunk_texts: list[str] | None = None,
) -> tuple[list[str], list[int]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
    query_texts: list[str] = []
    gold_indices: list[int] = []
    skipped = 0
    for q in raw.get("queries", []):
        qtext = str(q["text"]).strip()
        gold_id = q["relevant_chunk"]
        if gold_id not in id_to_idx:
            skipped += 1
            continue
        query_texts.append(qtext)
        gold_indices.append(id_to_idx[gold_id])

    if skipped:
        print(f"  Skipped {skipped} queries with stale chunk ids (label rag_queries.yaml later).")

    if not query_texts and chunk_texts:
        print("  Using synthetic hold-out queries (first 200 chars of sample chunks).")
        n = min(12, len(chunk_texts))
        step = max(1, len(chunk_texts) // n)
        for i in range(0, len(chunk_texts), step):
            if len(query_texts) >= n:
                break
            query_texts.append(chunk_texts[i][:200])
            gold_indices.append(i)

    if not query_texts:
        raise RuntimeError("No RAG queries available.")
    return query_texts, gold_indices


def build_corpus_chunks(cfg: ProjectConfig, *, refresh: bool = False) -> tuple[list[str], list[str]]:
    if cfg.rag.source == "yaml":
        return load_yaml_corpus(cfg.rag.corpus_path)

    if cfg.rag.source == "canvas":
        print("Syncing content from Canvas...")
        chunks = build_canvas_chunks(
            course_id=cfg.rag.canvas.course_id,
            chunk_chars=cfg.rag.chunk_chars,
            chunk_overlap=cfg.rag.chunk_overlap,
            refresh=refresh,
            dest_dir=cfg.canvas_pdf_dir,
        )
        if not chunks:
            raise RuntimeError("No text extracted from Canvas")
        manifest = [
            {"chunk_id": c.chunk_id, "source_file": c.source_file, "preview": c.text[:200]}
            for c in chunks
        ]
        (cfg.data_dir / "chunk_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(f"  -> {len(chunks)} chunks from Canvas pages/PDFs")
        return [c.chunk_id for c in chunks], [c.text for c in chunks]

    raise RuntimeError(f"Unknown rag.source: {cfg.rag.source!r} (use yaml or canvas)")


def fetch_rag_embeddings(cfg: ProjectConfig, *, refresh: bool = False) -> RagBundle:
    chunk_ids, chunk_texts = build_corpus_chunks(cfg, refresh=refresh)
    query_texts, gold_indices = load_queries(cfg.rag.queries_path, chunk_ids, chunk_texts)

    if (
        not refresh
        and cfg.rag_chunks_cache.exists()
        and cfg.rag_queries_cache.exists()
    ):
        chunk_df = pl.read_parquet(cfg.rag_chunks_cache)
        query_df = pl.read_parquet(cfg.rag_queries_cache)
        if chunk_df.height == len(chunk_ids):
            _, chunk_matrix = embedding_matrix(chunk_df.rename({"chunk_id": "token"}))
            _, query_matrix = embedding_matrix(query_df.rename({"query": "token"}))
            return RagBundle(
                chunk_ids=chunk_df["chunk_id"].to_list(),
                chunk_texts=chunk_df["text"].to_list(),
                chunk_matrix=chunk_matrix,
                query_texts=query_df["query"].to_list(),
                query_matrix=query_matrix,
                gold_chunk_indices=query_df["gold_chunk_index"].to_list(),
            )

    print("Embedding RAG chunks...")
    chunk_matrix = embed_texts(chunk_texts, cfg)
    print("Embedding RAG queries...")
    query_matrix = embed_texts(query_texts, cfg)

    dim = chunk_matrix.shape[1]
    chunk_rows = [
        {
            "chunk_id": cid,
            "text": text,
            **{f"d{i}": float(chunk_matrix[j, i]) for i in range(dim)},
        }
        for j, (cid, text) in enumerate(zip(chunk_ids, chunk_texts, strict=True))
    ]
    query_rows = [
        {
            "query": q,
            "gold_chunk_index": gold_indices[j],
            **{f"d{i}": float(query_matrix[j, i]) for i in range(dim)},
        }
        for j, q in enumerate(query_texts)
    ]

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(chunk_rows).write_parquet(cfg.rag_chunks_cache)
    pl.DataFrame(query_rows).write_parquet(cfg.rag_queries_cache)

    meta_path = cfg.data_dir / "rag_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "n_chunks": len(chunk_ids),
                "n_queries": len(query_texts),
                "recall_k": cfg.rag.recall_k,
                "source": cfg.rag.source,
                "corpus": cfg.rel_path(cfg.rag.corpus_path),
                "queries": cfg.rel_path(cfg.rag.queries_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return RagBundle(
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        chunk_matrix=chunk_matrix,
        query_texts=query_texts,
        query_matrix=query_matrix,
        gold_chunk_indices=gold_indices,
    )


def run_rag_compression_study(bundle: RagBundle, cfg: ProjectConfig) -> list[MethodResult]:
    keys = bundle.chunk_matrix
    query_matrix = bundle.query_matrix

    rng = np.random.default_rng(cfg.random_seed)
    n = keys.shape[0]
    pairs = rng.integers(0, n, size=(min(cfg.n_distance_pairs, n * n), 2))

    compressed_list: list[CompressedVectors] = [
        CompressedVectors(
            name="full_precision",
            keys=keys.astype(np.float32),
            bits_per_dim=32.0,
            metadata={"method": "none"},
        )
    ]

    for k in cfg.compression.jl_dims:
        if k < keys.shape[1]:
            compressed_list.append(build_jl_compressed(keys, k, rng))
    for k in cfg.compression.rank_k:
        compressed_list.append(compress_rank_k(keys, k))
    compressed_list.append(compress_sign(keys))
    for bits in cfg.compression.scalar_bits:
        compressed_list.append(compress_scalar(keys, bits))

    results: list[MethodResult] = []
    for comp in compressed_list:
        results.append(
            evaluate_rag_method(
                keys,
                comp,
                query_matrix,
                bundle.gold_chunk_indices,
                pairs=pairs,
                recall_k=cfg.rag.recall_k,
            )
        )
    return results


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    text: str
    score: float
    rank: int


def _compressed_index(
    keys: np.ndarray,
    method: str,
    cfg: ProjectConfig,
    rng: np.random.Generator,
) -> CompressedVectors:
    if method == "full_precision":
        return CompressedVectors(
            name="full_precision",
            keys=keys.astype(np.float32),
            bits_per_dim=32.0,
            metadata={"method": "none"},
        )
    if method.startswith("jl_"):
        k = int(method.split("_", 1)[1])
        return build_jl_compressed(keys, k, rng)
    if method.startswith("rank_"):
        k = int(method.split("_", 1)[1])
        return compress_rank_k(keys, k)
    if method == "sign_1bit":
        return compress_sign(keys)
    if method.startswith("scalar_"):
        bits = int(method.split("_", 1)[1].replace("bit", ""))
        return compress_scalar(keys, bits)
    raise ValueError(f"Unknown compression method: {method!r}")


def search_corpus(
    query: str,
    bundle: RagBundle,
    cfg: ProjectConfig,
    *,
    top_k: int = 5,
    method: str = "full_precision",
) -> list[SearchHit]:
    """Embed a question and return top matching lecture chunks from the class index."""
    from vector_linalg.compression import cosine_scores, scores_from_compressed

    q = embed_texts([query.strip()], cfg)[0]
    keys = bundle.chunk_matrix
    rng = np.random.default_rng(cfg.random_seed)
    comp = _compressed_index(keys, method, cfg, rng)
    if comp.metadata.get("method") == "none":
        scores = cosine_scores(keys, q)
    else:
        scores = scores_from_compressed(keys, q, comp)

    order = np.argsort(-scores)[:top_k]
    return [
        SearchHit(
            chunk_id=bundle.chunk_ids[i],
            text=bundle.chunk_texts[i],
            score=float(scores[i]),
            rank=rank,
        )
        for rank, i in enumerate(order, start=1)
    ]
