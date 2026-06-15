"""RAG retrieval benchmark under vector compression."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from vector_linalg.canvas import build_canvas_chunks
from vector_linalg.github_book import build_github_book_chunks
from vector_linalg.compression import (
    CompressedVectors,
    compress_rank_k,
    compress_scalar,
    compress_sign,
    compress_turboquant,
    scores_from_compressed,
)
from vector_linalg.config import ProjectConfig
from vector_linalg.embeddings import embed_texts, embedding_matrix
from vector_linalg.metrics import (
    MethodResult,
    build_jl_compressed,
    evaluate_rag_method,
    full_precision_top_k,
)


@dataclass(frozen=True)
class RagBundle:
    chunk_ids: list[str]
    chunk_texts: list[str]
    chunk_matrix: np.ndarray
    query_texts: list[str]
    query_matrix: np.ndarray
    gold_chunk_indices: list[int]
    auto_query_texts: list[str]
    auto_query_matrix: np.ndarray
    auto_source_chunk_indices: list[int]


def _chapter_key(chunk_id: str) -> str:
    return chunk_id.rsplit("_", 1)[0]


def section_title_from_chunk(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("##"):
        return None
    rest = stripped[2:].lstrip()
    numbered = re.match(
        r"^(\d+\.\d+\s+[A-Za-z][A-Za-z0-9\s\-—']{2,35})(?:\s+(?:This|We|The|Let|A|An|Before|Consider|In)\s|\\|\.)",
        rest,
    )
    if numbered:
        return numbered.group(1).strip()
    named = re.match(
        r"^([A-Za-z][A-Za-z0-9\s\-—',]{2,40}?)(?=\s+(?:Let|We|The|A|An|Before|Consider|This|In)\s|\\|\.)",
        rest,
    )
    if named:
        return named.group(1).strip()
    short = re.match(r"^([^\n\\]{5,40})", rest)
    return short.group(1).strip() if short else None


def query_text_from_chunk(text: str, style: str) -> str:
    if style == "preview":
        return text[:200].strip()
    title = section_title_from_chunk(text)
    if title:
        return f"What does the section \"{title}\" explain?"
    return None


def stratified_chunk_indices(
    chunk_ids: list[str],
    n: int,
    seed: int,
    *,
    pool: list[int] | None = None,
) -> list[int]:
    """Sample chunk indices across chapters (one stratum per chapter stem)."""
    if n <= 0:
        return []
    pool = pool if pool is not None else list(range(len(chunk_ids)))
    by_chapter: dict[str, list[int]] = defaultdict(list)
    for i in pool:
        by_chapter[_chapter_key(chunk_ids[i])].append(i)

    rng = np.random.default_rng(seed)
    chapters = sorted(by_chapter)
    n = min(n, len(pool))
    base = n // len(chapters)
    extra = n % len(chapters)

    picked: list[int] = []
    for j, chapter in enumerate(chapters):
        chapter_pool = by_chapter[chapter]
        count = min(len(chapter_pool), base + (1 if j < extra else 0))
        if count <= 0:
            continue
        choice = rng.choice(chapter_pool, size=count, replace=False)
        picked.extend(int(i) for i in choice)

    if len(picked) < n:
        remaining = sorted(set(pool) - set(picked))
        need = n - len(picked)
        extra_pick = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        picked.extend(int(i) for i in extra_pick)

    return sorted(picked[:n])


def build_auto_eval_queries(
    chunk_ids: list[str],
    chunk_texts: list[str],
    cfg: ProjectConfig,
) -> tuple[list[str], list[int]]:
    ev = cfg.rag.eval
    eligible = [i for i, t in enumerate(chunk_texts) if t.strip().startswith("##")]
    if not eligible:
        eligible = list(range(len(chunk_ids)))

    by_chapter: dict[str, list[int]] = defaultdict(list)
    for i in eligible:
        by_chapter[_chapter_key(chunk_ids[i])].append(i)

    target = min(ev.auto_n_queries, len(eligible))
    rng = np.random.default_rng(ev.auto_seed)
    chapters = sorted(by_chapter)
    per_chapter = max(1, target // len(chapters))

    query_texts: list[str] = []
    kept_indices: list[int] = []
    used: set[int] = set()

    for j, chapter in enumerate(chapters):
        pool = [i for i in by_chapter[chapter] if i not in used]
        rng.shuffle(pool)
        count = per_chapter + (1 if j < target % len(chapters) else 0)
        for i in pool:
            if len(query_texts) >= target:
                break
            q = query_text_from_chunk(chunk_texts[i], ev.auto_query_style)
            if not q:
                continue
            query_texts.append(q)
            kept_indices.append(i)
            used.add(i)
            count -= 1
            if count <= 0:
                break

    batch_no = 0
    while len(query_texts) < target:
        remaining_pool = [i for i in eligible if i not in used]
        if not remaining_pool:
            break
        batch = stratified_chunk_indices(
            chunk_ids,
            min((target - len(query_texts)) * 2, len(remaining_pool)),
            ev.auto_seed + 1000 + batch_no,
            pool=remaining_pool,
        )
        batch_no += 1
        if not batch:
            break
        for i in batch:
            used.add(i)
            q = query_text_from_chunk(chunk_texts[i], ev.auto_query_style)
            if not q:
                continue
            query_texts.append(q)
            kept_indices.append(i)
            if len(query_texts) >= target:
                break

    return query_texts, kept_indices


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

    if cfg.rag.source == "github_book":
        chunks = build_github_book_chunks(
            cfg.rag.github_book,
            chunk_chars=cfg.rag.chunk_chars,
            chunk_overlap=cfg.rag.chunk_overlap,
            cache_dir=cfg.github_book_cache_dir,
            refresh=refresh,
        )
        manifest = [
            {"chunk_id": c.chunk_id, "source_file": c.source_file, "preview": c.text[:200]}
            for c in chunks
        ]
        (cfg.data_dir / "chunk_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return [c.chunk_id for c in chunks], [c.text for c in chunks]

    raise RuntimeError(
        f"Unknown rag.source: {cfg.rag.source!r} (use yaml, canvas, or github_book)"
    )


def _empty_query_matrix(dim: int) -> np.ndarray:
    return np.empty((0, dim), dtype=np.float32)


def _load_rag_from_cache(
    cfg: ProjectConfig,
    chunk_ids: list[str],
    auto_query_texts: list[str],
    auto_source_indices: list[int],
    *,
    use_labeled: bool,
) -> RagBundle | None:
    if not cfg.rag_chunks_cache.exists():
        return None
    if auto_query_texts and not cfg.rag_auto_queries_cache.exists():
        return None
    if use_labeled and not cfg.rag_queries_cache.exists():
        return None

    chunk_df = pl.read_parquet(cfg.rag_chunks_cache)
    if chunk_df.height != len(chunk_ids):
        return None

    _, chunk_matrix = embedding_matrix(chunk_df.rename({"chunk_id": "token"}))
    dim = chunk_matrix.shape[1]

    query_texts: list[str] = []
    gold_indices: list[int] = []
    query_matrix = _empty_query_matrix(dim)
    if use_labeled:
        query_df = pl.read_parquet(cfg.rag_queries_cache)
        _, query_matrix = embedding_matrix(query_df.rename({"query": "token"}))
        query_texts = query_df["query"].to_list()
        gold_indices = query_df["gold_chunk_index"].to_list()

    auto_matrix = _empty_query_matrix(dim)
    loaded_auto_texts = auto_query_texts
    loaded_auto_sources = auto_source_indices
    if auto_query_texts:
        auto_df = pl.read_parquet(cfg.rag_auto_queries_cache)
        if auto_df.height != len(auto_query_texts):
            return None
        _, auto_matrix = embedding_matrix(auto_df.rename({"query": "token"}))
        loaded_auto_texts = auto_df["query"].to_list()
        loaded_auto_sources = auto_df["source_chunk_index"].to_list()

    if loaded_auto_texts and not cfg.rag_eval_manifest_path.exists():
        write_rag_eval_manifest(
            cfg,
            chunk_df["chunk_id"].to_list(),
            loaded_auto_texts,
            loaded_auto_sources,
            chunk_matrix,
            auto_matrix,
        )

    return RagBundle(
        chunk_ids=chunk_df["chunk_id"].to_list(),
        chunk_texts=chunk_df["text"].to_list(),
        chunk_matrix=chunk_matrix,
        query_texts=query_texts,
        query_matrix=query_matrix,
        gold_chunk_indices=gold_indices,
        auto_query_texts=loaded_auto_texts,
        auto_query_matrix=auto_matrix,
        auto_source_chunk_indices=loaded_auto_sources,
    )


def fetch_rag_embeddings(cfg: ProjectConfig, *, refresh: bool = False) -> RagBundle:
    chunk_ids, chunk_texts = build_corpus_chunks(cfg, refresh=refresh)
    use_labeled = cfg.rag.eval.use_labeled_queries
    if use_labeled:
        query_texts, gold_indices = load_queries(cfg.rag.queries_path, chunk_ids, chunk_texts)
    else:
        query_texts, gold_indices = [], []
    auto_query_texts, auto_source_indices = build_auto_eval_queries(chunk_ids, chunk_texts, cfg)

    if not refresh:
        cached = _load_rag_from_cache(
            cfg,
            chunk_ids,
            auto_query_texts,
            auto_source_indices,
            use_labeled=use_labeled,
        )
        if cached is not None:
            return cached

    chunk_matrix: np.ndarray | None = None
    query_matrix: np.ndarray | None = None
    if not refresh and cfg.rag_chunks_cache.exists():
        chunk_df = pl.read_parquet(cfg.rag_chunks_cache)
        if chunk_df.height == len(chunk_ids):
            _, chunk_matrix = embedding_matrix(chunk_df.rename({"chunk_id": "token"}))
            chunk_ids = chunk_df["chunk_id"].to_list()
            chunk_texts = chunk_df["text"].to_list()
            if use_labeled and cfg.rag_queries_cache.exists():
                query_df = pl.read_parquet(cfg.rag_queries_cache)
                if query_df.height == len(query_texts):
                    _, query_matrix = embedding_matrix(query_df.rename({"query": "token"}))
                    query_texts = query_df["query"].to_list()
                    gold_indices = query_df["gold_chunk_index"].to_list()

    if chunk_matrix is None:
        print("Embedding RAG chunks...")
        chunk_matrix = embed_texts(chunk_texts, cfg)
    dim = chunk_matrix.shape[1]
    if use_labeled and query_matrix is None:
        print("Embedding labeled RAG queries...")
        query_matrix = embed_texts(query_texts, cfg)
    elif not use_labeled:
        query_matrix = _empty_query_matrix(dim)

    auto_matrix = _empty_query_matrix(dim)
    if auto_query_texts:
        print(f"Embedding auto eval queries ({len(auto_query_texts)})...")
        auto_matrix = embed_texts(auto_query_texts, cfg)

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
    if use_labeled and query_texts:
        pl.DataFrame(query_rows).write_parquet(cfg.rag_queries_cache)
    if auto_query_texts:
        auto_rows = [
            {
                "query": q,
                "source_chunk_index": auto_source_indices[j],
                **{f"d{i}": float(auto_matrix[j, i]) for i in range(dim)},
            }
            for j, q in enumerate(auto_query_texts)
        ]
        pl.DataFrame(auto_rows).write_parquet(cfg.rag_auto_queries_cache)

    meta_path = cfg.data_dir / "rag_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "n_chunks": len(chunk_ids),
                "n_queries": len(query_texts) if use_labeled else 0,
                "n_auto_queries": len(auto_query_texts),
                "use_labeled_queries": use_labeled,
                "recall_k": cfg.rag.recall_k,
                "source": cfg.rag.source,
                "corpus": cfg.rel_path(cfg.rag.corpus_path),
                "queries": cfg.rel_path(cfg.rag.queries_path),
                "auto_query_style": cfg.rag.eval.auto_query_style,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_rag_eval_manifest(cfg, chunk_ids, auto_query_texts, auto_source_indices, chunk_matrix, auto_matrix)

    return RagBundle(
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        chunk_matrix=chunk_matrix,
        query_texts=query_texts,
        query_matrix=query_matrix,
        gold_chunk_indices=gold_indices,
        auto_query_texts=auto_query_texts,
        auto_query_matrix=auto_matrix,
        auto_source_chunk_indices=auto_source_indices,
    )


def write_rag_eval_manifest(
    cfg: ProjectConfig,
    chunk_ids: list[str],
    auto_query_texts: list[str],
    auto_source_indices: list[int],
    chunk_matrix: np.ndarray,
    auto_query_matrix: np.ndarray,
) -> None:
    """Record full-precision top-k per auto query (used as compression ground truth)."""
    if not auto_query_texts:
        return
    k = cfg.rag.recall_k
    entries: list[dict] = []
    for i, qtext in enumerate(auto_query_texts):
        q = auto_query_matrix[i]
        top = full_precision_top_k(chunk_matrix, q, k=k)
        entries.append(
            {
                "query": qtext,
                "source_chunk_id": chunk_ids[auto_source_indices[i]],
                "full_top_k": [
                    {"rank": rank, "chunk_id": chunk_ids[idx]}
                    for rank, idx in enumerate(top, start=1)
                ],
            }
        )
    cfg.rag_eval_manifest_path.write_text(
        json.dumps(
            {
                "recall_k": k,
                "n_queries": len(entries),
                "ground_truth": "full_precision_cosine_top_k",
                "queries": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def export_full_baseline_queries_yaml(
    cfg: ProjectConfig,
    bundle: RagBundle,
    dest: Path | None = None,
) -> Path:
    """Write rag_queries.yaml labels from full-precision rank-1 (scalable pseudo-labels)."""
    dest = dest or (cfg.data_dir / "rag_queries_full_baseline.yaml")
    k = cfg.rag.recall_k
    queries: list[dict[str, str]] = []
    for i, qtext in enumerate(bundle.auto_query_texts):
        q = bundle.auto_query_matrix[i]
        top = full_precision_top_k(bundle.chunk_matrix, q, k=k)
        queries.append({"text": qtext, "relevant_chunk": bundle.chunk_ids[top[0]]})
    dest.write_text(
        yaml.safe_dump(
            {
                "queries": queries,
                "_meta": {
                    "generated_from": "full_precision_rank1",
                    "n": len(queries),
                    "recall_k": k,
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return dest


def build_rag_compressed_list(keys: np.ndarray, cfg: ProjectConfig) -> list[CompressedVectors]:
    rng = np.random.default_rng(cfg.random_seed)
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
    for bits in cfg.compression.turboquant_stage_bits:
        compressed_list.append(compress_turboquant(keys, bits, rng))
    return compressed_list


def _compressed_top_k(
    chunk_keys: np.ndarray,
    query: np.ndarray,
    compressed: CompressedVectors,
    *,
    k: int,
) -> list[int]:
    scores = scores_from_compressed(chunk_keys, query, compressed)
    return np.argsort(-scores)[:k].tolist()


def write_rag_topk_drift_report(
    cfg: ProjectConfig,
    bundle: RagBundle,
    compressed_list: list[CompressedVectors],
) -> Path:
    """Per-method report: where compressed top-k differs from full-precision truth."""
    k = cfg.rag.recall_k
    chunk_ids = bundle.chunk_ids
    keys = bundle.chunk_matrix
    queries = bundle.auto_query_matrix
    query_texts = bundle.auto_query_texts

    full_tops: list[list[int]] = [
        full_precision_top_k(keys, queries[i], k=k) for i in range(len(queries))
    ]

    method_reports: list[dict] = []
    for comp in compressed_list:
        if comp.metadata.get("method") == "none":
            continue

        overlap_fracs: list[float] = []
        changed: list[dict] = []
        perfect = 0
        zero_overlap = 0

        for qi, q in enumerate(queries):
            full_idx = full_tops[qi]
            full_set = set(full_idx)
            comp_idx = _compressed_top_k(keys, q, comp, k=k)
            comp_set = set(comp_idx)
            overlap = len(full_set & comp_set)
            overlap_frac = overlap / k
            overlap_fracs.append(overlap_frac)

            if overlap == k:
                perfect += 1
            if overlap == 0:
                zero_overlap += 1
            if overlap < k:
                changed.append(
                    {
                        "query_idx": qi,
                        "query": query_texts[qi],
                        "source_chunk_id": chunk_ids[bundle.auto_source_chunk_indices[qi]],
                        "full_top_k": [chunk_ids[i] for i in full_idx],
                        "compressed_top_k": [chunk_ids[i] for i in comp_idx],
                        "overlap": overlap,
                        "only_in_full": [chunk_ids[i] for i in full_idx if i not in comp_set],
                        "only_in_compressed": [chunk_ids[i] for i in comp_idx if i not in full_set],
                    }
                )

        n = len(queries)
        method_reports.append(
            {
                "method": comp.name,
                "bits_per_dim": comp.bits_per_dim,
                "mean_overlap_fraction": float(np.mean(overlap_fracs)) if overlap_fracs else 0.0,
                "perfect_match_rate": perfect / n if n else 0.0,
                "zero_overlap_rate": zero_overlap / n if n else 0.0,
                "n_changed": len(changed),
                "changed_queries": changed,
            }
        )

    payload = {
        "recall_k": k,
        "n_queries": len(queries),
        "ground_truth": "full_precision_cosine_top_k",
        "methods": method_reports,
    }
    cfg.rag_drift_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cfg.rag_drift_report_path


def run_rag_compression_study(bundle: RagBundle, cfg: ProjectConfig) -> list[MethodResult]:
    keys = bundle.chunk_matrix

    rng = np.random.default_rng(cfg.random_seed)
    n = keys.shape[0]
    pairs = rng.integers(0, n, size=(min(cfg.n_distance_pairs, n * n), 2))

    compressed_list = build_rag_compressed_list(keys, cfg)
    write_rag_topk_drift_report(cfg, bundle, compressed_list)

    results: list[MethodResult] = []
    for comp in compressed_list:
        results.append(
            evaluate_rag_method(
                keys,
                comp,
                bundle.auto_query_matrix,
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
    if method.startswith("turboquant_"):
        bits = int(method.split("_", 1)[1].replace("bit", ""))
        return compress_turboquant(keys, bits, rng)
    raise ValueError(f"Unknown compression method: {method!r}")


def list_compression_methods(cfg: ProjectConfig, dim: int) -> list[str]:
    methods = ["full_precision"]
    for k in cfg.compression.jl_dims:
        if k < dim:
            methods.append(f"jl_{k}")
    for k in cfg.compression.rank_k:
        methods.append(f"rank_{k}")
    methods.append("sign_1bit")
    for bits in cfg.compression.scalar_bits:
        methods.append(f"scalar_{bits}bit")
    for bits in cfg.compression.turboquant_stage_bits:
        methods.append(f"turboquant_{bits}bit")
    return methods


def index_storage_bytes(comp: CompressedVectors, n: int, d: int) -> int:
    """Approximate in-memory index size for n vectors of dimension d."""
    bits = comp.bits_per_dim * n * d
    method = comp.metadata.get("method")
    if method == "johnson_lindenstrauss":
        r = comp.metadata.get("r")
        if r is not None:
            bits += int(r.size) * 32
    elif method == "rank_k_svd":
        basis = comp.metadata.get("basis")
        if basis is not None:
            bits += int(basis.size) * 32
    elif method == "scalar_uniform":
        bits += int(2 * d * 32)
    elif method == "sign_quantization":
        bits += int(n * 32)
    elif method == "turboquant":
        perm = comp.metadata.get("perm")
        if perm is not None:
            bits += int(perm.size) * int(np.ceil(np.log2(max(d, 2))))
        signs = comp.metadata.get("signs")
        if signs is not None:
            bits += int(signs.size)
        lo = comp.metadata.get("lo")
        hi = comp.metadata.get("hi")
        if lo is not None and hi is not None:
            bits += int((lo.size + hi.size) * 32)
        bits += int(n * 32)
    return int(bits / 8)


def storage_report(bundle: RagBundle, cfg: ProjectConfig) -> list[dict[str, float | str]]:
    keys = bundle.chunk_matrix
    n, d = keys.shape
    baseline = n * d * 4
    rng = np.random.default_rng(cfg.random_seed)
    rows: list[dict[str, float | str]] = []
    for method in list_compression_methods(cfg, d):
        comp = _compressed_index(keys, method, cfg, rng)
        nbytes = index_storage_bytes(comp, n, d)
        rows.append(
            {
                "method": method,
                "index_bytes": nbytes,
                "index_mb": round(nbytes / 1_000_000, 3),
                "compression_ratio": round(baseline / max(nbytes, 1), 2),
                "bits_per_dim": round(comp.bits_per_dim, 2),
            }
        )
    return rows


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
