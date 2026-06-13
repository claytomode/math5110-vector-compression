from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.core.corpus import compare_methods, get_corpus_state
from backend.schemas.search import (
    CorpusInfo,
    MethodResults,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StorageRow,
)
from vector_linalg.rag import search_corpus

router = APIRouter(prefix="/api", tags=["search"])

PREVIEW_CHARS = 520


def _hit_models(hits) -> list[SearchHit]:
    out: list[SearchHit] = []
    for hit in hits:
        preview = hit.text[:PREVIEW_CHARS].replace("\n", " ")
        if len(hit.text) > PREVIEW_CHARS:
            preview += "…"
        out.append(
            SearchHit(
                rank=hit.rank,
                chunk_id=hit.chunk_id,
                score=hit.score,
                text=hit.text,
                preview=preview,
            )
        )
    return out


@router.get("/corpus", response_model=CorpusInfo)
def corpus_info() -> CorpusInfo:
    try:
        cfg, bundle, methods, storage = get_corpus_state()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    n, d = bundle.chunk_matrix.shape
    return CorpusInfo(
        corpus=cfg.rag.source,
        n_chunks=n,
        dim=d,
        methods=methods,
        compare_methods=compare_methods(methods),
        storage=[StorageRow(**row) for row in storage],
    )


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        cfg, bundle, methods, _ = get_corpus_state()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    n, d = bundle.chunk_matrix.shape

    if body.compare:
        selected = compare_methods(methods)
        if not selected:
            raise HTTPException(status_code=400, detail="No compare methods available.")
        comparisons: list[MethodResults] = []
        for method in selected:
            try:
                hits = search_corpus(
                    query,
                    bundle,
                    cfg,
                    top_k=body.top_k,
                    method=method,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Search failed for {method!r}: {exc}",
                ) from exc
            comparisons.append(MethodResults(method=method, hits=_hit_models(hits)))
        return SearchResponse(
            mode="compare",
            corpus=cfg.rag.source,
            n_chunks=n,
            dim=d,
            query=query,
            comparisons=comparisons,
        )

    method = body.method if body.method in methods else "full_precision"
    try:
        hits = search_corpus(
            query,
            bundle,
            cfg,
            top_k=body.top_k,
            method=method,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Search failed for {method!r}: {exc}",
        ) from exc

    return SearchResponse(
        mode="single",
        corpus=cfg.rag.source,
        n_chunks=n,
        dim=d,
        query=query,
        method=method,
        hits=_hit_models(hits),
    )
