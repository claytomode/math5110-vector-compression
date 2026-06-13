from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StorageRow(BaseModel):
    method: str
    index_bytes: int
    index_mb: float
    compression_ratio: float
    bits_per_dim: float


class CorpusInfo(BaseModel):
    corpus: str
    n_chunks: int
    dim: int
    methods: list[str]
    compare_methods: list[str]
    storage: list[StorageRow]


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    score: float
    text: str
    preview: str


class MethodResults(BaseModel):
    method: str
    hits: list[SearchHit]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    method: str = "full_precision"
    top_k: int = Field(default=5, ge=1, le=20)
    compare: bool = False


class SearchResponse(BaseModel):
    ok: Literal[True] = True
    mode: Literal["single", "compare"]
    corpus: str
    n_chunks: int
    dim: int
    query: str
    hits: list[SearchHit] | None = None
    method: str | None = None
    comparisons: list[MethodResults] | None = None


class ErrorResponse(BaseModel):
    ok: Literal[False] = False
    error: str
