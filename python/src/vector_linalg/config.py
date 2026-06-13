"""Project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CompressionSweep:
    jl_dims: tuple[int, ...]
    rank_k: tuple[int, ...]
    scalar_bits: tuple[int, ...]


@dataclass(frozen=True)
class EmbeddingsConfig:
    provider: str
    model: str
    dimensions: int | None
    batch_size: int
    azure_deployment: str | None
    azure_api_version: str


@dataclass(frozen=True)
class CanvasConfig:
    course_id: int | None


@dataclass(frozen=True)
class GithubBookConfig:
    owner: str
    repo: str
    branch: str
    chapters_dir: str


@dataclass(frozen=True)
class RagConfig:
    enabled: bool
    source: str
    corpus_path: Path
    queries_path: Path
    recall_k: int
    chunk_chars: int
    chunk_overlap: int
    canvas: CanvasConfig
    github_book: GithubBookConfig


@dataclass(frozen=True)
class ProjectConfig:
    repo_root: Path
    data_dir: Path
    figures_dir: Path
    tokens: tuple[str, ...]
    embeddings: EmbeddingsConfig
    rag: RagConfig
    n_query_tokens: int
    n_distance_pairs: int
    recall_k: int
    random_seed: int
    compression: CompressionSweep

    @property
    def embeddings_cache(self) -> Path:
        return self.data_dir / "token_embeddings.parquet"

    @property
    def rag_chunks_cache(self) -> Path:
        return self.data_dir / "rag_chunk_embeddings.parquet"

    @property
    def rag_queries_cache(self) -> Path:
        return self.data_dir / "rag_query_embeddings.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "metadata.json"

    @property
    def figures_rag_dir(self) -> Path:
        return self.figures_dir / "rag"

    @property
    def github_book_cache_dir(self) -> Path:
        return self.data_dir / "github_book"

    @property
    def canvas_pdf_dir(self) -> Path:
        return self.data_dir / "canvas_pdfs"

    def rel_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    pyproject_candidates: list[Path] = []
    for parent in here.parents:
        if (parent / "python" / "config.yaml").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            pyproject_candidates.append(parent)
    for parent in pyproject_candidates:
        if (parent / "python").is_dir():
            return parent
    if pyproject_candidates:
        return pyproject_candidates[0]
    return here.parents[4]


def load_config(path: Path | None = None) -> ProjectConfig:
    if path is not None:
        cfg_path = path.resolve()
        root = cfg_path.parents[1]
    else:
        root = _repo_root()
        cfg_path = root / "python" / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    comp = raw.get("compression", {})
    emb = raw.get("embeddings", {})
    rag = raw.get("rag", {})
    canvas = rag.get("canvas", {})
    book = rag.get("github_book", {})
    return ProjectConfig(
        repo_root=root,
        data_dir=root / "python" / "data",
        figures_dir=root / "python" / "figures",
        tokens=tuple(raw["tokens"]),
        embeddings=EmbeddingsConfig(
            provider=emb.get("provider", "openai"),
            model=emb.get("model", "text-embedding-3-small"),
            dimensions=emb.get("dimensions"),
            batch_size=int(emb.get("batch_size", 128)),
            azure_deployment=emb.get("azure_deployment"),
            azure_api_version=emb.get("azure_api_version", "2024-02-01"),
        ),
        rag=RagConfig(
            enabled=bool(rag.get("enabled", True)),
            source=rag.get("source", "yaml"),
            corpus_path=root / "python" / "data" / rag.get("corpus_file", "rag_corpus.yaml"),
            queries_path=root / "python" / "data" / rag.get("queries_file", "rag_queries.yaml"),
            recall_k=int(rag.get("recall_k", 3)),
            chunk_chars=int(rag.get("chunk_chars", 900)),
            chunk_overlap=int(rag.get("chunk_overlap", 120)),
            canvas=CanvasConfig(
                course_id=canvas.get("course_id"),
            ),
            github_book=GithubBookConfig(
                owner=book.get("owner", "wanghemath"),
                repo=book.get("repo", "Book-AdvancedLinearAlgebraAI"),
                branch=book.get("branch", "main"),
                chapters_dir=book.get("chapters_dir", "chapters"),
            ),
        ),
        n_query_tokens=int(raw.get("n_query_tokens", 80)),
        n_distance_pairs=int(raw.get("n_distance_pairs", 2000)),
        recall_k=int(raw.get("recall_k", 10)),
        random_seed=int(raw.get("random_seed", 5110)),
        compression=CompressionSweep(
            jl_dims=tuple(comp.get("jl_dims", [8, 16, 32])),
            rank_k=tuple(comp.get("rank_k", [4, 8, 16])),
            scalar_bits=tuple(comp.get("scalar_bits", [2, 4, 8])),
        ),
    )
