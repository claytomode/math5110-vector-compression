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
class ProjectConfig:
    repo_root: Path
    data_dir: Path
    figures_dir: Path
    glove_url: str
    glove_dim: int
    max_glove_scan_lines: int
    tokens: tuple[str, ...]
    n_query_tokens: int
    n_distance_pairs: int
    recall_k: int
    random_seed: int
    compression: CompressionSweep

    @property
    def embeddings_cache(self) -> Path:
        return self.data_dir / "token_embeddings.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "metadata.json"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[4]


def load_config(path: Path | None = None) -> ProjectConfig:
    root = _repo_root()
    cfg_path = path or root / "python" / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    comp = raw.get("compression", {})
    return ProjectConfig(
        repo_root=root,
        data_dir=root / "python" / "data",
        figures_dir=root / "python" / "figures",
        glove_url=raw.get(
            "glove_url",
            "https://nlp.stanford.edu/data/glove.6B.zip",
        ),
        glove_dim=int(raw.get("glove_dim", 50)),
        max_glove_scan_lines=int(raw.get("max_glove_scan_lines", 400_000)),
        tokens=tuple(raw["tokens"]),
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
