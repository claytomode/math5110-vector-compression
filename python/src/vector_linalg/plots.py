"""Figures for compression survey application."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from vector_linalg.metrics import MethodResult


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_recall_vs_bits(results: list[MethodResult], recall_k: int) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.scatter(r.bits_per_dim, r.recall_at_k, s=60)
        ax.annotate(r.method, (r.bits_per_dim, r.recall_at_k), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Bits per dimension (approx storage)")
    ax.set_ylabel(f"Recall@{recall_k}")
    ax.set_title("Token nearest-neighbor recall vs compression")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def figure_distance_error(results: list[MethodResult]) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r.method for r in results]
    errs = [r.mean_distance_rel_error for r in results]
    ax.barh(names, errs)
    ax.set_xlabel("Mean relative pairwise distance error")
    ax.set_title("Geometry distortion by compression method")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def figure_token_pca(
    keys: np.ndarray,
    tokens: list[str],
    *,
    highlight: list[str] | None = None,
) -> Figure:
    keys = keys - keys.mean(axis=0)
    _, _, vt = np.linalg.svd(keys, full_matrices=False)
    coords = keys @ vt[:2, :].T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.5)
    highlight = highlight or []
    for i, t in enumerate(tokens):
        if t in highlight:
            ax.annotate(t, (coords[i, 0], coords[i, 1]), fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Token embedding cloud (GloVe, 2D PCA)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def generate_all_figures(
    results: list[MethodResult],
    keys: np.ndarray,
    tokens: list[str],
    figures_dir: Path,
    recall_k: int,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        _save(figure_recall_vs_bits(results, recall_k), figures_dir / "recall_vs_bits.png"),
        _save(figure_distance_error(results), figures_dir / "distance_error.png"),
        _save(
            figure_token_pca(keys, tokens, highlight=["king", "queen", "man", "woman", "paris", "france"]),
            figures_dir / "token_pca.png",
        ),
    ]
