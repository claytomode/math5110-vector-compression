"""Figures for compression survey application."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from vector_linalg.metrics import MethodResult

_FAMILY_COLORS = {
    "full": "#64748b",
    "scalar": "#10b981",
    "sign": "#3b82f6",
    "jl": "#f59e0b",
    "rank": "#8b5cf6",
    "turboquant": "#e11d48",
}


def _method_family(method: str) -> str:
    if method == "full_precision":
        return "full"
    if method.startswith("turboquant"):
        return "turboquant"
    if method.startswith("scalar"):
        return "scalar"
    if method.startswith("sign"):
        return "sign"
    if method.startswith("jl"):
        return "jl"
    if method.startswith("rank"):
        return "rank"
    return "full"


def _value_score(r: MethodResult) -> float:
    """Overlap × compression ratio — higher = more accuracy per storage saved."""
    return r.recall_at_k * r.compression_ratio


def _pareto_optimal(results: list[MethodResult]) -> list[MethodResult]:
    """Methods on the accuracy–compression Pareto frontier (excluding full baseline)."""
    candidates = [r for r in results if r.method != "full_precision"]
    frontier: list[MethodResult] = []
    for ri in candidates:
        dominated = any(
            rj.compression_ratio >= ri.compression_ratio
            and rj.recall_at_k >= ri.recall_at_k
            and (
                rj.compression_ratio > ri.compression_ratio or rj.recall_at_k > ri.recall_at_k
            )
            for rj in candidates
            if rj.method != ri.method
        )
        if not dominated:
            frontier.append(ri)
    return sorted(frontier, key=lambda r: r.compression_ratio)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def figure_recall_vs_bits(
    results: list[MethodResult],
    recall_k: int,
    *,
    title: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")

    for r in results:
        family = _method_family(r.method)
        color = _FAMILY_COLORS[family]
        size = 120 if r.method == "full_precision" else 70 + 20 * min(r.compression_ratio, 16)
        ax.scatter(
            r.bits_per_dim,
            r.recall_at_k,
            s=size,
            c=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.annotate(
            r.method,
            (r.bits_per_dim, r.recall_at_k),
            fontsize=7,
            xytext=(6, 6),
            textcoords="offset points",
            color="#334155",
        )

    frontier = _pareto_optimal(results)
    if len(frontier) >= 2:
        fx = [r.bits_per_dim for r in frontier]
        fy = [r.recall_at_k for r in frontier]
        order = np.argsort(fx)
        ax.plot(
            np.array(fx)[order],
            np.array(fy)[order],
            "--",
            color="#ef4444",
            alpha=0.6,
            linewidth=1.5,
            zorder=2,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Bits per dimension (storage cost →)", fontsize=10)
    ax.set_ylabel(f"Overlap vs full top-{recall_k} (accuracy →)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="600", pad=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, linestyle="--")
    legend_handles = [
        Patch(facecolor=c, label=label, edgecolor="white")
        for label, c in [
            ("scalar quant", _FAMILY_COLORS["scalar"]),
            ("sign 1-bit", _FAMILY_COLORS["sign"]),
            ("JL sketch", _FAMILY_COLORS["jl"]),
            ("rank-k SVD", _FAMILY_COLORS["rank"]),
            ("TurboQuant", _FAMILY_COLORS["turboquant"]),
            ("full precision", _FAMILY_COLORS["full"]),
        ]
    ]
    if len(frontier) >= 2:
        legend_handles.append(
            plt.Line2D([0], [0], linestyle="--", color="#ef4444", label="Pareto frontier")
        )
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


_FAMILY_LABELS = {
    "turboquant": "TurboQuant",
    "scalar": "Scalar quant",
    "sign": "Sign 1-bit",
    "jl": "JL sketch",
    "rank": "Rank-k SVD",
    "full": "Full precision",
}

def _short_method_label(method: str) -> str | None:
    if method.startswith("turboquant_"):
        bits = method.removeprefix("turboquant_").replace("bit", "")
        return f"TQ-{bits}"
    if method.startswith("scalar_"):
        bits = method.removeprefix("scalar_").replace("bit", "")
        return f"SC-{bits}"
    if method == "sign_1bit":
        return "SGN-1"
    if method.startswith("jl_"):
        return f"JL-{method.removeprefix('jl_')}"
    if method.startswith("rank_"):
        return f"RK-{method.removeprefix('rank_')}"
    if method == "full_precision":
        return "full"
    return None


_SHORT_LABEL_OFFSETS: dict[str, tuple[int, int]] = {
    "turboquant_8bit": (-10, 12),
    "scalar_8bit": (10, -14),
    "turboquant_4bit": (-10, 11),
    "scalar_4bit": (10, -13),
    "turboquant_3bit": (-10, -12),
    "scalar_3bit": (10, -11),
    "turboquant_2bit": (-10, 11),
    "scalar_2bit": (10, -12),
    "sign_1bit": (10, 10),
    "jl_128": (8, 6),
    "jl_64": (8, 4),
    "jl_32": (8, -4),
    "jl_16": (8, -8),
    "rank_64": (-10, 6),
    "rank_32": (-10, 0),
    "rank_16": (-10, -6),
    "rank_8": (-10, -10),
}


def _annotate_short_labels(ax, results: list[MethodResult], *, frontier_names: set[str]) -> None:
    for r in results:
        if r.method == "full_precision":
            continue
        label = _short_method_label(r.method)
        if label is None:
            continue
        on_frontier = r.method in frontier_names
        dx, dy = _SHORT_LABEL_OFFSETS.get(r.method, (8, 6))
        family = _method_family(r.method)
        ax.annotate(
            label,
            (r.compression_ratio, r.recall_at_k),
            fontsize=8 if on_frontier else 7,
            fontweight="700" if on_frontier else "500",
            color=_FAMILY_COLORS[family],
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#e2e8f0",
                "linewidth": 0.6,
                "alpha": 0.92,
            },
            zorder=7 if on_frontier else 6,
        )


def figure_compression_frontier(
    results: list[MethodResult],
    recall_k: int,
    *,
    title: str = "RAG retrieval: compression vs accuracy tradeoff",
) -> Figure:
    """Pareto plot — compression ratio (×) vs overlap with full-precision top-k."""
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")

    frontier = _pareto_optimal(results)
    frontier_names = {r.method for r in frontier}

    for r in results:
        if r.method == "full_precision":
            continue
        family = _method_family(r.method)
        on_frontier = r.method in frontier_names
        ax.scatter(
            r.compression_ratio,
            r.recall_at_k,
            s=130 if on_frontier else 85,
            c=_FAMILY_COLORS[family],
            marker="o",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
            zorder=4 if on_frontier else 3,
        )

    if len(frontier) >= 2:
        xs = [r.compression_ratio for r in frontier]
        ys = [r.recall_at_k for r in frontier]
        order = np.argsort(xs)
        ax.plot(
            np.array(xs)[order],
            np.array(ys)[order],
            "-",
            color="#ef4444",
            linewidth=2.5,
            alpha=0.85,
            zorder=4,
        )
        for r in frontier:
            ax.scatter(
                r.compression_ratio,
                r.recall_at_k,
                s=165,
                marker="o",
                facecolors="none",
                edgecolors="#ef4444",
                linewidths=2,
                zorder=5,
            )

    ax.axhline(
        next(r.recall_at_k for r in results if r.method == "full_precision"),
        color=_FAMILY_COLORS["full"],
        linestyle=":",
        alpha=0.7,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Compression ratio (higher = smaller index →)", fontsize=10)
    ax.set_ylabel(f"Top-{recall_k} overlap vs full precision (accuracy →)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="600", pad=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, linestyle="--")
    _annotate_short_labels(ax, results, frontier_names=frontier_names)

    families_present = {_method_family(r.method) for r in results if r.method != "full_precision"}
    legend_handles = [
        Patch(facecolor=_FAMILY_COLORS[f], label=_FAMILY_LABELS[f], edgecolor="white")
        for f in ("turboquant", "scalar", "sign", "jl", "rank")
        if f in families_present
    ]
    if len(frontier) >= 2:
        legend_handles.append(
            plt.Line2D([0], [0], linestyle="-", color="#ef4444", linewidth=2.5, label="Pareto frontier")
        )
    legend_handles.append(
        plt.Line2D([0], [0], linestyle=":", color=_FAMILY_COLORS["full"], label="Full-precision baseline")
    )
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    return fig


def figure_value_ranking(
    results: list[MethodResult],
    *,
    title: str = "Best bang-for-buck: overlap × compression ratio",
) -> Figure:
    ranked = sorted(
        (r for r in results if r.method != "full_precision"),
        key=_value_score,
        reverse=True,
    )
    names = [r.method for r in ranked]
    values = [_value_score(r) for r in ranked]
    overlaps = [r.recall_at_k for r in ranked]
    colors = [_FAMILY_COLORS[_method_family(r.method)] for r in ranked]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    y = np.arange(len(names))
    bars = ax.barh(y, values, color=colors, alpha=0.9, edgecolor="white", height=0.7)

    for i, (bar, r) in enumerate(zip(bars, ranked, strict=True)):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{overlaps[i]:.0%} overlap · {r.compression_ratio:.0f}×",
            va="center",
            fontsize=7,
            color="#475569",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Value score (overlap × compression ratio)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="600", pad=12)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig


def figure_rag_drift_summary(drift_path: Path) -> Figure | None:
    if not drift_path.exists():
        return None
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    methods = sorted(
        drift["methods"],
        key=lambda m: m["mean_overlap_fraction"],
        reverse=True,
    )
    names = [m["method"] for m in methods]
    perfect: list[float] = []
    partial: list[float] = []
    zero: list[float] = []
    for m in methods:
        perfect_pct = m["perfect_match_rate"] * 100
        zero_pct = m["zero_overlap_rate"] * 100
        partial_pct = max(0.0, 100.0 - perfect_pct - zero_pct)
        perfect.append(perfect_pct)
        partial.append(partial_pct)
        zero.append(zero_pct)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    y = np.arange(len(names))
    ax.barh(y, perfect, color="#10b981", label="Perfect top-k match", height=0.7)
    ax.barh(y, partial, left=perfect, color="#f59e0b", label="Partial drift", height=0.7)
    ax.barh(
        y,
        zero,
        left=np.array(perfect) + np.array(partial),
        color="#ef4444",
        label="Zero overlap",
        height=0.7,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of eval queries (%)", fontsize=10)
    ax.set_title(
        f"Top-{drift['recall_k']} drift from full precision ({drift['n_queries']} queries)",
        fontsize=11,
        fontweight="600",
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig


def figure_rag_compare(
    token_results: list[MethodResult],
    rag_results: list[MethodResult],
    *,
    token_k: int,
    rag_k: int,
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r.method for r in token_results]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, [r.recall_at_k for r in token_results], width, label=f"tokens @{token_k}")
    rag_by_name = {r.method: r.recall_at_k for r in rag_results}
    ax.bar(
        x + width / 2,
        [rag_by_name.get(n, 0.0) for n in names],
        width,
        label=f"RAG overlap@{rag_k} vs full",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Token NN recall vs RAG overlap with full-precision top-k")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _distance_error_label(err: float) -> str:
    if err == 0.0:
        return "0"
    if err < 0.01:
        return f"{err:.4f}"
    return f"{err:.3f}"


def figure_distance_error(results: list[MethodResult]) -> Figure:
    """Horizontal bar chart: pairwise distance error, sorted best → worst, colored by family."""
    ordered = sorted(results, key=lambda r: r.mean_distance_rel_error)
    n = len(ordered)
    fig_h = max(6.0, 0.38 * n + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")

    errs = [r.mean_distance_rel_error for r in ordered]
    colors = [_FAMILY_COLORS[_method_family(r.method)] for r in ordered]
    y = np.arange(n)

    bars = ax.barh(
        y,
        errs,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.72,
        zorder=3,
    )

    xmax = max(errs) if errs else 0.1
    outside_pad = xmax * 0.02
    for bar, err, r in zip(bars, errs, ordered, strict=True):
        family = _method_family(r.method)
        ax.text(
            bar.get_width() + outside_pad,
            bar.get_y() + bar.get_height() / 2,
            _distance_error_label(err),
            va="center",
            ha="left",
            fontsize=8,
            fontweight="600",
            color=_FAMILY_COLORS[family],
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#e2e8f0",
                "linewidth": 0.6,
                "alpha": 0.92,
            },
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([r.method for r in ordered], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel("Mean relative pairwise distance error (lower = better geometry)", fontsize=10)
    ax.set_title(
        "Geometry distortion by compression method",
        fontsize=11,
        fontweight="600",
        pad=12,
    )
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)

    families_present = {_method_family(r.method) for r in ordered}
    legend_handles = [
        Patch(facecolor=_FAMILY_COLORS[f], label=_FAMILY_LABELS[f], edgecolor="white")
        for f in ("turboquant", "scalar", "sign", "jl", "rank", "full")
        if f in families_present
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
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
    ax.set_title("Token embedding cloud (OpenAI, 2D PCA)")
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
        _save(
            figure_recall_vs_bits(results, recall_k, title="Token nearest-neighbor recall vs compression"),
            figures_dir / "recall_vs_bits.png",
        ),
        _save(
            figure_compression_frontier(
                results,
                recall_k,
                title="Token NN recall: compression vs accuracy tradeoff",
            ),
            figures_dir / "token_compression_frontier.png",
        ),
        _save(
            figure_value_ranking(
                results,
                title="Token study: best bang-for-buck (recall × compression)",
            ),
            figures_dir / "token_value_ranking.png",
        ),
        _save(figure_distance_error(results), figures_dir / "distance_error.png"),
        _save(
            figure_token_pca(keys, tokens, highlight=["king", "queen", "man", "woman", "paris", "france"]),
            figures_dir / "token_pca.png",
        ),
    ]


def generate_rag_figures(
    token_results: list[MethodResult],
    rag_results: list[MethodResult],
    figures_dir: Path,
    *,
    token_k: int,
    rag_k: int,
    drift_path: Path | None = None,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _save(
            figure_recall_vs_bits(
                rag_results,
                rag_k,
                title=f"RAG overlap@{rag_k} vs bits per dimension",
            ),
            figures_dir / "rag_hit_vs_bits.png",
        ),
        _save(
            figure_compression_frontier(rag_results, rag_k),
            figures_dir / "rag_compression_frontier.png",
        ),
        _save(
            figure_value_ranking(rag_results),
            figures_dir / "rag_value_ranking.png",
        ),
        _save(
            figure_rag_compare(token_results, rag_results, token_k=token_k, rag_k=rag_k),
            figures_dir / "token_vs_rag.png",
        ),
    ]
    if drift_path:
        drift_fig = figure_rag_drift_summary(drift_path)
        if drift_fig is not None:
            paths.append(_save(drift_fig, figures_dir / "rag_drift_summary.png"))
    return paths
