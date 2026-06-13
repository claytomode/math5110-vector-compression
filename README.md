# Vector compression for high-dimensional data

**Survey:** Johnson–Lindenstrauss sketches, spectral truncation, sign quantization  
**Application:** compress token embedding vectors (GloVe) and measure nearest-neighbor recall

Course project for applied linear algebra. Inspired by recent work on extreme vector compression ([Google Research — TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)); this repo implements **pedagogical** versions of the same linear-algebra ideas on word/token embeddings — not a reproduction of LLM KV-cache inference.

## Three-part structure

| Part | Content |
|------|---------|
| **1. Survey** | JL lemma, random projections, rank‑k / polar geometry, 1-bit signs, scalar quantization |
| **2. Computation** | NumPy implementations + recall@k / distance distortion metrics |
| **3. Application** | ~200 GloVe token vectors; compression tradeoff curves |

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
uv run python scripts/run_all.py
```

**Outputs:**

- `python/data/token_embeddings.parquet`, `metadata.json`
- `python/figures/recall_vs_bits.png`
- `python/figures/distance_error.png`
- `python/figures/token_pca.png`

**Notebook:** `python/notebooks/application.ipynb`  
**Survey write-up:** `docs/SURVEY.md`

## Repo layout

| Path | Purpose |
|------|---------|
| `python/src/vector_linalg/` | Embeddings, compression, metrics, plots |
| `scripts/run_all.py` | End-to-end pipeline |
| `docs/SURVEY.md` | Part 1 theory narrative |

## Data citation

Token vectors from [GloVe](https://nlp.stanford.edu/projects/glove/) (Pennington et al., 2014), via the slim 50d subset in [eyaler/word2vec-slim](https://github.com/eyaler/word2vec-slim). See `python/data/metadata.json` after first run.

## Presentation arc

1. **Survey (most of talk):** why high-dimensional vectors are expensive; JL preserves distances; polar/spectral = radius + direction; sign bits for dot products.
2. **Application (end):** show recall@k vs bits on token embeddings — “same math that powers compressed attention keys and vector search.”
