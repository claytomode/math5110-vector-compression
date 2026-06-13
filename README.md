# Vector compression for high-dimensional data

**Survey:** Johnson–Lindenstrauss sketches, spectral truncation, sign quantization  
**Application:** compress embeddings on token vectors and RAG retrieval (demo yaml corpus or **Canvas course PDFs**)

Course project for applied linear algebra. Inspired by recent work on extreme vector compression ([Google Research — TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)); this repo implements **pedagogical** versions of the same linear-algebra ideas on word/token embeddings — not a reproduction of LLM KV-cache inference.

## Three-part structure

| Part | Content |
|------|---------|
| **1. Survey** | JL lemma, random projections, rank‑k / polar geometry, 1-bit signs, scalar quantization |
| **2. Computation** | NumPy implementations + recall@k / distance distortion metrics |
| **3. Application** | Token + RAG retrieval under compression; optional Canvas PDF corpus |

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

Set credentials in `.env` (gitignored). **Do not paste keys in chat.**

**Azure OpenAI / AI Foundry** (often has university/free Azure credits):

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
```

In `python/config.yaml`, set `embeddings.provider: azure` and `embeddings.azure_deployment` to your **deployment name** from the Azure portal.

**Direct OpenAI** (requires billing on platform.openai.com):

```
OPENAI_API_KEY=sk-...
```

Set `embeddings.provider: openai` in config.

### Canvas PDFs (optional RAG corpus)

1. Canvas → **Account → Settings → Approved Integrations** → create access token.
2. Add to `.env`:
   ```
   CANVAS_BASE_URL=https://yourschool.instructure.com
   CANVAS_API_TOKEN=...
   CANVAS_COURSE_ID=...
   ```
3. Set `rag.source: canvas` in `python/config.yaml`.
4. Run `uv run python scripts/list_rag_chunks.py` after first sync; update `python/data/rag_queries.yaml` with real chunk ids.

PDFs cache to `python/data/canvas_pdfs/` (gitignored). Generated embeddings (`*.parquet`) are local cache only — never commit `.env` or course PDFs.

```bash
uv sync
uv run python scripts/run_all.py
```

**Outputs:**

- `python/data/token_embeddings.parquet`, `metadata.json`
- `python/figures/*.png` (token embedding compression)
- `python/figures/rag/*.png` (RAG hit@k vs compression, token vs RAG compare)

**Notebook:** `python/notebooks/application.ipynb`  
**Survey write-up:** `docs/SURVEY.md`

## Repo layout

| Path | Purpose |
|------|---------|
| `python/src/vector_linalg/` | Embeddings, compression, metrics, plots |
| `scripts/run_all.py` | End-to-end pipeline |
| `docs/SURVEY.md` | Part 1 theory narrative |

## Data citation

Token vectors from the [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings) (`text-embedding-3-small`, cached locally). See `python/data/metadata.json` after first run.

## Presentation arc

1. **Survey (most of talk):** why high-dimensional vectors are expensive; JL preserves distances; polar/spectral = radius + direction; sign bits for dot products.
2. **Application (end):** show recall@k vs bits on token embeddings — “same math that powers compressed attention keys and vector search.”
