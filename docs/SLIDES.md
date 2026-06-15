# MATH 5110 — Vector compression for embeddings & retrieval

**Presentation deck** — copy into PowerPoint / Google Slides / Quarto reveal.  
**Full math writeup (theory + results + figures):** [`WRITEUP.md`](WRITEUP.md)  
**Regenerate all figures + numbers:** `uv run python scripts/run_all.py`  
**Headline metrics (frozen):** `python/data/presentation_results.json`

---

## Slide 1 — Title

**Vector compression for high-dimensional data**

MATH 5110 applied linear algebra · `claytomode/math5110-vector-compression`

- Survey: JL, SVD, sign & scalar quant, **TurboQuant-style two-stage**
- Application: compressed **textbook RAG** (300-query auto eval) + live search UI

---

## Slide 2 — Why this project matters (not “just RAG”)

**Question:** Billions of vectors (embeddings, KV-cache keys, search indexes) → can we shrink them without breaking geometry?

**What we built:** A **reproducible lin-alg lab** that:

1. Implements the same **maps** Google cites (JL, spectral, 1-bit signs, TurboQuant pipeline)
2. Measures **honest retrieval** — 300 auto queries, full-precision top-3 = ground truth
3. Ships a **live demo** — search Prof. Wang’s MATH 5110 book with compressed indexes

**Why RAG here (not KV-cache):** Same dot-product math, but RAG = searchable index on **course material** you can demo in a browser. KV-cache = inside a running LLM (different stack, same compression ideas).

**What we learned (real findings, not failure):**

| Finding | Why it matters |
|---------|----------------|
| Raw JL **loses** top-k vs full precision | Theory (distances) ≠ engineering (ranking) |
| Scalar is a **strong baseline** at 4–8 bits | Honest negative result — good science |
| **TurboQuant beats scalar at aggressive bits** | 2-bit stage-1: **76%** overlap vs scalar 2-bit **63%** |
| Two-stage **QJL residual** fixes biased quant | This is Google’s actual insight |

> We didn’t reproduce Llama inference — we **tested the linear algebra** and showed **when** each method wins.

---

## Slide 3 — Where vectors live in AI

| System | Vectors | Compress what? |
|--------|---------|----------------|
| **RAG** (our demo) | Chunk embeddings | Search **index** |
| **KV cache** (TurboQuant) | Attention **keys** | Model **memory** |
| Embedding tables | Token vectors | Lookup table |

Storage $\propto n \cdot d \cdot$ bits per dimension.

**Motivation:** [TurboQuant — Google Research (2026)](https://research.google.blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

---

## Slide 4 — Project structure

| Part | What we did |
|------|-------------|
| **A. Survey** | JL, rank-k SVD, sign, scalar, TurboQuant (rotate → quant → QJL residual) |
| **B. Metrics** | Recall@k, distance distortion, **overlap@k vs full** (set overlap, rank-blind) |
| **C. Token study** | 230 words, $d=256$, recall@10 |
| **D. Book RAG** | 1380 chunks, **300 stratified queries**, overlap@3 |
| **E. Demo** | FastAPI + Svelte search UI |

---

## Slide 5 — Johnson–Lindenstrauss

Random $R \in \mathbb{R}^{k \times d}$, sketch $y = Rx$. Preserves **pairwise distances** with $k = O(\log n / \varepsilon^2)$.

**Our `jl_k`:** compress keys **and** queries into sketched space.

**Result:** weak top-k on this corpus (`jl_128` → **39%** RAG overlap at 2×) — distance lemma ≠ retrieval ranking.

---

## Slide 6 — TurboQuant pipeline (our implementation)

From the [TurboQuant blog](https://research.google.blog/turboquant-redefining-ai-efficiency-with-extreme-compression/):

1. **Rotate** (perm + random signs) — flatten geometry, O($d$) metadata
2. **Stage 1:** scalar quant in rotated space (`turboquant_2bit` … `turboquant_8bit`)
3. **Stage 2:** 1-bit **QJL** on residual + **full-precision query** at score time

$$
q^\top x \approx q^\top \hat{x}_{\text{stage1}} + \|r\|\,\frac{\mathrm{sign}(r)^\top q}{\sqrt{d}}


**Sizes we benchmark:** `turboquant_2bit`, `_3bit`, `_4bit`, `_8bit`

---

## Slide 7 — Sign, scalar, rank-k

- **Sign 1-bit:** extreme compression (~28×), ~48% RAG overlap alone
- **Scalar 2/4/8-bit:** strong baselines at moderate–high bits
- **Rank-k SVD:** shared subspace; middling retrieval here

---

## Slide 8 — Evaluation design

| Metric | Meaning |
|--------|---------|
| **Overlap@k** | $\| \text{full top-}k \cap \text{compressed top-}k \| / k$ — **swaps still count** |
| **Ground truth** | Full-precision cosine top-k (not hand labels) |
| **300 queries** | Section titles, stratified across 28 chapters |
| **Drift report** | `rag_topk_drift.json` — which chunks drop out per method |

---

## Slide 9 — Token results (Part A)

230 tokens · recall@10 · `text-embedding-3-small`

| Method | Overlap@10 | Compression |
|--------|------------|-------------|
| scalar_8bit | 99.7% | 4× |
| turboquant_8bit | 99.7% | 3.4× |
| turboquant_4bit | 97.2% | 5.9× |
| scalar_4bit | 95.8% | 8× |
| **turboquant_2bit** | **87.0%** | **9.3×** |
| scalar_2bit | 82.5% | 16× |
| jl_128 | 59.2% | 2× |

**Figure:** `python/figures/token_compression_frontier.png`

---

## Slide 10 — RAG results (Part B) — headline table

1380 chunks · **300 queries** · overlap@3 vs full precision

| Method | Overlap@3 | Compression | Index size |
|--------|-----------|-------------|------------|
| turboquant_8bit | **99.8%** | 3.5× | 0.41 MB |
| scalar_8bit | 99.7% | 4× | 0.36 MB |
| turboquant_4bit | 93.3% | 6.2× | 0.24 MB |
| scalar_4bit | 93.2% | 8× | 0.18 MB |
| turboquant_3bit | 87.0% | 7.7× | 0.19 MB |
| **turboquant_2bit** | **76.1%** | **10×** | **0.15 MB** |
| scalar_2bit | 62.8% | 16× | 0.09 MB |
| sign_1bit | 47.9% | 28× | 0.06 MB |
| jl_128 | 39.2% | 2× | 0.84 MB |

**Key slide point:** At **aggressive compression**, TurboQuant **beats scalar** — QJL residual pays off when stage-1 quant is harsh.

**Figures:** `python/figures/rag/rag_compression_frontier.png`, `rag_value_ranking.png`, `rag_drift_summary.png`

---

## Slide 11 — Pareto story (what to say)

**Moderate bits (4–8):** scalar ≈ TurboQuant (both ~93–100% overlap). Scalar wins on **pure compression ratio** (fewer metadata bits).

**Aggressive bits (~3–4 effective):** **TurboQuant pulls ahead** — e.g. 2-bit stage-1 + QJL: **76%** vs scalar 2-bit **63%** at similar budget.

**Extreme (sign only):** 28× but only 48% — TurboQuant’s recipe is how you fix that class of error **without** storing full vectors.

**Raw JL:** not on the frontier — wrong tool for top-k here.

---

## Slide 12 — Live demo

```bash
bun run dev    # UI http://localhost:5173 , API :8010
```

1. Ask: *“What does the spectral theorem say about symmetric matrices?”*
2. Compare indexes: `full_precision` · `turboquant_4bit` · `sign_1bit` · `scalar_8bit`
3. Show index sizes in UI storage table

**Notebook:** `python/notebooks/application.ipynb` (Parts A & B)

---

## Slide 13 — References

- Johnson & Lindenstrauss (1984)
- [TurboQuant blog — Google Research (2026)](https://research.google.blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- Wang, [*Book-AdvancedLinearAlgebraAI*](https://github.com/wanghemath/Book-AdvancedLinearAlgebraAI)
- Azure / OpenAI `text-embedding-3-small` ($d=256$)

---

## Speaker notes — 90 sec opening

> High-dimensional vectors are everywhere in AI — embeddings, search indexes, attention keys in the KV cache. Storage scales with dimension times bits per coordinate. We surveyed classical **linear maps** that shrink vectors: Johnson–Lindenstrauss random projections, SVD rank-k truncation, 1-bit sign quantization, and uniform scalar quant. We also implemented a **TurboQuant-style** pipeline from Google Research: rotate, quantize, then apply a 1-bit JL stage on the **residual** with a full-precision query at scoring time.
>
> We tested everything on two tasks: a token nearest-neighbor toy set and a **real RAG index** over the MATH 5110 textbook — 1380 chunks, **300 automatically generated queries**, with full-precision top-3 as ground truth. This isn’t a toy labeled set; it’s a large, reproducible benchmark.
>
> **Honest results:** raw JL does not win on retrieval ranking — distance guarantees aren’t enough. Scalar quantization is embarrassingly strong at 4–8 bits. But when we push compression hard, **TurboQuant beats plain scalar** — at 2-bit stage-1 we get **76%** overlap versus **63%** for scalar 2-bit. That’s exactly why the two-stage design exists.
>
> The live UI lets you search the book with compressed indexes. Same linear algebra Google uses for KV-cache compression — our project is the **understandable, demonstrable version** on course material.

---

## Speaker notes — 60 sec closing

> Takeaway: compression is a **tradeoff curve**, not one winner. Implement the maps, measure retrieval, report when simple baselines win and when residual QJL earns its bits. RAG was the right lab bench for a searchable MATH 5110 project; TurboQuant at LLM scale is the industrial deployment of the same ideas. Questions?

---

## Pre-flight checklist

- [ ] `uv run python scripts/run_all.py` (figures + `presentation_results.json`)
- [ ] `.env` has Azure/OpenAI embedding keys
- [ ] `bun run dev` — UI loads, search returns chunks
- [ ] Figures open: `rag_compression_frontier.png`, `token_compression_frontier.png`
- [ ] Optional: open `rag_topk_drift.json` for one “chunk swap” example
