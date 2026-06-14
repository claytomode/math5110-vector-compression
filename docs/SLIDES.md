# MATH 5110 — Vector compression for embeddings & retrieval

Slide deck outline (copy into PowerPoint / Google Slides / Quarto reveal).
Figures live in `python/figures/` after `uv run python scripts/run_all.py`.

---

## Slide 1 — Title

**Vector compression for high-dimensional data**

MATH 5110 applied linear algebra project

- Survey: JL, SVD rank‑k, sign & scalar quantization
- Application: compressed retrieval over the course textbook

---

## Slide 2 — Why vectors are expensive

Modern AI stores **billions** of vectors:

- Token embedding tables
- Attention **keys** in the KV cache
- RAG / search **indexes**

Cost scales with dimension \(d\) and count \(n\): storage \(\propto n \cdot d \cdot \text{bits per dim}\).

**Motivation:** [TurboQuant — Google Research (2026)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

---

## Slide 3 — Project structure

| Part | What we did |
|------|-------------|
| **1. Survey** | JL, spectral truncation, 1-bit signs, scalar quant |
| **2. Computation** | NumPy + recall@k + distance distortion |
| **3. Application** | Token embeddings + **book RAG** with compressed indexes |

Repo: `claytomode/math5110-vector-compression`

---

## Slide 4 — Johnson–Lindenstrauss (informal lemma)

For \(n\) points in \(\mathbb{R}^d\), there is a linear map \(R \in \mathbb{R}^{k \times d}\) with

\[
k = O\!\left(\frac{\log n}{\varepsilon^2}\right)
\]

that preserves **all pairwise distances** up to \((1 \pm \varepsilon)\).

**Construction we use:** Gaussian \(R_{ij} \sim \mathcal{N}(0, 1/k)\), sketch \(y = Rx\).

**Lin alg:** random projections, approximate isometries, concentration.

---

## Slide 5 — JL in code & in TurboQuant

**Our repo:** `compress_jl` — store sketched keys, project queries with same \(R\).

**TurboQuant / QJL (2026):** after structured compression, apply a **1-bit JL stage** on the residual — signs only, zero metadata overhead, full-precision query at scoring time.

Same linear-algebra story: **shrink vectors, preserve inner products / neighbors.**

---

## Slide 6 — Rank‑\(k\) / SVD truncation

Matrix of embeddings \(X \in \mathbb{R}^{n \times d}\):

\[
X \approx U_k \Sigma_k V_k^\top
\]

Keep top‑\(k\) right singular vectors → each row lives in a **\(k\)-dimensional subspace**.

**Polar intuition (TurboQuant’s PolarQuant):** radius + direction; shared basis across tokens/chunks.

**Our method:** `compress_rank_k` — SVD basis + coefficients.

---

## Slide 7 — Sign & scalar quantization

**Sign (1-bit):** store \(\mathrm{sign}(x_i)\); score with full-precision query + norm correction.

**Scalar:** uniform bins per coordinate (2 / 4 / 8 bits).

Extreme compression for **dot-product retrieval** — same scoring pattern as QJL.

---

## Slide 8 — What we measure

| Metric | Meaning |
|--------|---------|
| **Recall@k** | True nearest neighbors still in top‑\(k\) after compression |
| **Distance distortion** | Mean relative error on random pairs |
| **RAG hit@k** | Gold book chunk in top‑\(k\) for labeled questions |
| **bits/dim** | Storage budget |

**Figure:** `python/figures/recall_vs_bits.png`

---

## Slide 9 — Token study (Part A)

230 word embeddings, \(d=256\), `text-embedding-3-small`

- Toy nearest-neighbor universe (king/queen, matrix/vector, …)
- Scalar 8-bit ≈ strong recall at 4× compression

**Figures:** `token_pca.png`, `distance_error.png`

---

## Slide 10 — Book RAG (Part B)

Index **[Advanced Linear Algebra AI](https://github.com/wanghemath/Book-AdvancedLinearAlgebraAI)** (~1380 chunks)

1. Fetch `.qmd` chapters → chunk by `##` sections
2. Embed chunks → build compressed indexes
3. Evaluate **hit@3** on 12 labeled queries (`rag_queries.yaml`) — e.g. ~67% for full / sign / scalar on our run

**Figures:** `python/figures/rag/rag_hit_vs_bits.png`, `token_vs_rag.png`

---

## Slide 11 — Index sizes (demo)

| Method | ~size (1380 × 256) |
|--------|---------------------|
| full_precision | ~1.4 MB |
| jl_128 | ~700 KB |
| sign_1bit | ~48 KB |

**Live UI:** `bun run dev` → http://localhost:5173 — search + compare indexes.

---

## Slide 12 — Demo path

1. **Notebook** `application.ipynb` — Part A token plots, Part B RAG table
2. **Web UI** — ask “spectral theorem for symmetric matrices”, compare `full_precision` vs `sign_1bit`
3. **Takeaway:** compression trades bits for recall — sign/scalar often beat JL on this corpus

---

## Slide 13 — References

- Johnson & Lindenstrauss (1984)
- [TurboQuant blog — Google Research (2026)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- Wang, *Book-AdvancedLinearAlgebraAI* (MATH 5110 Quarto source)
- OpenAI embeddings API (`text-embedding-3-small`)

---

## Speaker notes (1 min closing)

> We implemented classical **linear maps** that shrink embedding indexes and measured whether **geometry survives** — neighbors, distances, and textbook retrieval. TurboQuant shows the same ideas at LLM scale (KV cache + vector search). Our project is the **lin-alg lab version**: JL sketch, SVD subspace, 1-bit signs, on real course material you can search in the browser.
