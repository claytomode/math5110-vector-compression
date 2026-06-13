# Part 1 — Survey: compressing high-dimensional vectors

> Theory narrative for the course report. Application (token embeddings) is in `python/notebooks/application.ipynb`.

## Why this topic?

Modern AI stores billions of **high-dimensional vectors**: token embeddings, attention keys/values, retrieval index entries. Memory and bandwidth scale with dimension \(d\). **Vector quantization** and **random projections** are classical linear-algebra tools to shrink storage while preserving geometry (distances, inner products).

Recent systems work such as [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) combines:

1. **Structured compression** (polar / spectral geometry),
2. **1-bit Johnson–Lindenstrauss-style** residuals,
3. **Theoretical guarantees** on distortion.

This project surveys those ideas and applies simplified versions to **token embedding vectors** (GloVe).

---

## 1. Johnson–Lindenstrauss (JL)

**Lemma (informal):** For \(n\) points in \(\mathbb{R}^d\), there exists a linear map \(R \in \mathbb{R}^{k \times d}\) with \(k = O(\varepsilon^{-2} \log n)\) such that pairwise distances are preserved up to \((1 \pm \varepsilon)\).

**Construction used here:** Gaussian random \(R\) with entries \(\mathcal{N}(0, 1/k)\). Sketch \(y = Rx\).

**Use case:** Dimensionality reduction before storage or search; foundation of **Quantized JL (QJL)** in TurboQuant’s residual stage.

**Linear algebra:** Orthogonal / random projections, singular values of \(R\), concentration inequalities (survey level — no proof required in code).

---

## 2. Spectral / rank‑\(k\) truncation (PolarQuant analogy)

Write a matrix of embedding vectors \(X \in \mathbb{R}^{n \times d}\). SVD:

\[
X \approx U_k \Sigma_k V_k^\top
\]

Keep top‑\(k\) right singular vectors \(V_k\): each row is reconstructed in a **\(k\)-dimensional subspace**.

**Polar intuition:** a vector splits into **magnitude** (radius) and **direction** (angles). Rank‑\(k\) methods capture dominant directions shared across tokens — similar in spirit to PolarQuant’s polar coordinate grid, without implementing their full pipeline.

---

## 3. Sign quantization (1-bit, QJL-style)

Store only \(\mathrm{sign}(x_i) \in \{+1,-1\}\) per coordinate. For query \(q\) (full precision), score keys with:

\[
\hat{q}^\top k \approx \|k\| \cdot \frac{\mathrm{sign}(k)^\top q}{\sqrt{d}}
\]

**Use case:** Extreme compression for inner-product / attention-style scoring. TurboQuant uses a 1-bit JL stage on residuals.

---

## 4. Scalar quantization

Uniform bins per coordinate: map \(x_j \in [a_j, b_j]\) to \(2^b\) levels. Simple baseline; introduces quantization error but gives direct **bits-per-dimension** control.

---

## 5. What we measure (Part 2–3 bridge)

| Metric | Meaning |
|--------|---------|
| **Recall@k** | Fraction of true nearest neighbors retained under compressed scoring |
| **Distance distortion** | Mean relative error on random token pairs |
| **Bits per dimension** | Storage budget (approximate, amortized for shared bases) |

These mirror vector-search benchmarks (1@k recall) discussed in the TurboQuant blog, at classroom scale.

---

## 6. Connection to tokens / LLMs (conceptual)

- **Token embedding table:** each token type \(\mapsto\) vector in \(\mathbb{R}^d\).
- **KV cache:** sequence positions store key vectors; compression reduces memory during long-context inference.
- **Our application:** GloVe word vectors as a stand-in embedding table; nearest-neighbor recall simulates “find similar tokens / entries under compression.”

We do **not** run transformer inference; the point is the **shared linear algebra**.

---

## References

- Pennington, Socher, Manning (2014). [GloVe](https://nlp.stanford.edu/projects/glove/).
- Johnson & Lindenstrauss (1984). Extensions of Lipschitz maps into Hilbert space.
- Zandieh, Mirrokni et al. (2026). [TurboQuant blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/).
