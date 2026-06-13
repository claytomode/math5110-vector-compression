<script lang="ts">
  import type { StorageRow } from "$lib/search/types.js";
  import { formatBytes, methodLabel } from "$lib/search/types.js";

  let {
    corpus,
    nChunks,
    dim,
    storage,
  }: {
    corpus: string;
    nChunks: number;
    dim: number;
    storage: StorageRow[];
  } = $props();
</script>

<section class="panel">
  <h2 class="panel-title">Index storage</h2>
  <p class="meta">
    Corpus <code>{corpus}</code> · {nChunks.toLocaleString()} chunks · d={dim}
  </p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th scope="col">Method</th>
          <th scope="col">Size</th>
          <th scope="col">vs float32</th>
          <th scope="col">bits/dim</th>
        </tr>
      </thead>
      <tbody>
        {#each storage as row (row.method)}
          <tr>
            <td>
              <span class="method-name">{methodLabel(row.method)}</span>
              <span class="method-code">{row.method}</span>
            </td>
            <td>{formatBytes(row.index_bytes)}</td>
            <td>{row.compression_ratio}×</td>
            <td>{row.bits_per_dim}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <p class="footnote">
    Approximate in-memory index size (vectors + small metadata such as JL matrix or rank-k basis).
  </p>
</section>

<style>
  .panel {
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    padding: 1.1rem 1.15rem 1.2rem;
    box-shadow: var(--shadow);
  }
  .panel-title {
    font-family: var(--font-serif);
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0 0 0.65rem;
    color: var(--color-text);
  }
  .meta {
    margin: 0 0 0.85rem;
    font-size: 0.9rem;
    color: var(--color-text-muted);
  }
  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  th,
  td {
    padding: 0.55rem 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--color-border);
  }
  th {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--color-text-muted);
    background: var(--color-bg-deep);
  }
  tr:last-child td {
    border-bottom: none;
  }
  .method-name {
    display: block;
    font-weight: 600;
    color: var(--color-text);
  }
  .method-code {
    display: block;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--color-text-faint);
    margin-top: 0.1rem;
  }
  .footnote {
    margin: 0.75rem 0 0;
    font-size: 0.82rem;
    color: var(--color-text-faint);
    line-height: 1.45;
  }
</style>
