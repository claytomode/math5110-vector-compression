<script lang="ts">
  import HitCard from "$lib/components/search/HitCard.svelte";
  import type { SearchResponse } from "$lib/search/types.js";
  import { methodLabel } from "$lib/search/types.js";

  let {
    data,
    loading,
  }: {
    data: SearchResponse | null;
    loading: boolean;
  } = $props();

  let allOpen = $state(false);
</script>

<section class="panel">
  <div class="panel-head">
    <h2 class="panel-title">Results</h2>
    {#if data && !loading}
      <button type="button" class="expand-all" onclick={() => (allOpen = !allOpen)}>
        {allOpen ? "Collapse all" : "Expand all"}
      </button>
    {/if}
  </div>

  {#if loading}
    <div class="skeleton" aria-hidden="true">
      <div class="sk-line wide"></div>
      <div class="sk-line"></div>
      <div class="sk-line"></div>
      <div class="sk-block"></div>
    </div>
    <p class="loading-note">Embedding query and scanning the index…</p>
  {:else if !data}
    <p class="empty">
      Run a search to see top matching chunks. Try comparing compressed indexes to see how retrieval
      shifts at smaller footprints.
    </p>
  {:else if data.mode === "compare" && data.comparisons}
    <p class="meta">
      Compared {data.comparisons.length} indexes · {data.n_chunks.toLocaleString()} chunks · d={data.dim}
    </p>
    <div class="compare-grid">
      {#each data.comparisons as block (block.method)}
        <div class="compare-col">
          <h3>{methodLabel(block.method)}</h3>
          <p class="method-code">{block.method}</p>
          {#if block.hits.length === 0}
            <p class="empty small">No hits.</p>
          {:else}
            {#each block.hits as hit (block.method + "-" + hit.rank)}
              <HitCard {hit} open={allOpen} />
            {/each}
          {/if}
        </div>
      {/each}
    </div>
  {:else if data.hits}
    <p class="meta">
      Index <code>{data.method}</code> · {data.n_chunks.toLocaleString()} chunks · d={data.dim}
    </p>
    {#if data.hits.length === 0}
      <p class="empty">No hits.</p>
    {:else}
      {#each data.hits as hit (hit.rank)}
        <HitCard {hit} open={allOpen} />
      {/each}
    {/if}
  {/if}
</section>

<style>
  .panel {
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    border-radius: var(--radius);
    padding: 1.1rem 1.15rem 1.2rem;
    box-shadow: var(--shadow);
    min-height: 12rem;
  }
  .panel-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0 0 0.85rem;
  }
  .panel-title {
    font-family: var(--font-serif);
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0;
    color: var(--color-text);
  }
  .expand-all {
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    color: var(--color-text-muted);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: var(--radius-sm);
    cursor: pointer;
  }
  .expand-all:hover {
    color: var(--color-accent);
    border-color: var(--color-accent);
  }
  .meta {
    margin: 0 0 0.75rem;
    font-size: 0.88rem;
    color: var(--color-text-muted);
  }
  .empty {
    margin: 0;
    color: var(--color-text-muted);
    font-size: 0.92rem;
    line-height: 1.55;
  }
  .empty.small {
    font-size: 0.85rem;
  }
  .loading-note {
    margin: 0.65rem 0 0;
    font-size: 0.85rem;
    color: var(--color-text-faint);
  }
  .compare-grid {
    display: grid;
    gap: 1rem;
  }
  @media (min-width: 900px) {
    .compare-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  .compare-col {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: 0.75rem 0.85rem;
    background: var(--color-surface);
  }
  .compare-col h3 {
    margin: 0;
    font-family: var(--font-serif);
    font-size: 1rem;
    font-weight: 600;
  }
  .method-code {
    margin: 0.15rem 0 0.65rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--color-text-faint);
  }
  .skeleton {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
  }
  .sk-line {
    height: 0.75rem;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--color-bg-deep), var(--color-border), var(--color-bg-deep));
    background-size: 200% 100%;
    animation: shimmer 1.2s ease-in-out infinite;
    width: 72%;
  }
  .sk-line.wide {
    width: 92%;
  }
  .sk-block {
    height: 5rem;
    border-radius: var(--radius-sm);
    background: linear-gradient(90deg, var(--color-bg-deep), var(--color-border), var(--color-bg-deep));
    background-size: 200% 100%;
    animation: shimmer 1.2s ease-in-out infinite;
    margin-top: 0.35rem;
  }
  @keyframes shimmer {
    0% {
      background-position: 100% 0;
    }
    100% {
      background-position: -100% 0;
    }
  }
</style>
