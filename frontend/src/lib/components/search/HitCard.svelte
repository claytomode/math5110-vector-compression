<script lang="ts">
  import "katex/dist/katex.min.css";
  import type { SearchHit } from "$lib/search/types.js";
  import { renderChunkMarkdown } from "$lib/search/chunk-markdown.js";
  import { chunkTitle } from "$lib/search/types.js";

  let { hit, open = false }: { hit: SearchHit; open?: boolean } = $props();
  let expanded = $state(false);

  $effect(() => {
    expanded = open;
  });

  const title = $derived(chunkTitle(hit.chunk_id));
  const html = $derived(expanded ? renderChunkMarkdown(hit.text) : "");
</script>

<article class="hit" class:expanded>
  <button
    type="button"
    class="hit-head"
    aria-expanded={expanded}
    onclick={() => (expanded = !expanded)}
  >
    <span class="chevron" aria-hidden="true">{expanded ? "▾" : "▸"}</span>
    <span class="rank">#{hit.rank}</span>
    <span class="title">{title}</span>
    <span class="score">{hit.score.toFixed(3)}</span>
  </button>
  {#if expanded}
    <code class="chunk-id">{hit.chunk_id}</code>
    <div class="chunk-body">{@html html}</div>
  {/if}
</article>

<style>
  .hit {
    border-bottom: 1px solid var(--color-border);
  }
  .hit:last-child {
    border-bottom: none;
  }
  .hit-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    width: 100%;
    padding: 0.6rem 0;
    border: none;
    background: none;
    text-align: left;
    cursor: pointer;
    color: inherit;
    font: inherit;
  }
  .hit-head:hover .title {
    color: var(--color-accent);
  }
  .chevron {
    font-size: 0.72rem;
    color: var(--color-text-faint);
    width: 0.8rem;
    flex-shrink: 0;
  }
  .rank {
    font-family: var(--font-serif);
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--color-accent);
    flex-shrink: 0;
  }
  .title {
    flex: 1;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--color-text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: color 0.12s ease;
  }
  .score {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--color-text-faint);
    flex-shrink: 0;
  }
  .chunk-id {
    display: block;
    font-size: 0.74rem;
    color: var(--color-text-faint);
    word-break: break-all;
    margin: 0 0 0.5rem 1.4rem;
  }
  .chunk-body {
    margin: 0 0 0.7rem 1.4rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--color-text);
    overflow-x: auto;
  }
  .chunk-body :global(h2),
  .chunk-body :global(h3) {
    font-family: var(--font-serif);
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.45rem;
    line-height: 1.3;
  }
  .chunk-body :global(p) {
    margin: 0 0 0.55rem;
  }
  .chunk-body :global(p:last-child) {
    margin-bottom: 0;
  }
  .chunk-body :global(strong) {
    font-weight: 600;
    color: var(--color-text);
  }
  .chunk-body :global(.katex-display) {
    margin: 0.65rem 0;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0.15rem 0;
  }
  .chunk-body :global(.katex) {
    font-size: 1.02em;
  }
</style>
