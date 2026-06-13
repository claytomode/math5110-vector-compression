<script lang="ts">
  import "katex/dist/katex.min.css";
  import type { SearchHit } from "$lib/search/types.js";
  import { renderChunkMarkdown } from "$lib/search/chunk-markdown.js";

  let { hit }: { hit: SearchHit } = $props();
  let expanded = $state(false);

  const body = $derived(expanded ? hit.text : hit.preview);
  const html = $derived(renderChunkMarkdown(body));
</script>

<article class="hit">
  <header class="hit-head">
    <span class="rank">#{hit.rank}</span>
    <span class="score">{hit.score.toFixed(3)}</span>
    <code class="chunk-id">{hit.chunk_id}</code>
  </header>
  <div class="chunk-body">{@html html}</div>
  {#if hit.text.length > hit.preview.length}
    <button type="button" class="toggle" onclick={() => (expanded = !expanded)}>
      {expanded ? "Show less" : "Show full chunk"}
    </button>
  {/if}
</article>

<style>
  .hit {
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--color-border);
  }
  .hit:last-child {
    border-bottom: none;
    padding-bottom: 0;
  }
  .hit-head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.5rem 0.75rem;
    margin-bottom: 0.45rem;
  }
  .rank {
    font-family: var(--font-serif);
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--color-accent);
  }
  .score {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--color-text-faint);
  }
  .chunk-id {
    font-size: 0.78rem;
    word-break: break-all;
  }
  .chunk-body {
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
  .toggle {
    margin-top: 0.45rem;
    padding: 0;
    border: none;
    background: none;
    color: var(--color-accent);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    text-decoration: underline;
    text-underline-offset: 2px;
  }
</style>
