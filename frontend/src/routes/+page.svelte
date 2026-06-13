<script lang="ts">
  import { onMount } from "svelte";
  import HeroSection from "$lib/components/search/HeroSection.svelte";
  import ResultsPanel from "$lib/components/search/ResultsPanel.svelte";
  import SearchPanel from "$lib/components/search/SearchPanel.svelte";
  import StoragePanel from "$lib/components/search/StoragePanel.svelte";
  import type { CorpusInfo, SearchResponse } from "$lib/search/types.js";
  import { readApiError } from "$lib/search/types.js";

  let corpus = $state<CorpusInfo | null>(null);
  let corpusErr = $state<string | null>(null);
  let query = $state("What is the spectral theorem for symmetric matrices?");
  let method = $state("full_precision");
  let topK = $state(5);
  let compare = $state(false);
  let loading = $state(false);
  let searchErr = $state<string | null>(null);
  let data = $state<SearchResponse | null>(null);

  onMount(async () => {
    try {
      const res = await fetch("/api/corpus");
      if (!res.ok) {
        corpusErr = await readApiError(res);
        return;
      }
      corpus = (await res.json()) as CorpusInfo;
      if (corpus.methods.length > 0 && !corpus.methods.includes(method)) {
        method = corpus.methods[0];
      }
    } catch (e) {
      corpusErr =
        e instanceof Error
          ? `${e.message} — is the API running? (bun run dev from repo root)`
          : String(e);
    }
  });

  async function search() {
    const q = query.trim();
    if (!q) return;

    loading = true;
    searchErr = null;
    data = null;
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          method,
          top_k: Number(topK),
          compare,
        }),
      });
      if (!res.ok) {
        searchErr = await readApiError(res);
        return;
      }
      data = (await res.json()) as SearchResponse;
    } catch (e) {
      searchErr = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }
</script>

<div class="page" aria-busy={loading ? true : undefined}>
  <HeroSection />

  {#if corpusErr}
    <p class="banner error" role="alert">{corpusErr}</p>
  {:else if corpus}
    <StoragePanel
      corpus={corpus.corpus}
      nChunks={corpus.n_chunks}
      dim={corpus.dim}
      storage={corpus.storage}
    />
  {/if}

  <div class="grid">
    <SearchPanel
      bind:query
      bind:method
      bind:topK
      bind:compare
      methods={corpus?.methods ?? ["full_precision"]}
      {loading}
      err={searchErr}
      onSearch={search}
    />
    <ResultsPanel {data} {loading} />
  </div>
</div>

<style>
  .page {
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
  }
  .grid {
    display: grid;
    gap: 1.1rem;
  }
  @media (min-width: 900px) {
    .grid {
      grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.05fr);
      align-items: start;
    }
  }
  .banner {
    margin: 0;
    padding: 0.75rem 0.9rem;
    border-radius: var(--radius-sm);
    font-size: 0.9rem;
  }
  .banner.error {
    background: var(--color-danger-bg);
    color: var(--color-danger);
    border: 1px solid #f0c4c4;
  }
</style>
