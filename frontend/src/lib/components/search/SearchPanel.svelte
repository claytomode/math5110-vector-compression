<script lang="ts">
  import { methodLabel } from "$lib/search/types.js";

  let {
    query = $bindable(""),
    method = $bindable("full_precision"),
    topK = $bindable(5),
    compare = $bindable(false),
    methods,
    loading,
    err,
    onSearch,
  }: {
    query: string;
    method: string;
    topK: number;
    compare: boolean;
    methods: string[];
    loading: boolean;
    err: string | null;
    onSearch: () => void;
  } = $props();

  const examples = [
    "What is the spectral theorem for symmetric matrices?",
    "What is the singular value decomposition?",
    "Explain the Perron-Frobenius theorem",
  ];
</script>

<section class="panel">
  <h2 class="panel-title">Search</h2>
  <label for="query">Question</label>
  <textarea
    id="query"
    bind:value={query}
    rows="3"
    placeholder="What is the spectral theorem for symmetric matrices?"
    onkeydown={(e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        onSearch();
      }
    }}
  ></textarea>

  <div class="chips" aria-label="Example queries">
    {#each examples as ex (ex)}
      <button type="button" class="chip" onclick={() => (query = ex)}>{ex}</button>
    {/each}
  </div>

  <div class="controls">
    <div class="field">
      <label for="method">Index type</label>
      <select id="method" bind:value={method} disabled={compare}>
        {#each methods as m (m)}
          <option value={m}>{methodLabel(m)}</option>
        {/each}
      </select>
    </div>
    <div class="field field-narrow">
      <label for="topk">Top K</label>
      <input id="topk" type="number" min="1" max="20" bind:value={topK} />
    </div>
    <label class="check-row">
      <input type="checkbox" bind:checked={compare} />
      Compare indexes
    </label>
  </div>

  <div class="row">
    <button
      type="button"
      class="btn-search"
      onclick={onSearch}
      disabled={loading || !query.trim()}
      aria-busy={loading ? true : undefined}
    >
      <span class="btn-left" aria-hidden="true">
        {#if loading}
          <span class="spinner"></span>
        {:else}
          <span class="btn-slot"></span>
        {/if}
      </span>
      <span class="btn-label">Search</span>
      <span class="btn-right" aria-hidden="true"><span class="btn-slot"></span></span>
    </button>
    <span class="hint muted">Ctrl+Enter to search</span>
  </div>

  {#if err}
    <p class="error" role="alert">{err}</p>
  {/if}
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
    margin: 0 0 0.85rem;
    color: var(--color-text);
  }
  label {
    display: block;
    margin: 0 0 0.4rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--color-text-muted);
  }
  textarea {
    width: 100%;
    box-sizing: border-box;
    font-family: var(--font-sans);
    font-size: 0.95rem;
    line-height: 1.5;
    background: var(--color-surface);
    color: var(--color-text);
    border: 1px solid var(--color-border-strong);
    border-radius: var(--radius-sm);
    padding: 0.75rem 0.85rem;
    resize: vertical;
    box-shadow: var(--shadow-inset);
  }
  textarea:focus {
    outline: 2px solid color-mix(in srgb, var(--color-accent) 45%, transparent);
    outline-offset: 1px;
    border-color: var(--color-accent);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.65rem;
  }
  .chip {
    font-family: var(--font-sans);
    font-size: 0.78rem;
    padding: 0.3rem 0.55rem;
    border-radius: 999px;
    border: 1px solid var(--color-border-strong);
    background: var(--color-surface);
    color: var(--color-text-muted);
    cursor: pointer;
  }
  .chip:hover {
    background: var(--color-bg-deep);
    color: var(--color-text);
  }
  .controls {
    display: grid;
    gap: 0.75rem;
    margin-top: 0.85rem;
  }
  @media (min-width: 640px) {
    .controls {
      grid-template-columns: 1fr auto auto;
      align-items: end;
    }
  }
  select,
  input[type="number"] {
    font-family: var(--font-sans);
    font-size: 0.88rem;
    padding: 0.4rem 0.5rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border-strong);
    background: var(--color-surface-raised);
    color: var(--color-text);
    width: 100%;
    max-width: 16rem;
  }
  .field-narrow input {
    max-width: 4.5rem;
  }
  .check-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.86rem;
    color: var(--color-text);
    font-weight: 500;
    margin: 0;
    padding-bottom: 0.35rem;
  }
  .row {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    margin-top: 0.85rem;
    flex-wrap: wrap;
  }
  .btn-search {
    display: inline-grid;
    grid-template-columns: 1.15rem auto 1.15rem;
    align-items: center;
    justify-items: center;
    column-gap: 0.35rem;
    font-family: var(--font-sans);
    background: var(--color-accent);
    color: #fdfcfa;
    border: 1px solid color-mix(in srgb, var(--color-accent) 88%, #000);
    border-radius: var(--radius-sm);
    padding: 0.55rem 1.05rem;
    font-weight: 600;
    font-size: 0.92rem;
    cursor: pointer;
  }
  .btn-search:hover:not(:disabled) {
    background: var(--color-accent-hover);
  }
  .btn-search:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .btn-left,
  .btn-right {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 1.15rem;
    height: 1.15rem;
  }
  .btn-slot {
    display: block;
    width: 1.05rem;
    height: 1.05rem;
  }
  .spinner {
    width: 1.05rem;
    height: 1.05rem;
    border: 2px solid rgba(255, 255, 255, 0.35);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .hint {
    font-size: 0.82rem;
  }
  .muted {
    color: var(--color-text-muted);
  }
  .error {
    margin: 0.75rem 0 0;
    padding: 0.65rem 0.75rem;
    border-radius: var(--radius-sm);
    background: var(--color-danger-bg);
    color: var(--color-danger);
    border: 1px solid #f0c4c4;
    font-size: 0.9rem;
  }
</style>
