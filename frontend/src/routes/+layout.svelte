<script lang="ts">
  import "../app.css";

  let { children } = $props();
</script>

<svelte:head>
  <title>Vector Search — MATH 5110</title>
  <meta
    name="description"
    content="Search course material with full-precision and compressed embedding indexes."
  />
</svelte:head>

<div class="shell">
  <nav class="site-nav" aria-label="Primary">
    <span class="brand">Vector Search</span>
    <span class="tag">MATH 5110</span>
  </nav>
  {@render children()}
</div>

<style>
  .shell {
    max-width: 1120px;
    margin: 0 auto;
    padding: clamp(1.25rem, 4vw, 2.25rem) clamp(1rem, 3vw, 1.5rem) 3rem;
  }
  .site-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
  }
  .brand {
    font-family: var(--font-serif);
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--color-text);
  }
  .tag {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--color-text-muted);
  }
</style>
