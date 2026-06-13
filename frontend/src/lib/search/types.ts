export type StorageRow = {
  method: string;
  index_bytes: number;
  index_mb: number;
  compression_ratio: number;
  bits_per_dim: number;
};

export type CorpusInfo = {
  corpus: string;
  n_chunks: number;
  dim: number;
  methods: string[];
  compare_methods: string[];
  storage: StorageRow[];
};

export type SearchHit = {
  rank: number;
  chunk_id: string;
  score: number;
  text: string;
  preview: string;
};

export type MethodResults = {
  method: string;
  hits: SearchHit[];
};

export type SearchResponse = {
  ok: true;
  mode: "single" | "compare";
  corpus: string;
  n_chunks: number;
  dim: number;
  query: string;
  hits?: SearchHit[];
  method?: string;
  comparisons?: MethodResults[];
};

export function formatBytes(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)} MB`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)} KB`;
  return `${n.toLocaleString()} B`;
}

export function methodLabel(method: string): string {
  const labels: Record<string, string> = {
    full_precision: "Full float32",
    sign_1bit: "Sign 1-bit",
  };
  if (labels[method]) return labels[method];
  if (method.startsWith("jl_")) return `JL → ${method.split("_")[1]}d`;
  if (method.startsWith("rank_")) return `Rank-${method.split("_")[1]} SVD`;
  if (method.startsWith("scalar_")) return `Scalar ${method.split("_")[1]}`;
  return method;
}

export async function readApiError(res: Response): Promise<string> {
  try {
    const raw = await res.json();
    if (typeof raw.detail === "string") return raw.detail;
    if (Array.isArray(raw.detail)) {
      return raw.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
    }
    if (typeof raw.error === "string") return raw.error;
  } catch {
    /* ignore */
  }
  return `Request failed (HTTP ${res.status} ${res.statusText})`;
}
