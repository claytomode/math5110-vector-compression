"""Fetch and cache token embedding vectors (OpenAI or Azure OpenAI)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from openai import AzureOpenAI, OpenAI

from vector_linalg.config import ProjectConfig


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from repo .env if present (not committed)."""
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if (parent / "pyproject.toml").exists():
            if not env_path.exists():
                return
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return


def _embedding_client(cfg: ProjectConfig) -> tuple[OpenAI | AzureOpenAI, str]:
    """Return (client, model_or_deployment_name)."""
    _load_dotenv()
    emb = cfg.embeddings

    if emb.provider == "azure":
        key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        if not key or not endpoint:
            raise RuntimeError(
                "Azure embeddings require .env entries:\n"
                "  AZURE_OPENAI_API_KEY=...\n"
                "  AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/\n"
                "Optional: AZURE_OPENAI_API_VERSION=2024-02-01\n"
                "Set embeddings.azure_deployment in config.yaml to your deployment name."
            )
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", emb.azure_api_version)
        deployment = (
            os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()
            or emb.azure_deployment
            or emb.model
        )
        endpoint = endpoint.rstrip("/")
        # AI Foundry "OpenAI v1 compatible" URL vs classic Azure OpenAI SDK base
        if endpoint.endswith("/openai/v1"):
            client = OpenAI(api_key=key, base_url=endpoint + "/")
            return client, deployment
        client = AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint + "/",
            api_version=api_version,
        )
        return client, deployment

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add to .env or use embeddings.provider: azure"
        )
    return OpenAI(api_key=key), emb.model


def embed_texts(texts: list[str], cfg: ProjectConfig) -> np.ndarray:
    """Embed arbitrary strings via configured OpenAI or Azure deployment."""
    client, model = _embedding_client(cfg)
    emb_cfg = cfg.embeddings
    vectors: list[np.ndarray] = []

    for start in range(0, len(texts), emb_cfg.batch_size):
        batch = texts[start : start + emb_cfg.batch_size]
        kwargs: dict = {"input": batch, "model": model}
        if emb_cfg.dimensions is not None:
            kwargs["dimensions"] = emb_cfg.dimensions
        try:
            response = client.embeddings.create(**kwargs)
        except Exception as exc:
            if "404" in str(exc):
                raise RuntimeError(
                    "Azure embeddings 404: set AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env "
                    "to the exact deployment name from Azure AI Foundry (not the model SKU)."
                ) from exc
            raise
        ordered = sorted(response.data, key=lambda row: row.index)
        vectors.extend(np.asarray(row.embedding, dtype=np.float64) for row in ordered)
        print(f"  embedded {min(start + len(batch), len(texts))}/{len(texts)}")

    return np.vstack(vectors)


def _fetch_embeddings(cfg: ProjectConfig) -> tuple[list[str], np.ndarray]:
    matrix = embed_texts(list(cfg.tokens), cfg)
    return list(cfg.tokens), matrix


def _to_dataframe(tokens: list[str], matrix: np.ndarray) -> pl.DataFrame:
    dim = matrix.shape[1]
    rows = [
        {"token": t, **{f"d{i}": float(matrix[j, i]) for i in range(dim)}}
        for j, t in enumerate(tokens)
    ]
    return pl.DataFrame(rows)


def fetch_token_embeddings(cfg: ProjectConfig, *, refresh: bool = False) -> pl.DataFrame:
    if cfg.embeddings_cache.exists() and not refresh:
        return pl.read_parquet(cfg.embeddings_cache)

    emb = cfg.embeddings
    label = f"Azure OpenAI ({emb.azure_deployment or emb.model})" if emb.provider == "azure" else f"OpenAI ({emb.model})"
    print(f"Fetching embeddings via {label}...")
    tokens, matrix = _fetch_embeddings(cfg)
    df = _to_dataframe(tokens, matrix)

    cfg.embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cfg.embeddings_cache)

    meta = {
        "source": "Azure OpenAI" if emb.provider == "azure" else "OpenAI Embeddings API",
        "provider": emb.provider,
        "model": emb.model,
        "deployment": emb.azure_deployment if emb.provider == "azure" else None,
        "dimensions": matrix.shape[1],
        "n_tokens": len(tokens),
        "citation": "https://learn.microsoft.com/azure/ai-services/openai/how-to/embeddings"
        if emb.provider == "azure"
        else "https://platform.openai.com/docs/guides/embeddings",
        "file": cfg.rel_path(cfg.embeddings_cache),
    }
    cfg.metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


def embedding_dim(df: pl.DataFrame) -> int:
    return len([c for c in df.columns if c.startswith("d")])


def embedding_matrix(df: pl.DataFrame) -> tuple[list[str], np.ndarray]:
    dim = embedding_dim(df)
    tokens = df["token"].to_list()
    cols = [f"d{i}" for i in range(dim)]
    matrix = df.select(cols).to_numpy()
    return tokens, matrix
