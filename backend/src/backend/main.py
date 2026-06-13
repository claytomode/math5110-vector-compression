"""ASGI entrypoint for `uvicorn backend.main:app`."""

from __future__ import annotations

from backend.api.app import app

__all__ = ["app"]
