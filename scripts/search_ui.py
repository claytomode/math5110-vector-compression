#!/usr/bin/env python3
"""Launch the Svelte + FastAPI search UI (replaces Gradio)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bun = shutil.which("bun")
    if not bun:
        print("Install Bun (https://bun.sh), then from the repo root run:")
        print("  bun install")
        print("  cd frontend && bun install")
        print("  bun run dev")
        sys.exit(1)

    pkg = ROOT / "package.json"
    if not pkg.exists():
        print(f"Missing {pkg}")
        sys.exit(1)

    print("Starting vector search UI (API :8010, frontend :5173)...")
    print("Press Ctrl+C to stop.\n")
    subprocess.run([bun, "run", "dev"], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
