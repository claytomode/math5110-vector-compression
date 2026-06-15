"""Convert LaTeX math delimiters to GitHub $ / $$ syntax."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOLLAR = chr(36)
DISPLAY = DOLLAR * 2


def convert_math(text: str) -> str:
    text = text.replace("\\[", DISPLAY).replace("\\]", DISPLAY)
    text = text.replace("\\(", DOLLAR).replace("\\)", DOLLAR)
    return text


def fix_table_pipes(text: str) -> str:
    return text.replace(
        "$\\mathbb{E}\\bigl[| \\|\\hat{u}-\\hat{v}\\| - \\|u-v\\| | / \\|u-v\\|\\bigr]$",
        "$\\mathbb{E}\\bigl[\\vert \\|\\hat{u}-\\hat{v}\\| - \\|u-v\\|\\vert / \\|u-v\\|\\bigr]$",
    ).replace(
        "$\\| \\mathrm{TopK}_{\\mathrm{full}} \\cap \\mathrm{TopK}_{\\mathrm{comp}} | / k$",
        "$\\vert \\mathrm{TopK}_{\\mathrm{full}} \\cap \\mathrm{TopK}_{\\mathrm{comp}}\\vert / k$",
    )


def main() -> None:
    rev = "8f3dc42" if "--from-git" in sys.argv else None
    for path in sorted(DOCS.glob("*.md")):
        if rev:
            rel = path.relative_to(ROOT).as_posix()
            raw = subprocess.check_output(
                ["git", "show", f"{rev}:{rel}"],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
            )
        else:
            raw = path.read_text(encoding="utf-8")
        path.write_text(fix_table_pipes(convert_math(raw)), encoding="utf-8")
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
