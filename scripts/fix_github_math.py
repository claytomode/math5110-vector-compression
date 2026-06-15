"""Convert LaTeX math delimiters to GitHub $ / ```math syntax."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOLLAR = chr(36)
DISPLAY = DOLLAR * 2

_DISPLAY_BLOCK = re.compile(r"\n\$\$\n(.*?)\n\$\$(?=\n|$)", re.DOTALL)


def convert_math(text: str) -> str:
    text = text.replace("\\[", DISPLAY).replace("\\]", DISPLAY)
    text = text.replace("\\(", DOLLAR).replace("\\)", DOLLAR)
    return text


def display_to_math_blocks(text: str) -> str:
    """GitHub renders ```math blocks more reliably than bare $$ delimiters."""

    def repl(match: re.Match[str]) -> str:
        body = match.group(1).strip("\n")
        return f"\n\n```math\n{body}\n```\n"

    return _DISPLAY_BLOCK.sub(repl, text)


def fix_table_pipes(text: str) -> str:
    return text.replace(
        "$\\mathbb{E}\\bigl[| \\|\\hat{u}-\\hat{v}\\| - \\|u-v\\| | / \\|u-v\\|\\bigr]$",
        "$\\mathbb{E}\\bigl[\\vert \\|\\hat{u}-\\hat{v}\\| - \\|u-v\\|\\vert / \\|u-v\\|\\bigr]$",
    ).replace(
        "$\\| \\mathrm{TopK}_{\\mathrm{full}} \\cap \\mathrm{TopK}_{\\mathrm{comp}} | / k$",
        "$\\vert \\mathrm{TopK}_{\\mathrm{full}} \\cap \\mathrm{TopK}_{\\mathrm{comp}}\\vert / k$",
    )


def process(text: str) -> str:
    return display_to_math_blocks(fix_table_pipes(convert_math(text)))


def main() -> None:
    blocks_only = "--blocks-only" in sys.argv
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
        if blocks_only:
            out = display_to_math_blocks(raw)
        else:
            out = process(raw)
        path.write_text(out, encoding="utf-8")
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
