"""Download course content from Canvas (pages + PDFs when allowed)."""

from __future__ import annotations

import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import httpx

from vector_linalg.embeddings import _load_dotenv
from vector_linalg.pdf_corpus import TextChunk, build_pdf_chunks, chunk_text


def _canvas_settings() -> tuple[str, str]:
    _load_dotenv()
    base = os.environ.get("CANVAS_BASE_URL", "").strip().rstrip("/")
    token = os.environ.get("CANVAS_API_TOKEN", "").strip()
    if not base or not token:
        raise RuntimeError(
            "Canvas requires .env entries:\n"
            "  CANVAS_BASE_URL=https://yourschool.instructure.com\n"
            "  CANVAS_API_TOKEN=...\n"
            "  CANVAS_COURSE_ID=12345"
        )
    return base, token


def _course_id(cfg_course_id: int | None) -> str:
    _load_dotenv()
    raw = os.environ.get("CANVAS_COURSE_ID", "").strip() or (
        str(cfg_course_id) if cfg_course_id is not None else ""
    )
    if not raw:
        raise RuntimeError("Set CANVAS_COURSE_ID in .env or rag.canvas.course_id in config.yaml")
    return raw


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            m = re.search(r"<([^>]+)>", part)
            if m:
                return m.group(1)
    return None


def _paginate(client: httpx.Client, url: str, headers: dict[str, str]) -> list[dict]:
    items: list[dict] = []
    while url:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        batch = resp.json()
        if not isinstance(batch, list):
            break
        items.extend(batch)
        url = _next_link(resp.headers.get("Link"))
    return items


def _is_pdf(file_meta: dict) -> bool:
    name = str(file_meta.get("display_name", "")).lower()
    ctype = str(file_meta.get("content-type", "")).lower()
    return name.endswith(".pdf") or "pdf" in ctype


def list_course_files_via_folders(base_url: str, token: str, course_id: str) -> list[dict]:
    """
    List course files by walking the folder tree.

    Northeastern (and some schools) return 403 on /courses/:id/files for students,
    but allow /courses/:id/folders and /folders/:id/files.
    """
    headers = _auth_headers(token)
    seen_folders: set[int] = set()
    seen_files: dict[int, dict] = {}

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        root_folders = _paginate(
            client,
            f"{base_url}/api/v1/courses/{course_id}/folders?per_page=100",
            headers,
        )
        queue = [int(f["id"]) for f in root_folders if "id" in f]

        while queue:
            folder_id = queue.pop(0)
            if folder_id in seen_folders:
                continue
            seen_folders.add(folder_id)

            try:
                for file_meta in _paginate(
                    client,
                    f"{base_url}/api/v1/folders/{folder_id}/files?per_page=100",
                    headers,
                ):
                    fid = int(file_meta["id"])
                    seen_files[fid] = file_meta
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 403:
                    raise

            for sub in _paginate(
                client,
                f"{base_url}/api/v1/folders/{folder_id}/folders?per_page=100",
                headers,
            ):
                sid = int(sub["id"])
                if sid not in seen_folders:
                    queue.append(sid)

    return list(seen_files.values())


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLText()
    parser.feed(html)
    return parser.text()


def list_module_pages(base_url: str, token: str, course_id: str) -> list[dict]:
    headers = _auth_headers(token)
    pages: list[dict] = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        mods = _paginate(
            client,
            f"{base_url}/api/v1/courses/{course_id}/modules?per_page=100",
            headers,
        )
        for mod in mods:
            items = _paginate(
                client,
                f"{base_url}/api/v1/courses/{course_id}/modules/{mod['id']}/items?per_page=100",
                headers,
            )
            for it in items:
                if it.get("type") == "Page" and it.get("page_url"):
                    pages.append(
                        {
                            "module": mod.get("name"),
                            "title": it.get("title") or it.get("page_url"),
                            "page_url": it["page_url"],
                        }
                    )
    return pages


def fetch_page_text(
    base_url: str,
    token: str,
    course_id: str,
    page_url: str,
) -> str:
    headers = _auth_headers(token)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(
            f"{base_url}/api/v1/courses/{course_id}/pages/{page_url}",
            headers=headers,
        )
        resp.raise_for_status()
        body = resp.json().get("body", "") or ""
    return _html_to_text(body)


def build_canvas_chunks(
    *,
    course_id: int | None,
    chunk_chars: int,
    chunk_overlap: int,
    refresh: bool = False,
    dest_dir: Path,
) -> list[TextChunk]:
    """Build text chunks from Canvas wiki pages and any accessible PDFs."""
    base_url, token = _canvas_settings()
    cid = _course_id(course_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dest_dir / "canvas_chunks.json"

    if not refresh and cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        return [
            TextChunk(chunk_id=c["chunk_id"], source_file=c["source_file"], text=c["text"])
            for c in raw
        ]

    out: list[TextChunk] = []

    # 1) PDFs linked inside module wiki pages (Northeastern / no Files sidebar)
    print("Syncing linked PDFs from module pages...")
    try:
        pdf_paths = sync_linked_pdfs(dest_dir / "pdfs", course_id=course_id, refresh=refresh)
        out.extend(
            build_pdf_chunks(pdf_paths, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
        )
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"  Linked PDFs failed: {exc}")

    # 2) Optional: wiki page text (supplement if PDFs sparse)
    if len(out) < 5:
        pages = list_module_pages(base_url, token, cid)
        print(f"  Supplementing with Canvas page text: {len(pages)} pages")
        for page in pages:
            slug = re.sub(r"[^\w.\-]+", "_", page["page_url"]).strip("_")
            try:
                text = fetch_page_text(base_url, token, cid, page["page_url"])
            except httpx.HTTPError as exc:
                print(f"  skip page {page['title']}: {exc}")
                continue
            if not text:
                continue
            pieces = chunk_text(text, chunk_chars=chunk_chars, chunk_overlap=chunk_overlap)
            source = f"{page['module']} :: {page['title']}"
            for i, piece in enumerate(pieces):
                out.append(
                    TextChunk(chunk_id=f"page_{slug}_c{i:03d}", source_file=source, text=piece)
                )

    if not out:
        raise RuntimeError("No Canvas content extracted.")

    cache_path.write_text(
        json.dumps(
            [{"chunk_id": c.chunk_id, "source_file": c.source_file, "text": c.text} for c in out],
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def _safe_name(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^\w.\-]+", "_", stem).strip("_")
    return cleaned or "file"


def discover_linked_file_ids(base_url: str, token: str, course_id: str) -> set[int]:
    """
    Northeastern-style courses: no Files sidebar, but module wiki pages link
    /courses/:id/files/:file_id URLs. Students can fetch each file by id.
    """
    headers = _auth_headers(token)
    file_ids: set[int] = set()
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        mods = _paginate(
            client,
            f"{base_url}/api/v1/courses/{course_id}/modules?per_page=100",
            headers,
        )
        for mod in mods:
            items = _paginate(
                client,
                f"{base_url}/api/v1/courses/{course_id}/modules/{mod['id']}/items?per_page=100",
                headers,
            )
            for it in items:
                if it.get("type") != "Page" or not it.get("page_url"):
                    continue
                resp = client.get(
                    f"{base_url}/api/v1/courses/{course_id}/pages/{it['page_url']}",
                    headers=headers,
                )
                resp.raise_for_status()
                body = resp.json().get("body", "") or ""
                file_ids.update(int(x) for x in re.findall(r"/files/(\d+)", body))
    return file_ids


def sync_linked_pdfs(
    dest_dir: Path,
    *,
    course_id: int | None,
    refresh: bool = False,
) -> list[Path]:
    """Download PDFs linked from module wiki pages (per-file API, not /courses/files list)."""
    base_url, token = _canvas_settings()
    cid = _course_id(course_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.json"

    if not refresh and manifest_path.exists() and any(dest_dir.glob("*.pdf")):
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [dest_dir / name for name in meta.get("files", [])]

    file_ids = discover_linked_file_ids(base_url, token, cid)
    if not file_ids:
        raise RuntimeError("No /files/<id> links found in Canvas module pages.")

    headers = _auth_headers(token)
    saved: list[str] = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for fid in sorted(file_ids):
            meta_resp = client.get(f"{base_url}/api/v1/files/{fid}", headers=headers)
            if meta_resp.status_code != 200:
                print(f"  skip file {fid}: metadata HTTP {meta_resp.status_code}")
                continue
            meta = meta_resp.json()
            display = str(meta.get("display_name", f"file_{fid}"))
            if not _is_pdf(meta):
                continue
            local_name = f"{fid}_{_safe_name(display)}.pdf"
            local_path = dest_dir / local_name
            if local_path.exists() and not refresh:
                saved.append(local_name)
                continue
            download_url = meta.get("url")
            if not download_url:
                continue
            resp = client.get(download_url, headers=headers)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            saved.append(local_name)
            print(f"  downloaded {display}")

    if not saved:
        raise RuntimeError("Found file links but no PDFs downloaded.")

    manifest_path.write_text(
        json.dumps(
            {
                "course_id": cid,
                "canvas_base": urlparse(base_url).netloc,
                "n_linked_ids": len(file_ids),
                "n_pdfs": len(saved),
                "files": saved,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  {len(saved)} PDFs from {len(file_ids)} linked file ids")
    return [dest_dir / name for name in saved]
