#!/usr/bin/env python3
"""Probe Canvas module items for downloadable content."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))
from vector_linalg.embeddings import _load_dotenv

_load_dotenv()
base = os.environ["CANVAS_BASE_URL"].rstrip("/")
token = os.environ["CANVAS_API_TOKEN"]
cid = os.environ["CANVAS_COURSE_ID"]
headers = {"Authorization": f"Bearer {token}"}

files: list[dict] = []
pages: list[dict] = []

with httpx.Client(timeout=30, follow_redirects=True) as client:
    mods = client.get(f"{base}/api/v1/courses/{cid}/modules?per_page=100", headers=headers).json()
    for mod in mods:
        url = f"{base}/api/v1/courses/{cid}/modules/{mod['id']}/items?per_page=100"
        while url:
            r = client.get(url, headers=headers)
            r.raise_for_status()
            for it in r.json():
                t = it.get("type")
                if t == "File":
                    files.append(it)
                elif t == "Page":
                    pages.append(it)
            link = r.headers.get("link", "")
            url = None
            m = re.search(r'<([^>]+)>; rel="next"', link)
            if m:
                url = m.group(1)

    print(f"modules: {len(mods)}, file items: {len(files)}, page items: {len(pages)}")
    for f in files[:10]:
        print(" FILE", f.get("title"), "content_id=", f.get("content_id"))

    if pages:
        page_url = pages[0]["page_url"]
        r = client.get(f"{base}/api/v1/courses/{cid}/pages/{page_url}", headers=headers)
        print("sample page", pages[0]["title"], r.status_code)
        if r.status_code == 200:
            body = r.json().get("body", "")
            print(" html length", len(body))
