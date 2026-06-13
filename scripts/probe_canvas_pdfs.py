#!/usr/bin/env python3
"""Find PDF file IDs linked inside Canvas module pages."""
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

file_ids: set[int] = set()

with httpx.Client(timeout=30, follow_redirects=True) as client:
    mods = client.get(f"{base}/api/v1/courses/{cid}/modules?per_page=100", headers=headers).json()
    print(f"modules: {len(mods)}")
    for mod in mods:
        items = client.get(
            f"{base}/api/v1/courses/{cid}/modules/{mod['id']}/items?per_page=100",
            headers=headers,
        ).json()
        for it in items:
            if it.get("type") != "Page":
                continue
            body = client.get(
                f"{base}/api/v1/courses/{cid}/pages/{it['page_url']}",
                headers=headers,
            ).json().get("body", "")
            found = set(int(x) for x in re.findall(r"/files/(\d+)", body))
            if found:
                print(f"  {it.get('title')}: {len(found)} file link(s)")
                file_ids |= found

print(f"\ntotal file ids: {len(file_ids)}")
with httpx.Client(timeout=30, follow_redirects=True) as client:
    pdf_ok = 0
    for fid in sorted(file_ids)[:12]:
        r = client.get(f"{base}/api/v1/files/{fid}", headers=headers)
        if r.status_code != 200:
            print(f"  {fid}: HTTP {r.status_code}")
            continue
        meta = r.json()
        name = meta.get("display_name", "?")
        ctype = meta.get("content-type", "")
        if "pdf" not in name.lower() and "pdf" not in ctype.lower():
            print(f"  {fid}: skip {name}")
            continue
        dr = client.get(meta["url"], headers=headers)
        print(f"  {fid}: {name[:60]} -> download {dr.status_code} ({len(dr.content)} bytes)")
        if dr.status_code == 200:
            pdf_ok += 1
    print(f"pdf downloads ok in sample: {pdf_ok}")
