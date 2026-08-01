"""Repair moved NHANES XPT URLs without changing the pinned TableShift task.

The TableShift commit used by the prospective Gate-2 replication points to
legacy CDC paths that now return HTML 404 pages. CDC exposes the same named
XPT files below /Nchs/Data/Nhanes/Public/<cycle>/DataFiles/. This utility
downloads only the exact source filenames selected by the pinned task,
validates their XPORT header, and records URL and content hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


CURRENT_ROOT = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
XPORT_MAGIC = b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def cache_name(url: str, year: str) -> str:
    filename = Path(urlparse(url).path).name
    stem, suffix = filename.rsplit(".", 1)
    return f"{stem}{year}.{suffix}"


def current_url(old_url: str, year: str) -> str:
    filename = Path(urlparse(old_url).path).name
    return f"{CURRENT_ROOT}/{year}/DataFiles/{filename}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    args = parser.parse_args()

    from tableshift.datasets.nhanes import get_nhanes_data_sources

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = get_nhanes_data_sources("lead", years=[
        1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017
    ])
    for year, urls in sorted(sources.items()):
        for old_url in urls:
            url = current_url(old_url, year)
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            payload = response.content
            if len(payload) < 80 or not payload.startswith(XPORT_MAGIC):
                raise RuntimeError(
                    f"official CDC response is not an XPORT file: {url} "
                    f"({len(payload)} bytes)"
                )
            destination = args.cache_dir / cache_name(old_url, year)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            rows.append(
                {
                    "year": int(year),
                    "legacy_url": old_url,
                    "current_official_url": url,
                    "cache_file": destination.name,
                    "size_bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )

    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["year", "cache_file"]
    ).to_csv(args.audit_csv, index=False)
    print(f"[OK] repaired {len(rows)} pinned NHANES XPT sources")


if __name__ == "__main__":
    main()
