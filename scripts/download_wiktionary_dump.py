#!/usr/bin/env python3
"""Download the official English Wiktionary articles dump with resume support."""
from __future__ import annotations

import argparse
import hashlib
import re
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

from graphnli_axb.data import project_root


DUMP_URL = "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2"
CHECKSUM_URL = "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-md5sums.txt"
CHUNK_SIZE = 1024 * 1024


def expected_md5(url: str, filename: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    match = re.search(rf"^([0-9a-fA-F]{{32}})\s+\*?{re.escape(filename)}$", text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find {filename} in {url}")
    return match.group(1).lower()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "GraphNLI-AXb/0.4 (research dump downloader)"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            partial.unlink(missing_ok=True)
            return download(url, destination)
        remaining = response.headers.get("Content-Length")
        total = offset + int(remaining) if remaining else None
        with partial.open("ab") as handle, tqdm(
            total=total,
            initial=offset,
            unit="B",
            unit_scale=True,
            desc="Downloading enwiktionary dump",
        ) as progress:
            while block := response.read(CHUNK_SIZE):
                handle.write(block)
                progress.update(len(block))
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/wiktionary/enwiktionary-latest-pages-articles.xml.bz2")
    parser.add_argument("--url", default=DUMP_URL)
    parser.add_argument("--checksum-url", default=CHECKSUM_URL)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    root = project_root()
    output = Path(args.out)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        download(args.url, output)

    if not args.no_verify:
        expected = expected_md5(args.checksum_url, output.name)
        actual = file_md5(output)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {output}: expected {expected}, got {actual}")
        print(f"verified {output} ({actual})")
    else:
        print(f"downloaded {output} without checksum verification")


if __name__ == "__main__":
    main()
