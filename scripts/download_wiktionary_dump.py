#!/usr/bin/env python3
"""Download an official Wiktionary articles dump with resume support."""
from __future__ import annotations

import argparse
import hashlib
import re
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

from graphnli_axb.data import project_root


CHUNK_SIZE = 1024 * 1024
USER_AGENT = "GraphNLI-AXb/0.4 (research dump downloader)"


def expected_md5(url: str, filename: str, language: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    match = re.search(rf"^([0-9a-fA-F]{{32}})\s+\*?{re.escape(filename)}$", text, flags=re.MULTILINE)
    # The `latest` URL redirects to a dated snapshot, while md5sums records
    # the snapshot filename rather than the stable `latest` alias.
    if not match and filename.endswith("-latest-pages-articles.xml.bz2"):
        match = re.search(
            rf"^([0-9a-fA-F]{{32}})\s+\*?{re.escape(language)}wiktionary-\d{{8}}-pages-articles\.xml\.bz2$",
            text,
            flags=re.MULTILINE,
        )
    if not match:
        raise RuntimeError(f"Could not find {filename} in {url}")
    return match.group(1).lower()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, language: str) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
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
            desc=f"Downloading {language}wiktionary dump",
        ) as progress:
            while block := response.read(CHUNK_SIZE):
                handle.write(block)
                progress.update(len(block))
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en", help="Wiktionary language edition, e.g. en, pt, es, de.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--checksum-url", default=None)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    language = args.language.strip().lower()
    if not re.fullmatch(r"[a-z-]+", language):
        raise ValueError(f"Invalid Wiktionary language code: {language}")
    filename = f"{language}wiktionary-latest-pages-articles.xml.bz2"
    dump_url = args.url or f"https://dumps.wikimedia.org/{language}wiktionary/latest/{filename}"
    checksum_url = args.checksum_url or f"https://dumps.wikimedia.org/{language}wiktionary/latest/{language}wiktionary-latest-md5sums.txt"
    output_name = args.out or f"data/raw/wiktionary/{filename}"

    root = project_root()
    output = Path(output_name)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        download(dump_url, output, language)

    if not args.no_verify:
        expected = expected_md5(checksum_url, output.name, language)
        actual = file_md5(output)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {output}: expected {expected}, got {actual}")
        print(f"verified {output} ({actual})")
    else:
        print(f"downloaded {output} without checksum verification")


if __name__ == "__main__":
    main()
