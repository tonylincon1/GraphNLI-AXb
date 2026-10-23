#!/usr/bin/env python3
"""Download and extract the official SICK English NLI dataset."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/2787612/files/SICK.zip?download=1"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "data" / "raw" / "sick"
    archive = directory / "SICK.zip"
    sick_file = directory / "SICK.txt"
    directory.mkdir(parents=True, exist_ok=True)
    if not sick_file.exists():
        if not archive.exists():
            print(f"downloading {URL}")
            with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as compressed:
            compressed.extractall(directory)
    print(f"ready: {sick_file}")


if __name__ == "__main__":
    main()
