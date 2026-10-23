#!/usr/bin/env python3
"""Build a PIIG cache from only Wiktionary terms used by this project datasets.

The complete Wikimedia XML dump is still needed once to prepare Wiktextract's
local SQLite database.  Unlike a full extraction, this script never writes an
all-entries JSONL: it asks Wiktextract only for lexical candidates found in
MNLI, AX-b, and XNLI, then stores PIIG's compact cache directly.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from tqdm.auto import tqdm

from build_wiktionary_cache_from_jsonl import collect_terms, open_jsonl
from graphnli_axb.data import project_root
from graphnli_axb.wiktionary import compact_wiktextract_entry, normalize_term


# Editions accepted by the wiktextract build vendored in this project. Some
# Wiktionary dumps exist but are not supported by this parser in dump mode.
WIKTEXTRACT_DUMP_EDITIONS = {
    "cs", "de", "el", "en", "es", "fr", "id", "it", "ja", "ko",
    "ku", "ms", "nl", "pl", "pt", "ru", "simple", "sv", "th", "tr", "vi", "zh",
}


def prepare_database(command: list[str]) -> None:
    """Run Wiktextract's one-time dump pass with visible elapsed time.

    Wiktextract does not expose a stable count of XML pages before parsing, so
    an ETA here would be invented. The following extraction stage has a known
    number of batches and therefore provides a real tqdm ETA.
    """
    print("+", " ".join(command))
    process = subprocess.Popen(command)
    started = time.monotonic()
    with tqdm(total=None, desc="Preparing Wiktextract DB", unit="s") as progress:
        while process.poll() is None:
            time.sleep(1)
            progress.update(1)
            progress.set_postfix_str(f"elapsed {int(time.monotonic() - started)}s")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def load_existing_cache(path: Path, wanted: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    found = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            term = entry.get("term")
            if term and normalize_term(term) in wanted:
                found[normalize_term(term)] = entry
    return found


def write_cache(path: Path, found: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(found):
            handle.write(json.dumps(found[key], ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default=None)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--sources", default="mnli,axb,xnli")
    parser.add_argument("--manifests", default="", help="Comma-separated JSONL manifests with premise/hypothesis fields.")
    parser.add_argument("--max-examples", type=int, default=10000)
    parser.add_argument("--max-terms", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--rebuild-db", action="store_true")
    args = parser.parse_args()

    root = project_root()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    if len(languages) != 1:
        raise ValueError("The local dump workflow processes one Wiktionary edition at a time; pass one --languages value.")
    language = languages[0]
    if language not in WIKTEXTRACT_DUMP_EDITIONS:
        supported = ", ".join(sorted(WIKTEXTRACT_DUMP_EDITIONS))
        raise ValueError(
            f"The installed wiktextract does not support the {language!r} edition "
            f"in local-dump mode. Supported editions: {supported}. "
            "Use scripts/build_wiktionary_cache.py for this language, or install "
            "a wiktextract version that supports its dump."
        )
    dump_value = args.dump or f"data/raw/wiktionary/{language}wiktionary-latest-pages-articles.xml.bz2"
    db_value = args.db_path or f"data/cache/wiktextract_{language}.db"
    out_value = args.out or f"data/cache/wiktionary_entries_{language}.jsonl"
    dump, db_path, output = resolve(dump_value), resolve(db_value), resolve(out_value)
    if not dump.exists():
        raise FileNotFoundError(f"Wiktionary dump not found: {dump}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    manifests = []
    for value in [item.strip() for item in args.manifests.split(",") if item.strip()]:
        path = Path(value)
        manifests.append(path if path.is_absolute() else root / path)
    invalid = sorted(set(sources) - {"sick", "mnli", "axb", "xnli"})
    if invalid:
        raise ValueError(f"Unsupported sources: {', '.join(invalid)}")
    wanted = collect_terms(root / args.data_dir, languages, sources, args.max_examples, args.max_terms, manifests)
    found = load_existing_cache(output, wanted)
    titles = sorted(wanted - set(found))
    print(
        f"Selected {len(wanted)} lexical candidates from {', '.join(sources)} "
        f"({len(found)} already cached; {len(titles)} to extract)."
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rebuild_db and db_path.exists():
        db_path.unlink()
    if not db_path.exists():
        prepare_database([
            "wiktwords", "--edition", language, "--db-path", str(db_path),
            "--num-processes", str(args.num_processes), str(dump),
        ])

    batches = range(0, len(titles), args.batch_size)
    for offset in tqdm(batches, total=(len(titles) + args.batch_size - 1) // args.batch_size, desc="Extracting task terms", unit="batch"):
        batch = titles[offset : offset + args.batch_size]
        # Wiktextract treats an existing local path as page-source text. Run
        # inside an empty directory so lexical titles such as "data" cannot
        # collide with this repository's data/ directory.
        with tempfile.TemporaryDirectory(prefix="graphnli-wiktionary-") as temporary_dir:
            temporary_output = Path(temporary_dir) / "entries.jsonl"
            command = [
                "wiktwords", "--edition", language, "--db-path", str(db_path),
                "--language-code", language, "--linkages", "--num-processes", str(args.num_processes),
                "--quiet", "--out", str(temporary_output),
            ]
            for title in batch:
                command.extend(("--page", title))
            subprocess.run(command, check=True, cwd=temporary_dir)
            with open_jsonl(temporary_output) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = compact_wiktextract_entry(json.loads(line), language=language)
                    if entry and normalize_term(entry["term"]) in wanted:
                        found.setdefault(normalize_term(entry["term"]), entry)
        write_cache(output, found)

    write_cache(output, found)
    print(f"wrote {output} | matched={len(found)} of {len(wanted)} task terms | database={db_path}")


if __name__ == "__main__":
    main()
