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
    parser.add_argument("--dump", default="data/raw/wiktionary/enwiktionary-latest-pages-articles.xml.bz2")
    parser.add_argument("--db-path", default="data/cache/wiktextract_en.db")
    parser.add_argument("--out", default="data/cache/wiktionary_entries.jsonl")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--sources", default="mnli,axb,xnli")
    parser.add_argument("--max-examples", type=int, default=10000)
    parser.add_argument("--max-terms", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--rebuild-db", action="store_true")
    args = parser.parse_args()

    root = project_root()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    dump, db_path, output = resolve(args.dump), resolve(args.db_path), resolve(args.out)
    if not dump.exists():
        raise FileNotFoundError(f"Wiktionary dump not found: {dump}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    invalid = sorted(set(sources) - {"mnli", "axb", "xnli"})
    if invalid:
        raise ValueError(f"Unsupported sources: {', '.join(invalid)}")
    if languages != ["en"]:
        raise ValueError("The local enwiktionary workflow currently supports --languages en only.")

    wanted = collect_terms(root / args.data_dir, languages, sources, args.max_examples, args.max_terms)
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
            "wiktwords", "--edition", "en", "--db-path", str(db_path),
            "--num-processes", str(args.num_processes), "--skip-extraction", str(dump),
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
                "wiktwords", "--edition", "en", "--db-path", str(db_path),
                "--language-code", "en", "--linkages", "--num-processes", str(args.num_processes),
                "--quiet", "--out", str(temporary_output),
            ]
            for title in batch:
                command.extend(("--page", title))
            subprocess.run(command, check=True, cwd=temporary_dir)
            with open_jsonl(temporary_output) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = compact_wiktextract_entry(json.loads(line))
                    if entry and normalize_term(entry["term"]) in wanted:
                        found.setdefault(normalize_term(entry["term"]), entry)
        write_cache(output, found)

    write_cache(output, found)
    print(f"wrote {output} | matched={len(found)} of {len(wanted)} task terms | database={db_path}")


if __name__ == "__main__":
    main()
