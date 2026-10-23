#!/usr/bin/env python3
"""Make the compact PIIG v4 cache from a local Wiktextract JSONL export."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from tqdm.auto import tqdm

from graphnli_axb.data import load_axb, load_mnli, load_xnli, project_root
from graphnli_axb.wiktionary import compact_wiktextract_entry, extract_lexical_candidates, normalize_term


def collect_terms(data_dir: Path, languages: list[str], sources: list[str], max_examples: int, max_terms: int, manifests: list[Path] | None = None) -> set[str]:
    groups = []
    if "sick" in sources:
        sick_path = data_dir / "raw" / "sick" / "SICK.txt"
        if not sick_path.exists():
            raise FileNotFoundError(f"Missing {sick_path}")
        sick_rows = []
        with sick_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("SemEval_set") == "TRAIN":
                    sentence1 = row.get("sentence_A")
                    sentence2 = row.get("sentence_B")
                    if sentence1 and sentence2:
                        sick_rows.append((sentence1, sentence2))
        groups.append(sick_rows)
    if "mnli" in sources:
        groups.append(load_mnli(data_dir, "train"))
    if "axb" in sources:
        groups.append(load_axb(data_dir))
    if "xnli" in sources:
        rows = []
        for split in ("dev", "test"):
            rows.extend(load_xnli(data_dir, split, languages))
        groups.append(rows)
    for manifest in manifests or []:
        rows = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sentence1 = row.get("sentence1", row.get("premise", row.get("sentence_A")))
                sentence2 = row.get("sentence2", row.get("hypothesis", row.get("sentence_B")))
                if sentence1 and sentence2:
                    rows.append((str(sentence1), str(sentence2)))
        groups.append(rows)
    if max_examples > 0 and groups:
        base, remainder = divmod(max_examples, len(groups))
        rows = [row for index, group in enumerate(groups) for row in group[: base + (1 if index < remainder else 0)]]
    else:
        rows = [row for group in groups for row in group]
    terms = set()
    for row in rows:
        texts = row if isinstance(row, tuple) else (row.sentence1, row.sentence2)
        for text in texts:
            terms.update(extract_lexical_candidates(text, max_terms))
    return terms


def open_jsonl(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Local JSONL produced by Wiktextract/wiktwords.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/cache/wiktionary_entries.jsonl")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--language", default=None, help="Language code represented by the input JSONL, e.g. en or pt.")
    parser.add_argument("--sources", default="mnli,axb,xnli")
    parser.add_argument("--manifests", default="", help="Comma-separated JSONL manifests with premise/hypothesis fields.")
    parser.add_argument("--max-examples", type=int, default=10000)
    parser.add_argument("--max-terms", type=int, default=4)
    args = parser.parse_args()

    root = project_root()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output = root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    manifests = []
    for value in [item.strip() for item in args.manifests.split(",") if item.strip()]:
        path = Path(value)
        manifests.append(path if path.is_absolute() else root / path)
    language = args.language or (languages[0] if languages else "en")
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    invalid = sorted(set(sources) - {"sick", "mnli", "axb", "xnli"})
    if invalid:
        raise ValueError(f"Unsupported sources: {', '.join(invalid)}")
    wanted = collect_terms(root / args.data_dir, languages, sources, args.max_examples, args.max_terms, manifests)
    found = {}
    with open_jsonl(input_path) as handle:
        for line in tqdm(handle, desc="Indexing Wiktextract JSONL", unit="entry"):
            if not line.strip():
                continue
            row = compact_wiktextract_entry(json.loads(line), language=language)
            if row and normalize_term(row["term"]) in wanted:
                found.setdefault(normalize_term(row["term"]), row)
    with output.open("w", encoding="utf-8") as handle:
        for key in sorted(found):
            handle.write(json.dumps(found[key], ensure_ascii=False) + "\n")
    print(f"wrote {output} | matched={len(found)} of {len(wanted)} task terms")


if __name__ == "__main__":
    main()
