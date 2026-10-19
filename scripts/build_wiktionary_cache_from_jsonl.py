#!/usr/bin/env python3
"""Make the compact PIIG v4 cache from a local Wiktextract JSONL export."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from tqdm.auto import tqdm

from graphnli_axb.data import load_axb, load_mnli, load_xnli, project_root
from graphnli_axb.wiktionary import extract_lexical_candidates, normalize_term


def collect_terms(data_dir: Path, languages: list[str], sources: list[str], max_examples: int, max_terms: int) -> set[str]:
    groups = []
    if "mnli" in sources:
        groups.append(load_mnli(data_dir, "train"))
    if "axb" in sources:
        groups.append(load_axb(data_dir))
    if "xnli" in sources:
        rows = []
        for split in ("dev", "test"):
            rows.extend(load_xnli(data_dir, split, languages))
        groups.append(rows)
    if max_examples > 0 and groups:
        base, remainder = divmod(max_examples, len(groups))
        rows = [row for index, group in enumerate(groups) for row in group[: base + (1 if index < remainder else 0)]]
    else:
        rows = [row for group in groups for row in group]
    terms = set()
    for row in rows:
        for text in (row.sentence1, row.sentence2):
            terms.update(extract_lexical_candidates(text, max_terms))
    return terms


def words(values) -> list[str]:
    result = []
    for value in values or []:
        word = value.get("word") if isinstance(value, dict) else value
        if word and normalize_term(str(word)) not in {normalize_term(item) for item in result}:
            result.append(str(word))
    return result


def compact_entry(row: dict) -> dict | None:
    word = row.get("word")
    if not word or row.get("lang_code") != "en" or row.get("redirect"):
        return None
    senses = []
    linkages = {name: words(row.get(name)) for name in ("synonyms", "antonyms", "hypernyms", "hyponyms")}
    for sense in row.get("senses") or []:
        glosses = sense.get("glosses") or sense.get("raw_glosses") or []
        gloss = str(glosses[0]).strip() if glosses else ""
        if not gloss:
            continue
        values = {
            name: words(sense.get(name)) or linkages[name]
            for name in ("synonyms", "antonyms", "hypernyms", "hyponyms")
        }
        senses.append({"pos": row.get("pos", ""), "gloss": gloss, **values})
        if len(senses) >= 3:
            break
    if not senses:
        return None
    return {"term": word, "aliases": [word], "language": "en", "senses": senses, "verified": True, "source": "wiktextract_dump"}


def open_jsonl(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Local JSONL produced by Wiktextract/wiktwords.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/cache/wiktionary_entries.jsonl")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--sources", default="mnli,axb,xnli")
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
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    invalid = sorted(set(sources) - {"mnli", "axb", "xnli"})
    if invalid:
        raise ValueError(f"Unsupported sources: {', '.join(invalid)}")
    wanted = collect_terms(root / args.data_dir, languages, sources, args.max_examples, args.max_terms)
    found = {}
    with open_jsonl(input_path) as handle:
        for line in tqdm(handle, desc="Indexing Wiktextract JSONL", unit="entry"):
            if not line.strip():
                continue
            row = compact_entry(json.loads(line))
            if row and normalize_term(row["term"]) in wanted:
                found.setdefault(normalize_term(row["term"]), row)
    with output.open("w", encoding="utf-8") as handle:
        for key in sorted(found):
            handle.write(json.dumps(found[key], ensure_ascii=False) + "\n")
    print(f"wrote {output} | matched={len(found)} of {len(wanted)} task terms")


if __name__ == "__main__":
    main()
