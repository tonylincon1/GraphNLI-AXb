"""Build token-graph lexicon caches (EN and PT) from ConceptNet 5.7.

Reads the official assertions dump (conceptnet-assertions-5.7.0.csv.gz, one edge per line:
uri, relation, start, end, json info) and keeps single-word edges inside one language for the
relations the token graph understands:

    /r/Synonym        -> synonyms   (both directions)
    /r/Antonym        -> antonyms   (both directions)
    /r/DistinctFrom   -> antonyms   (mutually exclusive terms, e.g. cat / dog; a contradiction cue)
    /r/IsA            -> hypernyms of start, hyponyms of end

Edges below --min-weight are dropped (ConceptNet weights are ~1 for a single source).
Output: one JSONL row per term, in the format of wordnet_relations_en.jsonl, so the
LexicalRelationIndex reads it unchanged:

    {"term": "dog", "aliases": ["dog"], "language": "en", "verified": true,
     "source": "conceptnet-5.7", "senses": [{"pos": "conceptnet", "gloss": "",
     "synonyms": [...], "antonyms": [...], "hypernyms": [...], "hyponyms": [...]}]}

Usage:
    curl -L -o data/raw/conceptnet/conceptnet-assertions-5.7.0.csv.gz \\
        https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
    poetry run python scripts/build_conceptnet_cache.py
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"/r/Synonym": "synonyms", "/r/Antonym": "antonyms", "/r/DistinctFrom": "antonyms", "/r/IsA": "hypernyms"}
SYMMETRIC = {"/r/Synonym", "/r/Antonym", "/r/DistinctFrom"}


def concept(uri: str) -> tuple[str, str] | None:
    """'/c/en/dog/n/...' -> ('en', 'dog'); None for multiword terms."""
    parts = uri.split("/")
    if len(parts) < 4 or parts[1] != "c":
        return None
    term = parts[3].lower()
    if "_" in term or not term:
        return None
    return parts[2], term


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", default=str(ROOT / "data/raw/conceptnet/conceptnet-assertions-5.7.0.csv.gz"))
    parser.add_argument("--languages", default="en,pt")
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(ROOT / "data/cache"))
    args = parser.parse_args()

    languages = set(args.languages.split(","))
    table: dict[str, dict[str, dict[str, set]]] = {lang: defaultdict(lambda: defaultdict(set)) for lang in languages}
    kept = defaultdict(int)
    with gzip.open(args.dump, "rt", encoding="utf-8") as handle:
        for line in handle:
            _, relation, start, end, info = line.rstrip("\n").split("\t")
            if relation not in FIELDS:
                continue
            left, right = concept(start), concept(end)
            if not left or not right or left[0] != right[0] or left[0] not in languages or left[1] == right[1]:
                continue
            if json.loads(info).get("weight", 1.0) < args.min_weight:
                continue
            lang, a, b = left[0], left[1], right[1]
            field = FIELDS[relation]
            table[lang][a][field].add(b)
            if relation in SYMMETRIC:
                table[lang][b][field].add(a)
            else:  # a IsA b: b is more general than a
                table[lang][b]["hyponyms"].add(a)
            kept[(lang, relation)] += 1

    for lang in sorted(languages):
        path = Path(args.out_dir) / f"conceptnet_relations_{lang}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for term in sorted(table[lang]):
                fields = table[lang][term]
                sense = {"pos": "conceptnet", "gloss": ""}
                sense.update({f: sorted(fields.get(f, ())) for f in ("synonyms", "antonyms", "hypernyms", "hyponyms")})
                out.write(json.dumps({"term": term, "aliases": [term], "language": lang, "verified": True,
                                      "source": "conceptnet-5.7", "senses": [sense]}, ensure_ascii=False) + "\n")
        counts = ", ".join(f"{r.split('/')[-1]} {n:,}" for (l, r), n in sorted(kept.items()) if l == lang)
        print(f"{path}: {len(table[lang]):,} terms ({counts})")


if __name__ == "__main__":
    main()
