#!/usr/bin/env python3
"""Convert PT-LexicalSemantics relations into a compact PIIG cache.

The input is the ``triplos_pesados_norm.tsv.zip`` file from
NLP-CISUC/PT-LexicalSemantics. When manifests are provided, only relations
touching words observed in those manifests are retained.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from graphnli_axb.data import project_root
from graphnli_axb.wiktionary import extract_lexical_candidates, normalize_term


RELATION_MAP = {
    "SINONIMO_N_DE": "synonym",
    "SINONIMO_V_DE": "synonym",
    "SINONIMO_ADJ_DE": "synonym",
    "SINONIMO_ADV_DE": "synonym",
    "HIPERONIMO_DE": "hypernym",
    "HIPERONIMO_ACCAO_DE": "hypernym",
    "PARTE_DE": "part",
    "MEMBRO_DE": "member",
    "LOCAL_ORIGEM_DE": "origin",
    "FINALIDADE_DE": "purpose",
    "ACCAO_QUE_CAUSA": "causes",
    "DIZ_SE_DO_QUE": "describes",
    "DIZ_SE_SOBRE": "about",
    "PROPRIEDADE_SEMELHANTE_A": "similar_property",
    "FAZ_SE_COM": "used_with",
    "PARTE_DE_ALGO_COM_PROPRIEDADE": "property_part",
}


def manifest_terms(paths: list[Path], max_terms: int) -> set[str]:
    terms: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("sentence1", "sentence2", "premise", "hypothesis", "sentence_A", "sentence_B"):
                    if row.get(key):
                        terms.update(normalize_term(t) for t in extract_lexical_candidates(str(row[key]), max_terms))
    return terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="triplos_pesados_norm.tsv.zip")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifests", default="", help="Comma-separated JSONL manifests used to filter terms")
    parser.add_argument("--max-terms", type=int, default=8)
    parser.add_argument("--min-resources", type=int, default=1)
    args = parser.parse_args()

    root = project_root()
    input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
    output = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    paths = []
    for value in args.manifests.split(","):
        if value.strip():
            path = Path(value.strip())
            paths.append(path if path.is_absolute() else root / path)
    wanted = manifest_terms(paths, args.max_terms) if paths else None
    entries: dict[str, dict] = {}
    relation_count = 0

    with ZipFile(input_path) as archive:
        tsv_name = next(name for name in archive.namelist() if name.endswith("triplos_pesados_norm.tsv"))
        with archive.open(tsv_name) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 12 or not row[0]:
                    continue
                match = re.match(r"^(.*?)\s+([A-Z_]+)\s+(.*)$", row[0])
                if not match or match.group(2) not in RELATION_MAP:
                    continue
                left = normalize_term(match.group(1))
                raw_relation = match.group(2)
                right = normalize_term(match.group(3))
                try:
                    resources = int(row[1])
                except ValueError:
                    continue
                if resources < args.min_resources or (wanted and left not in wanted and right not in wanted):
                    continue
                relation = RELATION_MAP[raw_relation]
                relation_count += 1
                scores = [float(value) for value in row[2:8]]
                edge = {"term": right, "relation": relation, "resources": resources, "bert": scores[:3], "gpt": scores[3:]}
                for source, target in ((left, right), (right, left)):
                    entry = entries.setdefault(source, {
                        "term": source, "aliases": [source], "language": "pt", "verified": True,
                        "source": "pt_lexical_semantics", "senses": [],
                    })
                    sense = next((item for item in entry["senses"] if item.get("pos") == "pt-lkb"), None)
                    if sense is None:
                        sense = {"pos": "pt-lkb", "gloss": "PT-LKB lexical relations", "relations": []}
                        entry["senses"].append(sense)
                    if source == left:
                        relation_edge = edge
                    else:
                        reverse = {"hypernym": "hyponym", "part": "whole", "member": "group"}.get(relation, relation)
                        relation_edge = {**edge, "term": left, "relation": reverse}
                    if not any(item["term"] == target and item["relation"] == relation_edge["relation"] for item in sense["relations"]):
                        sense["relations"].append(relation_edge)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for term in sorted(entries):
            handle.write(json.dumps(entries[term], ensure_ascii=False) + "\n")
    print(f"wrote {output} | entries={len(entries)} | relations={relation_count} | filtered_terms={len(wanted) if wanted else 'none'}")


if __name__ == "__main__":
    main()
