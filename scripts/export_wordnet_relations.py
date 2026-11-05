#!/usr/bin/env python3
"""Export English WordNet relations in the lexicon JSONL format read by the token graph.

The token-graph variants (natlog_ot_v1, rgat_adapter_v1, coverage_pl_v1) build their
lexical edges from JSONL rows ``{"term", "aliases", "senses": [{"synonyms",
"antonyms", "hypernyms", "hyponyms"}]}``. For Portuguese they use PT-LKB +
Wiktionary-PT; this script produces the English counterpart from WordNet 3.0 so the
English experiment (experiment_0010) uses a comparable lexicon.

Only single-word lemmas are kept, and only the first ``--senses-per-pos`` synsets of
each part of speech (WordNet orders senses by frequency), which keeps rare senses
(e.g. dog -> "frump") from adding noisy edges. Hypernyms and hyponyms are direct links
(including instance links), as in PT-LKB.

    poetry run python scripts/export_wordnet_relations.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def single_word(name: str) -> str | None:
    name = name.lower()
    return None if "_" in name or "-" in name else name


def lemma_names(synsets) -> list[str]:
    out = []
    for synset in synsets:
        for lemma in synset.lemmas():
            word = single_word(lemma.name())
            if word:
                out.append(word)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "cache" / "wordnet_relations_en.jsonl")
    parser.add_argument("--senses-per-pos", type=int, default=3)
    args = parser.parse_args()

    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        wn.ensure_loaded()

    senses_by_word: dict[str, list[dict]] = defaultdict(list)
    words = {single_word(name) for name in wn.all_lemma_names()}
    words.discard(None)
    for word in sorted(words):
        for pos in ("n", "v", "a", "r"):
            for synset in wn.synsets(word, pos=pos)[: args.senses_per_pos]:
                own = [lemma for lemma in synset.lemmas() if lemma.name().lower() == word]
                antonyms = [single_word(ant.name()) for lemma in own for ant in lemma.antonyms()]
                sense = {
                    "pos": pos,
                    "gloss": synset.definition(),
                    "synonyms": sorted({w for w in lemma_names([synset]) if w != word}),
                    "antonyms": sorted({w for w in antonyms if w}),
                    "hypernyms": sorted(set(lemma_names(synset.hypernyms() + synset.instance_hypernyms())) - {word}),
                    "hyponyms": sorted(set(lemma_names(synset.hyponyms() + synset.instance_hyponyms())) - {word}),
                }
                if any(sense[key] for key in ("synonyms", "antonyms", "hypernyms", "hyponyms")):
                    senses_by_word[word].append(sense)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with args.output.open("w", encoding="utf-8") as handle:
        for word, senses in senses_by_word.items():
            for sense in senses:
                for key in ("synonyms", "antonyms", "hypernyms", "hyponyms"):
                    counts[key] += len(sense[key])
            row = {"term": word, "aliases": [word], "language": "en", "senses": senses,
                   "verified": True, "source": "wordnet-3.0"}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(senses_by_word)} terms to {args.output} ({dict(counts)})")


if __name__ == "__main__":
    main()
