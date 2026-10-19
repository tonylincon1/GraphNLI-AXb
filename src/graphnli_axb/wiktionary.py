from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


WIKTIONARY_RELATIONS = [
    "wiktionary_entry_present",
    "wiktionary_gloss_present",
    "wiktionary_synonym",
    "wiktionary_antonym",
    "wiktionary_hypernym",
    "wiktionary_hyponym",
]

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "he", "her", "him",
    "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "they", "this", "to", "was",
    "were", "will", "with", "would", "you", "your",
}
LEXICAL_RELATION_NAMES = ("synonym", "antonym", "hypernym", "hyponym")


@dataclass(frozen=True)
class WiktionarySpec:
    entries: List[dict]
    pairs: List[dict]
    relations: np.ndarray


def normalize_term(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_lexical_candidates(text: str, max_terms: int) -> List[str]:
    """Keep content words that can usefully receive an English dictionary gloss."""
    terms = []
    seen = set()
    for match in TOKEN_RE.finditer(text):
        term = normalize_term(match.group(0))
        if len(term) < 3 or term in LEXICAL_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def load_wiktionary_cache(path: Path | None) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            term = row.get("term") or row.get("word") or row.get("title")
            if not term:
                continue
            cache[normalize_term(term)] = row
            for alias in row.get("aliases") or []:
                cache[normalize_term(alias)] = row
    return cache


def _sense_values(entry: dict, key: str) -> set[str]:
    values = set()
    for sense in entry.get("senses") or []:
        for value in sense.get(key) or []:
            values.add(normalize_term(str(value)))
    return values


def relation_between(left_term: str, left_entry: dict, right_term: str, right_entry: dict) -> str | None:
    """Find explicit cached lexical evidence, never infer a relation from a label."""
    left_key, right_key = normalize_term(left_term), normalize_term(right_term)
    if right_key in _sense_values(left_entry, "synonyms") or left_key in _sense_values(right_entry, "synonyms"):
        return "synonym"
    if right_key in _sense_values(left_entry, "antonyms") or left_key in _sense_values(right_entry, "antonyms"):
        return "antonym"
    if right_key in _sense_values(left_entry, "hypernyms") or left_key in _sense_values(right_entry, "hyponyms"):
        return "hypernym"
    if right_key in _sense_values(left_entry, "hyponyms") or left_key in _sense_values(right_entry, "hypernyms"):
        return "hyponym"
    return None


def entry_summary(entry: dict) -> str:
    term = entry.get("term") or entry.get("word") or "term"
    senses = entry.get("senses") or []
    if not senses:
        return f"wiktionary entry: {term}"
    sense = senses[0]
    pos = sense.get("pos") or ""
    gloss = sense.get("gloss") or ""
    return " ".join(piece for piece in ["wiktionary entry:", str(term), str(pos), str(gloss)] if piece).strip()


def build_wiktionary_spec(
    sentence1: str,
    sentence2: str,
    cache: Dict[str, dict],
    max_entries: int,
) -> WiktionarySpec:
    """Retrieve cached lexical evidence for differing premise/hypothesis terms.

    Retrieval is deterministic and local. It only decides which evidence is
    available; PIIG cross-attention still learns whether each gloss matters.
    """
    premise_terms = extract_lexical_candidates(sentence1, max_entries * 3)
    hypothesis_terms = extract_lexical_candidates(sentence2, max_entries * 3)
    premise = [(term, cache[term]) for term in premise_terms if term in cache and cache[term].get("verified", False)]
    hypothesis = [(term, cache[term]) for term in hypothesis_terms if term in cache and cache[term].get("verified", False)]

    pairs = []
    used_terms = []
    for left_term, left_entry in premise:
        for right_term, right_entry in hypothesis:
            if left_term == right_term:
                continue
            relation = relation_between(left_term, left_entry, right_term, right_entry)
            if relation:
                pairs.append({"premise_term": left_term, "hypothesis_term": right_term, "relation": relation})
                for term, entry in ((left_term, left_entry), (right_term, right_entry)):
                    if not any(existing[0] == term for existing in used_terms):
                        used_terms.append((term, entry))

    for term, entry in premise + hypothesis:
        if len(used_terms) >= max_entries:
            break
        if not any(existing[0] == term for existing in used_terms):
            used_terms.append((term, entry))

    entries = []
    for term, entry in used_terms[:max_entries]:
        copied = dict(entry)
        copied["matched_term"] = term
        entries.append(copied)

    selected_terms = {entry["matched_term"] for entry in entries}
    pairs = [
        pair
        for pair in pairs
        if pair["premise_term"] in selected_terms and pair["hypothesis_term"] in selected_terms
    ][:max_entries]

    relations = np.zeros(len(WIKTIONARY_RELATIONS), dtype=np.float32)
    if entries:
        relations[WIKTIONARY_RELATIONS.index("wiktionary_entry_present")] = 1.0
    if any((entry.get("senses") or [{}])[0].get("gloss") for entry in entries):
        relations[WIKTIONARY_RELATIONS.index("wiktionary_gloss_present")] = 1.0
    for pair in pairs:
        relations[WIKTIONARY_RELATIONS.index(f"wiktionary_{pair['relation']}")] = 1.0
    return WiktionarySpec(entries=entries, pairs=pairs, relations=relations)
