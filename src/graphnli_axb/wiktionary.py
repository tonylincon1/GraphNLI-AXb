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
    "wiktionary_part",
    "wiktionary_whole",
    "wiktionary_member",
    "wiktionary_group",
    "wiktionary_origin",
    "wiktionary_purpose",
    "wiktionary_causes",
    "wiktionary_describes",
    "wiktionary_about",
    "wiktionary_similar_property",
    "wiktionary_used_with",
    "wiktionary_property_part",
]

# Unicode-aware lexical candidates so Portuguese and other XNLI languages keep
# accented and non-ASCII alphabetic tokens.
TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", flags=re.UNICODE)
LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "he", "her", "him",
    "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "they", "this", "to", "was",
    "were", "will", "with", "would", "you", "your",
    "a", "ao", "aos", "as", "com", "da", "das", "de", "do", "dos", "e", "em", "na", "nas", "no", "nos", "o", "os", "para", "por", "que", "se",
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


def wiktextract_words(values) -> List[str]:
    """Normalize a Wiktextract linkage field into distinct display terms."""
    result = []
    seen = set()
    for value in values or []:
        word = value.get("word") if isinstance(value, dict) else value
        if not word:
            continue
        normalized = normalize_term(str(word))
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(str(word))
    return result


def compact_wiktextract_entry(row: dict, language: str = "en") -> dict | None:
    """Keep the small lexical subset of one Wiktextract JSON record for PIIG."""
    word = row.get("word")
    if not word or row.get("lang_code") != language or row.get("redirect"):
        return None
    relation_names = ("synonyms", "antonyms", "hypernyms", "hyponyms")
    fallback_linkages = {name: wiktextract_words(row.get(name)) for name in relation_names}
    senses = []
    for sense in row.get("senses") or []:
        glosses = sense.get("glosses") or sense.get("raw_glosses") or []
        gloss = str(glosses[0]).strip() if glosses else ""
        if not gloss:
            continue
        values = {
            name: wiktextract_words(sense.get(name)) or fallback_linkages[name]
            for name in relation_names
        }
        senses.append({"pos": row.get("pos", ""), "gloss": gloss, **values})
        if len(senses) >= 3:
            break
    if not senses:
        return None
    return {
        "term": word,
        "aliases": [word],
        "language": language,
        "senses": senses,
        "verified": True,
        "source": "wiktextract_dump",
    }


def load_wiktionary_cache(path: Path | None) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    cache = {}
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for cache_path in paths:
        with cache_path.open("r", encoding="utf-8") as handle:
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
    if left_entry.get("source") == "pt_lexical_semantics" or right_entry.get("source") == "pt_lexical_semantics":
        for sense in left_entry.get("senses") or []:
            for relation in sense.get("relations") or []:
                if normalize_term(str(relation.get("term", ""))) == right_key:
                    return str(relation.get("relation"))
        return None
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
    if entry.get("source") == "pt_lexical_semantics":
        relation_text = "; ".join(
            f"{item.get('relation', 'relation')} {item.get('term', '')} resources={item.get('resources', 0)}"
            for item in (sense.get("relations") or [])[:3]
        )
        gloss = f"{gloss}; {relation_text}" if relation_text else gloss
    prefix = "PT-LKB entry:" if entry.get("source") == "pt_lexical_semantics" else "wiktionary entry:"
    return " ".join(piece for piece in [prefix, str(term), str(pos), str(gloss)] if piece).strip()


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
