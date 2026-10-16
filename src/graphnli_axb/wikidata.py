from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


WIKIDATA_RELATIONS = [
    "wikidata_entity_present",
    "wikidata_same_entity",
    "wikidata_premise_only",
    "wikidata_hypothesis_only",
    "wikidata_has_description",
    "wikidata_has_type",
]

# High-precision local linker candidates: title-cased multiword names and
# acronym/mixed-case forms such as NASA and IMDb.  Sentence-initial common
# words are deliberately excluded.
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}[A-Za-z]*)\b")


@dataclass(frozen=True)
class WikidataSpec:
    texts: List[str]
    mask: List[bool]
    relations: np.ndarray
    entries: List[dict | None] | None = None


def normalize_entity(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def extract_entity_candidates(text: str, max_entities: int) -> List[str]:
    seen = set()
    entities = []
    for match in ENTITY_RE.finditer(text):
        entity = " ".join(match.group(0).split())
        key = normalize_entity(entity)
        if len(entity) < 2 or key in seen:
            continue
        seen.add(key)
        entities.append(entity)
        if len(entities) >= max_entities:
            break
    return entities


def load_wikidata_cache(path: Path | None) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label") or row.get("title") or row.get("entity")
            if not label:
                continue
            cache[normalize_entity(label)] = row
        for alias in row.get("aliases") or []:
            cache[normalize_entity(alias)] = row
    return cache


def cache_entry_text(label: str, entry: dict | None) -> str:
    if not entry:
        return f"wikidata entity: {label}"
    qid = entry.get("qid") or entry.get("id") or ""
    description = entry.get("description") or ""
    instance_of = entry.get("instance_of") or entry.get("types") or []
    subclass_of = entry.get("subclass_of") or []
    facts = entry.get("facts") or []
    if isinstance(instance_of, str):
        instance_of = [instance_of]
    if isinstance(subclass_of, str):
        subclass_of = [subclass_of]
    pieces = [
        "wikidata entity:",
        label,
        qid,
        description,
        "instance of",
        " ".join(map(str, instance_of[:4])),
        "subclass of",
        " ".join(map(str, subclass_of[:4])),
        "facts",
        " ".join(f"{fact.get('property', '')} {fact.get('value', '')}" for fact in facts[:4]),
    ]
    return " ".join(piece for piece in pieces if piece).strip()


def build_wikidata_spec(
    sentence1: str,
    sentence2: str,
    cache: Dict[str, dict],
    max_entities: int,
) -> WikidataSpec:
    premise_entities = extract_entity_candidates(sentence1, max_entities)
    hypothesis_entities = extract_entity_candidates(sentence2, max_entities)
    premise_keys = {normalize_entity(entity) for entity in premise_entities}
    hypothesis_keys = {normalize_entity(entity) for entity in hypothesis_entities}
    all_entities = []
    seen = set()
    for entity in premise_entities + hypothesis_entities:
        key = normalize_entity(entity)
        if key in seen:
            continue
        seen.add(key)
        all_entities.append(entity)
        if len(all_entities) >= max_entities:
            break

    relations = np.zeros(len(WIKIDATA_RELATIONS), dtype=np.float32)
    if all_entities:
        relations[WIKIDATA_RELATIONS.index("wikidata_entity_present")] = 1.0
    if premise_keys & hypothesis_keys:
        relations[WIKIDATA_RELATIONS.index("wikidata_same_entity")] = 1.0
    if premise_keys - hypothesis_keys:
        relations[WIKIDATA_RELATIONS.index("wikidata_premise_only")] = 1.0
    if hypothesis_keys - premise_keys:
        relations[WIKIDATA_RELATIONS.index("wikidata_hypothesis_only")] = 1.0

    texts = []
    mask = []
    entries = []
    for entity in all_entities:
        entry = cache.get(normalize_entity(entity))
        if entry and entry.get("description"):
            relations[WIKIDATA_RELATIONS.index("wikidata_has_description")] = 1.0
        if entry and (entry.get("instance_of") or entry.get("types") or entry.get("subclass_of")):
            relations[WIKIDATA_RELATIONS.index("wikidata_has_type")] = 1.0
        texts.append(cache_entry_text(entity, entry))
        mask.append(bool(entry and entry.get("verified", False)))
        entries.append(entry)

    if not texts:
        texts = ["wikidata entity: none"]
        mask = [False]
        entries = [None]

    return WikidataSpec(texts=texts[:max_entities], mask=mask[:max_entities], relations=relations, entries=entries[:max_entities])
