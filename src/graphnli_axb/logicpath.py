"""Dynamic transformation graphs for the LogicPath Transformer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .graph import (
    CONDITIONALS,
    CONJUNCTIONS,
    DISJUNCTIONS,
    DOWNWARD_MONOTONE,
    NEGATIONS,
    STOPWORDS,
    TEMPORALS,
    UNIVERSALS,
    simple_tokens,
    wordnet_relation,
)
from .wiktionary import build_wiktionary_spec, entry_summary


LOGICPATH_NODE_TYPES = [
    "premise_state",
    "hypothesis_state",
    "premise_unit",
    "hypothesis_unit",
    "transformation",
]
LOGICPATH_NODE_TYPE_TO_ID = {name: index for index, name in enumerate(LOGICPATH_NODE_TYPES)}

LOGICPATH_RELATIONS = [
    "none",
    "self",
    "contains",
    "lexical_relation",
    "supports",
    "contradicts",
    "operation",
]
LOGICPATH_REL_TO_ID = {name: index for index, name in enumerate(LOGICPATH_RELATIONS)}

LOGICPATH_V2_NODE_TYPES = LOGICPATH_NODE_TYPES + [
    "wiktionary_entry",
    "wiktionary_gloss",
]
LOGICPATH_V2_NODE_TYPE_TO_ID = {name: index for index, name in enumerate(LOGICPATH_V2_NODE_TYPES)}
LOGICPATH_V2_RELATIONS = LOGICPATH_RELATIONS + [
    "wiktionary_evidence",
    "wiktionary_relation",
]
LOGICPATH_V2_REL_TO_ID = {name: index for index, name in enumerate(LOGICPATH_V2_RELATIONS)}

TRANSFORMATIONS = [
    "lexical_identity",
    "lexical_synonym",
    "hypernym_generalization",
    "hyponym_specialization",
    "antonym_contradiction",
    "negation_flip",
    "quantifier_change",
    "conjunction_change",
    "conditional_alignment",
    "numeric_change",
    "temporal_change",
]
TRANSFORMATION_TO_ID = {name: index for index, name in enumerate(TRANSFORMATIONS)}


@dataclass(frozen=True)
class LogicPathGraphSpec:
    texts: List[str]
    node_type_ids: List[int]
    edge_types: np.ndarray
    targets: np.ndarray
    inverse_targets: np.ndarray


def _unit_tokens(text: str, limit: int = 4) -> List[str]:
    logical = NEGATIONS | CONDITIONALS | CONJUNCTIONS | DISJUNCTIONS | UNIVERSALS | DOWNWARD_MONOTONE
    result = []
    seen = set()
    for token in simple_tokens(text):
        if token in seen or (token in STOPWORDS and token not in logical):
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def _detect_transformations(left: str, right: str) -> List[tuple[str, str, str, str]]:
    left_tokens = simple_tokens(left)
    right_tokens = simple_tokens(right)
    left_units = _unit_tokens(left, limit=8)
    right_units = _unit_tokens(right, limit=8)
    transformations: List[tuple[str, str, str, str]] = []
    seen = set()
    for left_token in left_units:
        for right_token in right_units:
            relation = wordnet_relation(left_token, right_token)
            mapping = {
                "identity": "lexical_identity",
                "synonym": "lexical_synonym",
                "hypernym": "hypernym_generalization",
                "hyponym": "hyponym_specialization",
                "antonym": "antonym_contradiction",
            }
            name = mapping.get(relation)
            if name and (name, left_token, right_token) not in seen:
                seen.add((name, left_token, right_token))
                transformations.append((name, left_token, right_token, relation or "lexical"))

    checks = [
        ("negation_flip", bool(set(left_tokens) & NEGATIONS) != bool(set(right_tokens) & NEGATIONS), "polarity"),
        ("quantifier_change", bool(set(left_tokens) & (UNIVERSALS | DOWNWARD_MONOTONE)) != bool(set(right_tokens) & (UNIVERSALS | DOWNWARD_MONOTONE)), "quantifier"),
        ("conjunction_change", bool(set(left_tokens) & CONJUNCTIONS) != bool(set(right_tokens) & CONJUNCTIONS), "conjunction"),
        ("conditional_alignment", "if" in left_tokens or "if" in right_tokens, "conditional"),
        ("numeric_change", bool([token for token in left_tokens + right_tokens if token.replace('.', '', 1).isdigit()]), "number"),
        ("temporal_change", bool(set(left_tokens + right_tokens) & TEMPORALS), "temporal"),
    ]
    for name, active, marker in checks:
        if active and (name, marker) not in {(item[0], item[3]) for item in transformations}:
            transformations.append((name, "premise", "hypothesis", marker))
    return transformations


def _base_logicpath_graph(
    sentence1: str,
    sentence2: str,
    max_nodes: int,
    node_types: list[str],
    relation_ids: dict[str, int],
    wiktionary_cache: dict | None = None,
    max_wiktionary_entries: int = 0,
    lexicon_name: str = "Wiktionary",
) -> LogicPathGraphSpec:
    """Build a pair-conditioned transformation graph, optionally with lexical evidence."""
    premise_units = _unit_tokens(sentence1)
    hypothesis_units = _unit_tokens(sentence2)
    transformations = _detect_transformations(sentence1, sentence2)

    texts = ["premise state: " + sentence1, "hypothesis state: " + sentence2]
    types = ["premise_state", "hypothesis_state"]
    for token in premise_units:
        if len(texts) >= max_nodes:
            break
        texts.append(f"premise unit: {token}")
        types.append("premise_unit")
    for token in hypothesis_units:
        if len(texts) >= max_nodes:
            break
        texts.append(f"hypothesis unit: {token}")
        types.append("hypothesis_unit")

    kept_transformations = []
    for name, left, right, relation in transformations:
        if len(texts) >= max_nodes:
            break
        texts.append(f"transformation {name}: {left} -> {right}")
        types.append("transformation")
        kept_transformations.append((name, left, right, relation, len(texts) - 1))

    wiktionary = None
    wiktionary_entries = []
    if wiktionary_cache is not None and max_wiktionary_entries > 0 and len(texts) < max_nodes:
        wiktionary = build_wiktionary_spec(
            sentence1,
            sentence2,
            wiktionary_cache,
            max_entries=max_wiktionary_entries,
        )
        for entry in wiktionary.entries:
            if len(texts) + 2 > max_nodes:
                break
            term = entry.get("matched_term") or entry.get("term") or "term"
            entry_index = len(texts)
            texts.append(f"{lexicon_name} entry: {term}")
            types.append("wiktionary_entry")
            gloss_index = len(texts)
            summary = entry_summary(entry)
            if lexicon_name != "Wiktionary" and summary.startswith("PT-LKB entry:"):
                summary = summary.replace("PT-LKB entry:", f"{lexicon_name} entry:", 1)
            texts.append(summary)
            types.append("wiktionary_gloss")
            wiktionary_entries.append((term, entry_index, gloss_index))

    targets = np.zeros(len(TRANSFORMATIONS), dtype=np.float32)
    for name, *_ in kept_transformations:
        targets[TRANSFORMATION_TO_ID[name]] = 1.0

    inverse_targets = np.zeros(len(TRANSFORMATIONS), dtype=np.float32)
    inverse_names = {
        "hypernym_generalization": "hyponym_specialization",
        "hyponym_specialization": "hypernym_generalization",
    }
    for name, *_ in kept_transformations:
        inverse_targets[TRANSFORMATION_TO_ID[inverse_names.get(name, name)]] = 1.0

    node_count = len(texts)
    edges = np.zeros((node_count, node_count), dtype=np.int64)
    for index in range(node_count):
        edges[index, index] = relation_ids["self"]
    edges[0, 1] = edges[1, 0] = relation_ids["supports"]
    premise_indices = [index for index, node_type in enumerate(types) if node_type == "premise_unit"]
    hypothesis_indices = [index for index, node_type in enumerate(types) if node_type == "hypothesis_unit"]
    for index in premise_indices:
        edges[0, index] = edges[index, 0] = relation_ids["contains"]
    for index in hypothesis_indices:
        edges[1, index] = edges[index, 1] = relation_ids["contains"]

    unit_index = {
        ("premise", texts[index].split(":", 1)[-1].strip()): index for index in premise_indices
    }
    unit_index.update({
        ("hypothesis", texts[index].split(":", 1)[-1].strip()): index for index in hypothesis_indices
    })
    for name, left, right, relation, transformation_index in kept_transformations:
        edges[transformation_index, 0] = edges[0, transformation_index] = relation_ids["operation"]
        edges[transformation_index, 1] = edges[1, transformation_index] = relation_ids["operation"]
        if ("premise", left) in unit_index:
            source = unit_index[("premise", left)]
            edges[source, transformation_index] = edges[transformation_index, source] = relation_ids["contradicts" if "contradiction" in name or "flip" in name else "operation"]
        if ("hypothesis", right) in unit_index:
            target = unit_index[("hypothesis", right)]
            edges[target, transformation_index] = edges[transformation_index, target] = relation_ids["contradicts" if "contradiction" in name or "flip" in name else "operation"]

    for left in premise_indices:
        for right in hypothesis_indices:
            left_token = texts[left].split(":", 1)[-1].strip()
            right_token = texts[right].split(":", 1)[-1].strip()
            if wordnet_relation(left_token, right_token):
                edges[left, right] = edges[right, left] = relation_ids["lexical_relation"]

    # Link cached dictionary evidence to the matching lexical unit and its gloss.
    if wiktionary_entries:
        for term, entry_index, gloss_index in wiktionary_entries:
            for side, unit in unit_index.items():
                if side[1] != term:
                    continue
                edges[unit, entry_index] = edges[entry_index, unit] = relation_ids["wiktionary_evidence"]
            edges[entry_index, gloss_index] = edges[gloss_index, entry_index] = relation_ids["wiktionary_evidence"]
        if wiktionary is not None:
            entry_by_term = {term: index for term, index, _ in wiktionary_entries}
            for pair in wiktionary.pairs:
                left = entry_by_term.get(pair["premise_term"])
                right = entry_by_term.get(pair["hypothesis_term"])
                if left is not None and right is not None:
                    edges[left, right] = edges[right, left] = relation_ids["wiktionary_relation"]

    return LogicPathGraphSpec(
        texts=texts,
        node_type_ids=[{name: index for index, name in enumerate(node_types)}[node_type] for node_type in types],
        edge_types=edges,
        targets=targets,
        inverse_targets=inverse_targets,
    )




def build_logicpath_graph(sentence1: str, sentence2: str, max_nodes: int = 24) -> LogicPathGraphSpec:
    return _base_logicpath_graph(
        sentence1, sentence2, max_nodes, LOGICPATH_NODE_TYPES, LOGICPATH_REL_TO_ID
    )


def build_logicpath_v2_graph(
    sentence1: str,
    sentence2: str,
    wiktionary_cache: dict,
    max_nodes: int = 32,
    max_wiktionary_entries: int = 2,
    lexicon_name: str = "Wiktionary",
) -> LogicPathGraphSpec:
    return _base_logicpath_graph(
        sentence1,
        sentence2,
        max_nodes,
        LOGICPATH_V2_NODE_TYPES,
        LOGICPATH_V2_REL_TO_ID,
        wiktionary_cache=wiktionary_cache,
        max_wiktionary_entries=max_wiktionary_entries,
        lexicon_name=lexicon_name,
    )
