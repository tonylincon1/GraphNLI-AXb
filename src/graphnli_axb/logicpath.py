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


def build_logicpath_graph(sentence1: str, sentence2: str, max_nodes: int = 24) -> LogicPathGraphSpec:
    """Build a pair-conditioned graph of transformations from sentence1 to sentence2."""
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
        edges[index, index] = LOGICPATH_REL_TO_ID["self"]
    edges[0, 1] = edges[1, 0] = LOGICPATH_REL_TO_ID["supports"]
    premise_indices = [index for index, node_type in enumerate(types) if node_type == "premise_unit"]
    hypothesis_indices = [index for index, node_type in enumerate(types) if node_type == "hypothesis_unit"]
    for index in premise_indices:
        edges[0, index] = edges[index, 0] = LOGICPATH_REL_TO_ID["contains"]
    for index in hypothesis_indices:
        edges[1, index] = edges[index, 1] = LOGICPATH_REL_TO_ID["contains"]

    unit_index = {
        ("premise", texts[index].split(":", 1)[-1].strip()): index for index in premise_indices
    }
    unit_index.update({
        ("hypothesis", texts[index].split(":", 1)[-1].strip()): index for index in hypothesis_indices
    })
    for name, left, right, relation, transformation_index in kept_transformations:
        edges[transformation_index, 0] = edges[0, transformation_index] = LOGICPATH_REL_TO_ID["operation"]
        edges[transformation_index, 1] = edges[1, transformation_index] = LOGICPATH_REL_TO_ID["operation"]
        if ("premise", left) in unit_index:
            source = unit_index[("premise", left)]
            edges[source, transformation_index] = edges[transformation_index, source] = LOGICPATH_REL_TO_ID["contradicts" if "contradiction" in name or "flip" in name else "operation"]
        if ("hypothesis", right) in unit_index:
            target = unit_index[("hypothesis", right)]
            edges[target, transformation_index] = edges[transformation_index, target] = LOGICPATH_REL_TO_ID["contradicts" if "contradiction" in name or "flip" in name else "operation"]

    for left in premise_indices:
        for right in hypothesis_indices:
            left_token = texts[left].split(":", 1)[-1].strip()
            right_token = texts[right].split(":", 1)[-1].strip()
            if wordnet_relation(left_token, right_token):
                edges[left, right] = edges[right, left] = LOGICPATH_REL_TO_ID["lexical_relation"]

    return LogicPathGraphSpec(
        texts=texts,
        node_type_ids=[LOGICPATH_NODE_TYPE_TO_ID[node_type] for node_type in types],
        edge_types=edges,
        targets=targets,
        inverse_targets=inverse_targets,
    )
