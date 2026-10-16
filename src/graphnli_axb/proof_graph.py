from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .graph import (
    CONDITIONALS,
    DISJUNCTIONS,
    DOWNWARD_MONOTONE,
    STOPWORDS,
    UNIVERSALS,
    REL_TO_ID,
    build_graph,
    simple_tokens,
    wordnet_relation,
)
from .wikidata import WIKIDATA_RELATIONS, WikidataSpec


PROOF_NODE_TYPES = [
    "premise_phrase",
    "hypothesis_phrase",
    "lexical_bridge",
    "negation_conflict",
    "upward_generalization",
    "conjunction_elimination",
    "conditional_direction",
    "lexical_generalization_candidate",
    "quantifier_polarity_context",
    "disjunction_scope",
    "conditional_antecedent_observed",
    "conditional_antecedent_unobserved",
    "external_knowledge",
    "external_fact",
    "entity_type_path",
]
PROOF_NODE_TYPE_TO_ID = {name: index for index, name in enumerate(PROOF_NODE_TYPES)}

PROOF_RELATIONS = ["none", "self", "phrase_pair", "phrase_evidence", "phrase_knowledge", "evidence_knowledge"]
PROOF_REL_TO_ID = {name: index for index, name in enumerate(PROOF_RELATIONS)}

PROOF_TARGETS = [
    "lexical_bridge",
    "negation_conflict",
    "upward_generalization",
    "conjunction_elimination",
    "conditional_direction",
    "external_fact_path",
    "entity_type_path",
]
V3_PROOF_TARGETS = PROOF_TARGETS + [
    "lexical_generalization_candidate",
    "quantifier_polarity_context",
    "disjunction_scope",
    "conditional_antecedent_observed",
    "conditional_antecedent_unobserved",
]
PROOF_TARGET_TO_ID = {name: index for index, name in enumerate(PROOF_TARGETS)}
V3_PROOF_TARGET_TO_ID = {name: index for index, name in enumerate(V3_PROOF_TARGETS)}


@dataclass(frozen=True)
class ProofGraphSpec:
    texts: List[str]
    node_type_ids: List[int]
    edge_types: np.ndarray
    targets: np.ndarray


def key_phrase(text: str, limit: int = 10) -> str:
    """A compact phrase anchor keeps the proof graph local and interpretable."""
    tokens = simple_tokens(text)
    return " ".join(tokens[:limit]) if tokens else "empty"


def content_tokens(tokens: List[str]) -> set[str]:
    return {
        token
        for token in tokens
        if token not in STOPWORDS | CONDITIONALS | DISJUNCTIONS | UNIVERSALS | DOWNWARD_MONOTONE
    }


def overlap_ratio(left: List[str], right: List[str]) -> float:
    left_content = content_tokens(left)
    right_content = content_tokens(right)
    if not left_content:
        return 0.0
    return len(left_content & right_content) / len(left_content)


def conditional_overlap_ratio(left: List[str], right: List[str]) -> float:
    """Keep auxiliary verbs: they often carry an antecedent's predicate."""
    predicate_tokens = {"has", "have", "had", "does", "do", "did"}
    ignored = (STOPWORDS - predicate_tokens) | CONDITIONALS | DISJUNCTIONS | UNIVERSALS | DOWNWARD_MONOTONE
    left_content = {token for token in left if token not in ignored}
    right_content = {token for token in right if token not in ignored}
    if not left_content:
        return 0.0
    return len(left_content & right_content) / len(left_content)


def conditional_parts(tokens: List[str]) -> tuple[List[str], List[str]] | None:
    """Extract a simple if/then antecedent and consequent when available."""
    if "if" not in tokens:
        return None
    if_index = tokens.index("if")
    if "then" in tokens[if_index + 1 :]:
        then_index = tokens.index("then", if_index + 1)
        return tokens[if_index + 1 : then_index], tokens[then_index + 1 :]
    return tokens[if_index + 1 :], []


def conditional_status(source_text: str, other_text: str) -> str | None:
    parts = conditional_parts(simple_tokens(source_text))
    if not parts:
        return None
    antecedent, consequent = parts
    other_tokens = simple_tokens(other_text)
    # A strong consequent alignment paired with weak antecedent support is the
    # invalid modus-ponens pattern seen in AX-b conditional diagnostics.
    if consequent and conditional_overlap_ratio(consequent, other_tokens) >= 0.5:
        return "observed" if conditional_overlap_ratio(antecedent, other_tokens) >= 0.75 else "unobserved"
    return None


def quantifier_context(sentence1: str, sentence2: str) -> List[str]:
    contexts = []
    for side, text in (("premise", sentence1), ("hypothesis", sentence2)):
        token_list = simple_tokens(text)
        tokens = set(token_list)
        # `at` is only downward-sensitive as part of the multiword quantifier
        # "at most"; treating "at random" as monotonicity was a false cue.
        downward_tokens = DOWNWARD_MONOTONE - {"at", "most"}
        has_at_most = any(left == "at" and right == "most" for left, right in zip(token_list, token_list[1:]))
        if tokens & downward_tokens or has_at_most:
            contexts.append(f"{side} downward quantifier: restrictor polarity is downward")
        elif tokens & UNIVERSALS:
            contexts.append(f"{side} universal quantifier: restrictor downward, scope upward")
    return contexts


def disjunction_status(source_text: str, other_text: str) -> str | None:
    tokens = simple_tokens(source_text)
    disjunction_positions = [index for index, token in enumerate(tokens) if token in DISJUNCTIONS]
    if not disjunction_positions:
        return None
    index = disjunction_positions[-1]
    left_branch = tokens[:index]
    right_branch = tokens[index + 1 :]
    other_tokens = simple_tokens(other_text)
    if max(overlap_ratio(left_branch, other_tokens), overlap_ratio(right_branch, other_tokens)) >= 0.5:
        return "selected_branch"
    return "unresolved"


def build_proof_graph(
    sentence1: str,
    sentence2: str,
    wikidata: WikidataSpec,
    max_nodes: int = 12,
    knowledge_mode: str = "v1",
) -> ProofGraphSpec:
    lexical = build_graph(sentence1, sentence2, max_nodes=32)
    texts = [f"premise phrase: {key_phrase(sentence1)}", f"hypothesis phrase: {key_phrase(sentence2)}"]
    types = ["premise_phrase", "hypothesis_phrase"]
    target_names = V3_PROOF_TARGETS if knowledge_mode == "v3" else PROOF_TARGETS
    target_to_id = V3_PROOF_TARGET_TO_ID if knowledge_mode == "v3" else PROOF_TARGET_TO_ID
    targets = np.zeros(len(target_names), dtype=np.float32)

    premise_count = sum(side == 0 for side in lexical.node_side_ids)
    cross_edges = lexical.edge_types[:premise_count, premise_count:]
    cross_relation_ids = set(np.unique(cross_edges).tolist())
    has_bridge = bool(cross_relation_ids & {REL_TO_ID[name] for name in ["identity", "synonym", "antonym", "hypernym", "hyponym"]})
    premise_types = set(lexical.node_type_ids[:premise_count])
    hypothesis_types = set(lexical.node_type_ids[premise_count:])

    def add(name: str, text: str) -> None:
        if len(texts) >= max_nodes:
            return
        texts.append(text)
        types.append(name)
        if name in target_to_id:
            targets[target_to_id[name]] = 1.0

    if has_bridge:
        add("lexical_bridge", "evidence lexical bridge across premise and hypothesis")
    negation_id = 2
    conjunction_id = 6
    conditional_id = 7
    if (negation_id in premise_types) != (negation_id in hypothesis_types):
        add("negation_conflict", "evidence polarity contrast from negation scope")
    # In the existing lexical graph the relation name is anchored on the
    # right-hand concept: premise dog -> hypothesis animal is `hypernym`.
    has_upward_generalization = REL_TO_ID["hypernym"] in cross_relation_ids
    if not has_upward_generalization:
        premise_tokens = lexical.node_texts[:premise_count]
        hypothesis_tokens = lexical.node_texts[premise_count:]
        for left in premise_tokens:
            for right in hypothesis_tokens:
                # The base graph stays surface-form conservative; this local
                # normalization only lets a proof candidate recover dog->animal.
                left_lemma = left[:-1] if left.endswith("s") and len(left) > 3 else left
                right_lemma = right[:-1] if right.endswith("s") and len(right) > 3 else right
                if wordnet_relation(left_lemma, right_lemma) == "hypernym":
                    has_upward_generalization = True
                    break
            if has_upward_generalization:
                break
    if has_upward_generalization:
        if knowledge_mode == "v3":
            add(
                "lexical_generalization_candidate",
                "evidence lexical generalization candidate; validity depends on quantifier polarity",
            )
        else:
            add("upward_generalization", "evidence upward generalization through hyponym to hypernym")
    if conjunction_id in premise_types and has_bridge:
        add("conjunction_elimination", "evidence conjunction elimination from premise structure")
    if conditional_id in premise_types or conditional_id in hypothesis_types:
        add("conditional_direction", "evidence conditional direction requires an observed antecedent")

    if knowledge_mode == "v3":
        contexts = quantifier_context(sentence1, sentence2)
        if contexts:
            add("quantifier_polarity_context", f"evidence quantifier polarity context: {'; '.join(contexts)}")
        for source, other in ((sentence1, sentence2), (sentence2, sentence1)):
            status = conditional_status(source, other)
            if status == "observed":
                add("conditional_antecedent_observed", "evidence conditional consequent aligns and antecedent is observed")
                break
            if status == "unobserved":
                add("conditional_antecedent_unobserved", "evidence conditional consequent aligns but antecedent is unobserved")
                break
        for source, other in ((sentence1, sentence2), (sentence2, sentence1)):
            status = disjunction_status(source, other)
            if status == "selected_branch":
                add("disjunction_scope", "evidence disjunction scope: one alternative aligns; disjunction alone does not entail that alternative")
                break
            if status == "unresolved":
                add("disjunction_scope", "evidence disjunction scope: alternatives require contextual resolution")
                break

    has_type = bool(wikidata.relations[WIKIDATA_RELATIONS.index("wikidata_has_type")])
    same_entity = bool(wikidata.relations[WIKIDATA_RELATIONS.index("wikidata_same_entity")])
    entries = wikidata.entries or [None] * len(wikidata.texts)
    for text, active, entry in zip(wikidata.texts, wikidata.mask, entries):
        if active and len(texts) < max_nodes:
            add("external_knowledge", f"external knowledge: {text}")
        if knowledge_mode == "v2" and active and entry:
            for fact in (entry.get("facts") or [])[:3]:
                add("external_fact", f"external fact: {entry.get('label', 'entity')} {fact.get('property', '')} {fact.get('value', '')}")
                targets[target_to_id["external_fact_path"]] = 1.0
    if has_type and same_entity:
        add("entity_type_path", "evidence entity type path from linked external knowledge")

    node_count = len(texts)
    edges = np.zeros((node_count, node_count), dtype=np.int64)
    for index in range(node_count):
        edges[index, index] = PROOF_REL_TO_ID["self"]
    edges[0, 1] = edges[1, 0] = PROOF_REL_TO_ID["phrase_pair"]
    for index, node_type in enumerate(types[2:], start=2):
        relation = "phrase_knowledge" if node_type in {"external_knowledge", "external_fact", "entity_type_path"} else "phrase_evidence"
        rel_id = PROOF_REL_TO_ID[relation]
        edges[0, index] = edges[index, 0] = rel_id
        edges[1, index] = edges[index, 1] = rel_id
    for left in range(2, node_count):
        for right in range(left + 1, node_count):
            if types[left] in {"external_knowledge", "external_fact"} or types[right] in {"external_knowledge", "external_fact"}:
                edges[left, right] = edges[right, left] = PROOF_REL_TO_ID["evidence_knowledge"]

    return ProofGraphSpec(
        texts=texts,
        node_type_ids=[PROOF_NODE_TYPE_TO_ID[node_type] for node_type in types],
        edge_types=edges,
        targets=targets,
    )
