from __future__ import annotations

import re
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
from .wiktionary import WiktionarySpec, entry_summary


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
    "wiktionary_entry",
    "wiktionary_gloss",
    "wiktionary_relation",
    "negation_scope_evidence",
    "quantifier_polarity_evidence",
    "monotonicity_direction",
    "disjunction_scope_evidence",
    "numeric_relation_evidence",
    "directed_lexical_relation",
    "premise_word",
    "hypothesis_word",
]
PROOF_NODE_TYPE_TO_ID = {name: index for index, name in enumerate(PROOF_NODE_TYPES)}

PROOF_RELATIONS = [
    "none",
    "self",
    "phrase_pair",
    "phrase_evidence",
    "phrase_knowledge",
    "evidence_knowledge",
    "phrase_lexicon",
    "evidence_lexicon",
    "phrase_word",
    "word_alignment",
    "word_lexicon",
    "word_evidence",
    "typed_support",
    "typed_contradiction",
]
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
V4_PROOF_TARGETS = PROOF_TARGETS + [
    "wiktionary_entry_path",
    "wiktionary_gloss_path",
    "wiktionary_relation_path",
]
V5_PROOF_TARGETS = V4_PROOF_TARGETS + [
    "negation_scope_path",
    "quantifier_polarity_path",
    "monotonicity_direction_path",
    "disjunction_scope_path",
    "numeric_relation_path",
    "directed_lexical_relation_path",
]
PROOF_TARGET_TO_ID = {name: index for index, name in enumerate(PROOF_TARGETS)}
V3_PROOF_TARGET_TO_ID = {name: index for index, name in enumerate(V3_PROOF_TARGETS)}
V4_PROOF_TARGET_TO_ID = {name: index for index, name in enumerate(V4_PROOF_TARGETS)}
V5_PROOF_TARGET_TO_ID = {name: index for index, name in enumerate(V5_PROOF_TARGETS)}
V6_PROOF_TARGETS = V4_PROOF_TARGETS + [
    "hybrid_word_path",
    "word_alignment_path",
    "typed_support_path",
    "typed_contradiction_path",
]
V6_PROOF_TARGET_TO_ID = {name: index for index, name in enumerate(V6_PROOF_TARGETS)}


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


def select_v6_word_nodes(sentence1: str, sentence2: str, lexical, limit: int = 4) -> List[tuple[str, str]]:
    """Select a small, pair-conditioned set of word anchors for the hybrid graph."""
    premise_count = sum(side == 0 for side in lexical.node_side_ids)
    cross_edges = lexical.edge_types[:premise_count, premise_count:]
    lexical_relations = {
        REL_TO_ID[name] for name in ("identity", "synonym", "antonym", "hypernym", "hyponym")
    }
    cross_tokens: set[str] = set()
    for left_index, row in enumerate(cross_edges):
        for right_index, relation_id in enumerate(row):
            if int(relation_id) in lexical_relations:
                cross_tokens.add(lexical.node_texts[left_index])
                cross_tokens.add(lexical.node_texts[premise_count + right_index])

    logical_anchors = {
        "not", "never", "no", "none", "nothing", "every", "all", "each", "some", "any",
        "if", "then", "or", "and",
    }
    candidates: List[tuple[int, int, str, str]] = []
    for side, text in (("premise_word", sentence1), ("hypothesis_word", sentence2)):
        seen: set[str] = set()
        for position, token in enumerate(simple_tokens(text)):
            if token in seen or (token in STOPWORDS and token not in logical_anchors):
                continue
            seen.add(token)
            score = 0
            if token in cross_tokens:
                score += 3
            if token in logical_anchors:
                score += 2
            if token not in STOPWORDS:
                score += 1
            candidates.append((score, -position, side, token))

    selected: List[tuple[str, str]] = []
    for side in ("premise_word", "hypothesis_word"):
        side_candidates = sorted(
            (candidate for candidate in candidates if candidate[2] == side),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        selected.extend((side, token) for _, _, _, token in side_candidates[: max(1, limit // 2)])
    return selected[:limit]


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
    wiktionary: WiktionarySpec | None = None,
    max_nodes: int = 12,
    knowledge_mode: str = "v1",
) -> ProofGraphSpec:
    lexical = build_graph(sentence1, sentence2, max_nodes=32)
    texts = [f"premise phrase: {key_phrase(sentence1)}", f"hypothesis phrase: {key_phrase(sentence2)}"]
    types = ["premise_phrase", "hypothesis_phrase"]
    if knowledge_mode == "v3":
        target_names, target_to_id = V3_PROOF_TARGETS, V3_PROOF_TARGET_TO_ID
    elif knowledge_mode == "v4":
        target_names, target_to_id = V4_PROOF_TARGETS, V4_PROOF_TARGET_TO_ID
    elif knowledge_mode == "v5":
        target_names, target_to_id = V5_PROOF_TARGETS, V5_PROOF_TARGET_TO_ID
    elif knowledge_mode == "v6":
        target_names, target_to_id = V6_PROOF_TARGETS, V6_PROOF_TARGET_TO_ID
    else:
        target_names, target_to_id = PROOF_TARGETS, PROOF_TARGET_TO_ID
    targets = np.zeros(len(target_names), dtype=np.float32)

    premise_count = sum(side == 0 for side in lexical.node_side_ids)
    cross_edges = lexical.edge_types[:premise_count, premise_count:]
    cross_relation_ids = set(np.unique(cross_edges).tolist())
    has_bridge = bool(cross_relation_ids & {REL_TO_ID[name] for name in ["identity", "synonym", "antonym", "hypernym", "hyponym"]})
    premise_types = set(lexical.node_type_ids[:premise_count])
    hypothesis_types = set(lexical.node_type_ids[premise_count:])

    def add(name: str, text: str) -> bool:
        if len(texts) >= max_nodes:
            return False
        texts.append(text)
        types.append(name)
        if name in target_to_id:
            targets[target_to_id[name]] = 1.0
        return True

    if knowledge_mode == "v6":
        word_nodes = select_v6_word_nodes(sentence1, sentence2, lexical)
        for node_type, token in word_nodes:
            if add(node_type, f"{node_type.replace('_', ' ')}: {token}"):
                targets[target_to_id["hybrid_word_path"]] = 1.0

    # Glosses expose lexical meaning as evidence tokens. Their presence does
    # not assert entailment: cross-attention learns whether a cached sense is
    # relevant to this particular premise-hypothesis pair.
    if knowledge_mode in {"v4", "v5", "v6"} and wiktionary:
        for entry in wiktionary.entries:
            if len(texts) >= max_nodes:
                break
            term = entry.get("matched_term") or entry.get("term") or "term"
            if add("wiktionary_entry", entry_summary(entry)):
                targets[target_to_id["wiktionary_entry_path"]] = 1.0
            senses = entry.get("senses") or []
            if senses and senses[0].get("gloss"):
                if add("wiktionary_gloss", f"wiktionary gloss: {term}: {senses[0]['gloss']}"):
                    targets[target_to_id["wiktionary_gloss_path"]] = 1.0
        for pair in wiktionary.pairs:
            if add(
                "wiktionary_relation",
                f"wiktionary relation: {pair['premise_term']} {pair['relation']} {pair['hypothesis_term']}",
            ):
                targets[target_to_id["wiktionary_relation_path"]] = 1.0

    if knowledge_mode == "v5":
        premise_tokens = simple_tokens(sentence1)
        hypothesis_tokens = simple_tokens(sentence2)
        premise_polarity = "negated" if 2 in premise_types else "affirmed"
        hypothesis_polarity = "negated" if 2 in hypothesis_types else "affirmed"
        if 2 in premise_types or 2 in hypothesis_types:
            add(
                "negation_scope_evidence",
                f"evidence negation scope: premise is {premise_polarity}; hypothesis is {hypothesis_polarity}",
            )
        contexts = quantifier_context(sentence1, sentence2)
        if contexts:
            add("quantifier_polarity_evidence", f"evidence quantifier polarity: {'; '.join(contexts)}")
        monotone_tokens = sorted((set(premise_tokens) | set(hypothesis_tokens)) & (UNIVERSALS | DOWNWARD_MONOTONE))
        if monotone_tokens:
            direction = "downward" if set(monotone_tokens) & DOWNWARD_MONOTONE else "upward"
            add(
                "monotonicity_direction",
                f"evidence monotonicity direction: {direction} under {' '.join(monotone_tokens)}",
            )
        for source, other in ((sentence1, sentence2), (sentence2, sentence1)):
            status = disjunction_status(source, other)
            if status:
                add(
                    "disjunction_scope_evidence",
                    f"evidence disjunction scope: {status}; alternatives are not interchangeable",
                )
                break
        premise_numbers = re.findall(r"\d+(?:\.\d+)?", sentence1)
        hypothesis_numbers = re.findall(r"\d+(?:\.\d+)?", sentence2)
        if premise_numbers or hypothesis_numbers:
            add(
                "numeric_relation_evidence",
                f"evidence numeric relation: premise numbers {premise_numbers or ['none']}; hypothesis numbers {hypothesis_numbers or ['none']}",
            )
        relation_names = [
            name for name in ("identity", "synonym", "antonym", "hypernym", "hyponym")
            if REL_TO_ID[name] in cross_relation_ids
        ]
        if relation_names:
            add(
                "directed_lexical_relation",
                f"evidence directed lexical relation from premise to hypothesis: {', '.join(relation_names)}",
            )

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

    if knowledge_mode not in {"v4", "v5", "v6"}:
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
        if node_type in {"external_knowledge", "external_fact", "entity_type_path"}:
            relation = "phrase_knowledge"
        elif node_type in {"wiktionary_entry", "wiktionary_gloss", "wiktionary_relation"}:
            relation = "phrase_lexicon"
        elif node_type in {"premise_word", "hypothesis_word"}:
            relation = "phrase_word"
        else:
            relation = "phrase_evidence"
        rel_id = PROOF_REL_TO_ID[relation]
        edges[0, index] = edges[index, 0] = rel_id
        edges[1, index] = edges[index, 1] = rel_id
    for left in range(2, node_count):
        for right in range(left + 1, node_count):
            if types[left] in {"external_knowledge", "external_fact"} or types[right] in {"external_knowledge", "external_fact"}:
                edges[left, right] = edges[right, left] = PROOF_REL_TO_ID["evidence_knowledge"]
            elif types[left] in {"wiktionary_entry", "wiktionary_gloss", "wiktionary_relation"} or types[right] in {"wiktionary_entry", "wiktionary_gloss", "wiktionary_relation"}:
                edges[left, right] = edges[right, left] = PROOF_REL_TO_ID["evidence_lexicon"]

    if knowledge_mode == "v6":
        word_types = {"premise_word", "hypothesis_word"}
        lexicon_types = {"wiktionary_entry", "wiktionary_gloss", "wiktionary_relation"}
        evidence_types = set(types) - {"premise_phrase", "hypothesis_phrase"} - word_types - lexicon_types
        word_indices = [index for index, node_type in enumerate(types) if node_type in word_types]
        lexicon_indices = [index for index, node_type in enumerate(types) if node_type in lexicon_types]
        evidence_indices = [index for index, node_type in enumerate(types) if node_type in evidence_types]
        word_relation_found = False
        for index in word_indices:
            phrase_index = 0 if types[index] == "premise_word" else 1
            edges[index, phrase_index] = edges[phrase_index, index] = PROOF_REL_TO_ID["phrase_word"]
            for lexicon_index in lexicon_indices:
                edges[index, lexicon_index] = edges[lexicon_index, index] = PROOF_REL_TO_ID["word_lexicon"]
            for evidence_index in evidence_indices:
                edges[index, evidence_index] = edges[evidence_index, index] = PROOF_REL_TO_ID["word_evidence"]

        for left in word_indices:
            if types[left] != "premise_word":
                continue
            for right in word_indices:
                if types[right] != "hypothesis_word":
                    continue
                left_token = texts[left].split(":", 1)[-1].strip()
                right_token = texts[right].split(":", 1)[-1].strip()
                relation = wordnet_relation(left_token, right_token)
                if relation == "identity":
                    edge_relation = "word_alignment"
                    target_name = "word_alignment_path"
                elif relation in {"synonym", "hypernym", "hyponym"}:
                    edge_relation = "typed_support"
                    target_name = "typed_support_path"
                elif relation == "antonym":
                    edge_relation = "typed_contradiction"
                    target_name = "typed_contradiction_path"
                else:
                    continue
                edges[left, right] = edges[right, left] = PROOF_REL_TO_ID[edge_relation]
                targets[target_to_id[target_name]] = 1.0
                word_relation_found = True
        if word_relation_found:
            targets[target_to_id["word_alignment_path"]] = 1.0

    return ProofGraphSpec(
        texts=texts,
        node_type_ids=[PROOF_NODE_TYPE_TO_ID[node_type] for node_type in types],
        edge_types=edges,
        targets=targets,
    )
