from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np


RELATIONS = [
    "none",
    "self",
    "intra_sentence",
    "identity",
    "synonym",
    "antonym",
    "hypernym",
    "hyponym",
    "negation_scope",
    "quantifier",
    "number_equal",
    "number_conflict",
    "disjunction_scope",
    "conjunction_scope",
    "conditional_scope",
    "temporal_order",
    "universal_scope",
    "downward_monotone_scope",
]
REL_TO_ID = {name: idx for idx, name in enumerate(RELATIONS)}

NODE_TYPES = [
    "premise",
    "hypothesis",
    "negation",
    "quantifier",
    "number",
    "disjunction",
    "conjunction",
    "conditional",
    "temporal",
    "monotone",
]
NODE_TYPE_TO_ID = {name: idx for idx, name in enumerate(NODE_TYPES)}

# These are the operators for which AX-b is especially sensitive to scope and
# sentence direction.  They form the compact, dedicated v8 logic subgraph.
LOGIC_NODE_TYPES = (
    "negation",
    "quantifier",
    "disjunction",
    "conjunction",
    "conditional",
    "monotone",
    "temporal",
)
LOGIC_NODE_TYPE_IDS = tuple(NODE_TYPE_TO_ID[name] for name in LOGIC_NODE_TYPES)

NEGATIONS = {
    "no",
    "not",
    "n't",
    "never",
    "none",
    "nothing",
    "nobody",
    "neither",
    "nor",
    "without",
    "cannot",
    "can't",
    "doesn't",
    "don't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "ne",
    "pas",
    "jamais",
    "nicht",
    "kein",
    "keine",
    "no",
    "nunca",
    "sin",
    "den",
    "μη",
    "δεν",
    "не",
    "нет",
    "لا",
    "لم",
    "لن",
    "ليس",
    "नहीं",
    "न",
    "si",
    "hapana",
    "ไม่",
    "değil",
    "değil",
    "نہیں",
    "نہ",
    "không",
    "chẳng",
    "不",
    "没",
    "無",
}

QUANTIFIERS = {
    "all",
    "some",
    "many",
    "few",
    "most",
    "several",
    "every",
    "each",
    "any",
    "both",
    "none",
    "more",
    "less",
    "fewer",
    "least",
    "much",
    "exactly",
    "only",
}

DISJUNCTIONS = {"or", "either", "nor"}
CONJUNCTIONS = {"and", "both", "also"}
CONDITIONALS = {"if", "then", "unless", "when", "whenever"}
TEMPORALS = {"before", "after", "during", "while", "until", "since", "once", "earlier", "later"}
UNIVERSALS = {"all", "every", "each", "any"}
DOWNWARD_MONOTONE = {"no", "few", "fewer", "less", "at", "most", "without", "unless"}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "with",
    "as",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
}

TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+(?:[.,]\d+)?", re.UNICODE)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


@dataclass(frozen=True)
class GraphSpec:
    node_texts: List[str]
    node_type_ids: List[int]
    node_side_ids: List[int]
    edge_types: np.ndarray
    evidence: np.ndarray


def simple_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0).lower()
        # Chinese has no whitespace segmentation; character nodes retain graph
        # coverage until a dedicated segmenter is introduced.
        if CJK_RE.fullmatch(token):
            tokens.extend(list(token))
        else:
            tokens.append(token)
    return tokens


def node_type(token: str, side: str) -> str:
    if token in NEGATIONS:
        return "negation"
    if token in DISJUNCTIONS:
        return "disjunction"
    if token in CONJUNCTIONS:
        return "conjunction"
    if token in CONDITIONALS:
        return "conditional"
    if token in TEMPORALS:
        return "temporal"
    if token in DOWNWARD_MONOTONE:
        return "monotone"
    if token in QUANTIFIERS:
        return "quantifier"
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return "number"
    return side


def select_anchors(text: str, side: str, max_nodes: int) -> List[Tuple[str, str]]:
    raw = simple_tokens(text)
    anchors: List[Tuple[str, str]] = []
    for token in raw:
        typ = node_type(token, side)
        if typ not in {"premise", "hypothesis"} or token not in STOPWORDS:
            anchors.append((token, typ))
        if len(anchors) >= max_nodes:
            break
    return anchors


@lru_cache(maxsize=20000)
def wordnet_relation(left: str, right: str) -> str | None:
    if left == right:
        return "identity"
    if not left.isascii() or not right.isascii() or not left.isalpha() or not right.isalpha():
        return None
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
    except (ImportError, LookupError):
        return None

    try:
        left_synsets = wn.synsets(left)
        right_synsets = wn.synsets(right)
    except LookupError:
        return None
    if not left_synsets or not right_synsets:
        return None

    right_names = {syn.name() for syn in right_synsets}
    left_names = {syn.name() for syn in left_synsets}
    if left_names & right_names:
        return "synonym"

    for syn in left_synsets[:5]:
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                if ant.name().replace("_", " ").lower() == right:
                    return "antonym"

    right_hyper = set()
    for syn in right_synsets[:4]:
        right_hyper.update(node.name() for node in syn.closure(lambda s: s.hypernyms(), depth=3))
    left_hyper = set()
    for syn in left_synsets[:4]:
        left_hyper.update(node.name() for node in syn.closure(lambda s: s.hypernyms(), depth=3))
    if left_names & right_hyper:
        return "hyponym"
    if right_names & left_hyper:
        return "hypernym"
    return None


def numeric_value(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def scope_relation(node_typ: str) -> str | None:
    if node_typ == "negation":
        return "negation_scope"
    if node_typ == "quantifier":
        return "quantifier"
    if node_typ == "disjunction":
        return "disjunction_scope"
    if node_typ == "conjunction":
        return "conjunction_scope"
    if node_typ == "conditional":
        return "conditional_scope"
    if node_typ == "temporal":
        return "temporal_order"
    if node_typ == "monotone":
        return "downward_monotone_scope"
    return None


def add_scope_edges(edge_types: np.ndarray, nodes: List[Tuple[str, str]], start: int, stop: int) -> None:
    for idx in range(start, stop):
        rel = scope_relation(nodes[idx][1])
        if not rel:
            continue
        rel_id = REL_TO_ID[rel]
        left = max(start, idx - 3)
        right = min(stop, idx + 4)
        for neighbor in range(left, right):
            if neighbor == idx:
                continue
            edge_types[idx, neighbor] = rel_id
            edge_types[neighbor, idx] = rel_id


def build_graph(sentence1: str, sentence2: str, max_nodes: int = 32) -> GraphSpec:
    per_side = max(2, max_nodes // 2)
    premise = select_anchors(sentence1, "premise", per_side)
    hypothesis = select_anchors(sentence2, "hypothesis", per_side)
    nodes = premise + hypothesis
    if not nodes:
        nodes = [("empty", "premise")]

    n = len(nodes)
    edge_types = np.zeros((n, n), dtype=np.int64)
    for idx in range(n):
        edge_types[idx, idx] = REL_TO_ID["self"]

    premise_size = len(premise)
    for start, stop in [(0, premise_size), (premise_size, n)]:
        for idx in range(start, max(start, stop - 1)):
            edge_types[idx, idx + 1] = REL_TO_ID["intra_sentence"]
            edge_types[idx + 1, idx] = REL_TO_ID["intra_sentence"]
        add_scope_edges(edge_types, nodes, start, stop)

    for i, (left, left_type) in enumerate(nodes[:premise_size]):
        for offset, (right, right_type) in enumerate(nodes[premise_size:]):
            j = premise_size + offset
            rel = None
            left_num = numeric_value(left)
            right_num = numeric_value(right)
            if left_num is not None and right_num is not None:
                rel = "number_equal" if left_num == right_num else "number_conflict"
            elif left_type == "temporal" or right_type == "temporal":
                rel = "temporal_order"
            elif left_type == "conditional" or right_type == "conditional":
                rel = "conditional_scope"
            elif left_type == "disjunction" or right_type == "disjunction":
                rel = "disjunction_scope"
            elif left_type == "conjunction" or right_type == "conjunction":
                rel = "conjunction_scope"
            elif left_type == "monotone" or right_type == "monotone":
                rel = "downward_monotone_scope"
            elif left_type in {"negation", "quantifier"} or right_type in {"negation", "quantifier"}:
                rel = scope_relation(left_type) or scope_relation(right_type) or "quantifier"
            elif left in UNIVERSALS or right in UNIVERSALS:
                rel = "universal_scope"
            else:
                rel = wordnet_relation(left, right)
            if rel:
                edge_types[i, j] = REL_TO_ID[rel]
                edge_types[j, i] = REL_TO_ID[rel]

    evidence = np.zeros(len(RELATIONS), dtype=np.float32)
    for rel_id in np.unique(edge_types):
        if rel_id > 0:
            evidence[int(rel_id)] = 1.0

    return GraphSpec(
        node_texts=[token for token, _ in nodes],
        node_type_ids=[NODE_TYPE_TO_ID.get(typ, 0) for _, typ in nodes],
        node_side_ids=[0] * premise_size + [1] * (n - premise_size),
        edge_types=edge_types,
        evidence=evidence,
    )
