"""Language-aware token-level evidence graphs for the experiment_0008 models.

Earlier graph builders (``graph.py``/``logicpath.py``) were designed for
English: their operator lexicons do not contain Portuguese ``não``/``nenhum``,
treat the Portuguese contraction ``no`` ("in the") as a negation, keep
Portuguese function words as lexical units, and query English WordNet. On the
Portuguese corpora this produced spurious ``negation_flip`` and
``quantifier_change`` signals in roughly 12% of the pairs, while real lexical
relations were found in only about 2% of them.

This module builds the graph directly over the *subword tokens* of the joint
premise/hypothesis encoding, so models can attach symbolic evidence to
contextual Transformer states instead of to bags of static embeddings:

* ``tg_seq``      sequence id per token (0 premise, 1 hypothesis, -1 other);
* ``tg_node``     1 on the first subword of every content word or operator;
* ``tg_op``       logical-operator type of the token's word;
* ``tg_rel``      typed token-to-token relation matrix (row -> column).

Cross-sentence relations come from exact/stem matching, number comparison,
and the local PT-LKB / Wiktionary caches. Within-sentence ``scope`` edges
link an operator (negation, quantifier, conditional, ...) to the words in its
right-hand scope.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch


TOKGRAPH_RELATIONS = [
    "none",
    "same_word",        # subwords of the same word
    "identical",        # same surface form across sentences
    "same_stem",        # shared light stem across sentences
    "synonym",
    "antonym",
    "hypernym",         # column word is more general than row word (row ⊏ column)
    "hyponym",          # column word is more specific than row word (row ⊐ column)
    "number_equal",
    "number_conflict",
    "scope_out",        # row is an operator, column is in its scope
    "scope_in",         # row is in the scope of the column operator
]
TOKGRAPH_REL_TO_ID = {name: index for index, name in enumerate(TOKGRAPH_RELATIONS)}
CROSS_RELATION_IDS = tuple(
    TOKGRAPH_REL_TO_ID[name]
    for name in ("identical", "same_stem", "synonym", "antonym", "hypernym", "hyponym", "number_equal", "number_conflict")
)
INVERSE_RELATION = {
    "hypernym": "hyponym",
    "hyponym": "hypernym",
    "scope_out": "scope_in",
    "scope_in": "scope_out",
}

OP_TYPES = [
    "none",
    "negation",
    "universal",
    "existential",
    "downward",
    "restrictive",
    "conditional",
    "disjunction",
    "conjunction",
    "comparative",
    "modal",
    "temporal",
    "number",
]
OP_TO_ID = {name: index for index, name in enumerate(OP_TYPES)}
# Operators whose right-hand scope matters for monotonicity/polarity.
SCOPED_OPS = {"negation", "universal", "existential", "downward", "restrictive", "conditional", "modal"}
# Operators that are always graph nodes even if they look like function words.
NODE_OPS = {
    "negation", "universal", "existential", "downward", "restrictive", "conditional",
    "disjunction", "comparative", "modal", "number",
}

PT_OPERATORS: Dict[str, Set[str]] = {
    "negation": {
        "não", "nao", "nem", "nunca", "jamais", "ninguém", "ninguem", "nada", "nenhum", "nenhuma",
        "nenhuns", "nenhumas", "sem", "tampouco",
    },
    "universal": {"todo", "toda", "todos", "todas", "cada", "qualquer", "quaisquer", "sempre", "ambos", "ambas"},
    "existential": {
        "algum", "alguma", "alguns", "algumas", "alguém", "alguem", "algo", "vários", "várias", "varios",
        "varias", "diversos", "diversas", "muitos", "muitas", "certos", "certas",
    },
    "downward": {"poucos", "poucas", "pouco", "pouca", "raramente", "menos", "dificilmente"},
    "restrictive": {"apenas", "somente", "só", "exclusivamente", "unicamente"},
    "conditional": {"se", "caso", "senão", "senao"},
    "disjunction": {"ou", "quer"},
    "conjunction": {"e", "também", "tambem"},
    "comparative": {"mais", "maior", "maiores", "menor", "menores", "melhor", "melhores", "pior", "piores"},
    "modal": {
        "talvez", "provavelmente", "possivelmente", "pode", "podem", "poderia", "poderiam", "deve",
        "devem", "deveria", "deveriam", "supostamente",
    },
    "temporal": {"antes", "depois", "durante", "enquanto", "após", "apos", "até", "ate", "desde", "já", "ainda"},
}
EN_OPERATORS: Dict[str, Set[str]] = {
    "negation": {
        "no", "not", "n't", "never", "none", "nothing", "nobody", "neither", "nor", "without", "cannot",
    },
    "universal": {"all", "every", "each", "any", "always", "both"},
    "existential": {"some", "several", "many", "a few", "someone", "something", "various"},
    "downward": {"few", "fewer", "less", "rarely", "seldom", "hardly"},
    "restrictive": {"only", "just", "solely", "exclusively"},
    "conditional": {"if", "unless", "provided"},
    "disjunction": {"or", "either"},
    "conjunction": {"and", "also"},
    "comparative": {"more", "most", "larger", "smaller", "better", "worse", "bigger", "greater"},
    "modal": {"may", "might", "could", "possibly", "probably", "perhaps", "should", "must"},
    "temporal": {"before", "after", "during", "while", "until", "since", "already", "still"},
}

PT_STOPWORDS = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "num", "numa", "por", "pelo", "pela", "pelos", "pelas", "para", "pra", "com", "ao",
    "aos", "à", "às", "que", "é", "são", "está", "estão", "estava", "estavam", "foi", "foram", "ser",
    "estar", "seu", "sua", "seus", "suas", "lhe", "lhes", "me", "te", "isso", "isto", "esse", "essa",
    "este", "esta", "esses", "essas", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "dele",
    "dela", "deles", "delas", "onde", "como", "qual", "quais", "cujo", "cuja", "mas", "porém",
    "se",  # reflexive clitic ("está se alimentando"); clause-initial "se" is re-tagged as conditional
}
# Portuguese words that are conditionals only at the start of a clause
# ("Se chover, ..." / "..., caso ele venha") and otherwise a reflexive
# pronoun or a noun ("está se alimentando", "o caso").
PT_CLAUSE_INITIAL_CONDITIONALS = {"se", "caso"}
PT_CLAUSE_OPENERS = {"e", "mas", "ou", "que", "porque", "pois", "só", "somente", "apenas", "mesmo"}
EN_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with", "as", "that", "this",
    "these", "those", "it", "its", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "his", "her", "their", "our", "which", "who", "whom", "but",
}

PT_NUMBER_WORDS = {
    "dois": 2, "duas": 2, "três": 3, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6, "sete": 7, "oito": 8,
    "nove": 9, "dez": 10, "onze": 11, "doze": 12, "treze": 13, "quinze": 15, "vinte": 20, "trinta": 30,
    "quarenta": 40, "cinquenta": 50, "cem": 100, "cento": 100, "mil": 1000, "dezenas": 20, "centenas": 200,
}
EN_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "hundred": 100, "thousand": 1000, "dozen": 12,
}

PUNCTUATION_RE = re.compile(r"^[^\w]+$", re.UNICODE)
NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)*$")
SCOPE_WINDOW = 6


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


@lru_cache(maxsize=200000)
def light_stem(word: str, language: str = "pt") -> str:
    """A deliberately small suffix stripper (no external lemmatizer needed)."""
    w = strip_accents(word.lower())
    if len(w) <= 3:
        return w
    if language == "pt":
        for suffix, repl in (
            ("mente", ""), ("coes", "c"), ("cao", "c"), ("oes", "o"), ("aes", "a"), ("ais", "al"),
            ("eis", "el"), ("ns", "m"), ("ando", ""), ("endo", ""), ("indo", ""), ("ados", ""),
            ("adas", ""), ("idos", ""), ("idas", ""), ("ado", ""), ("ada", ""), ("ido", ""), ("ida", ""),
            ("res", "r"), ("zes", "z"), ("ses", "s"), ("s", ""),
        ):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[: len(w) - len(suffix)] + repl
                break
        if len(w) > 4 and w[-1] in "aoe":
            w = w[:-1]
        return w
    for suffix in ("ingly", "edly", "ing", "ies", "ed", "es", "ly", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: len(w) - len(suffix)]
    return w


def lemma_candidates(word: str, language: str = "pt") -> List[str]:
    """Cheap lemma guesses used only for dictionary lookup."""
    w = word.lower()
    out = [w]
    if language == "pt":
        rules = [
            ("ões", "ão"), ("ães", "ão"), ("ais", "al"), ("éis", "el"), ("eis", "el"), ("ns", "m"),
            ("ses", "s"), ("res", "r"), ("zes", "z"), ("s", ""), ("ando", "ar"), ("endo", "er"),
            ("indo", "ir"), ("ado", "ar"), ("ada", "ar"), ("ados", "ar"), ("adas", "ar"), ("ido", "er"),
            ("ida", "er"), ("idos", "er"), ("idas", "er"),
        ]
        for suffix, repl in rules:
            if w.endswith(suffix) and len(w) - len(suffix) >= 2:
                out.append(w[: len(w) - len(suffix)] + repl)
        base = list(out)
        for item in base:
            if item.endswith("a") and len(item) > 3:
                out.append(item[:-1] + "o")
            if item[-1:] in {"a", "e", "i"} and len(item) > 3:
                out.append(item + "r")
    else:
        for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e")):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                out.append(w[: len(w) - len(suffix)] + repl)
    seen, unique = set(), []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


class LexicalRelationIndex:
    """Directional synonym/antonym/hypernym index merged from local caches.

    ``up[x]`` holds terms that are more general than ``x`` (x ⊏ y). PT-LKB
    stores ``x hyponym y`` when x is a hyponym of y, while wiktextract stores
    ``y`` in ``x.hypernyms``; both are normalised to ``up``.
    """

    def __init__(self) -> None:
        self.syn: Dict[str, Set[str]] = {}
        self.ant: Dict[str, Set[str]] = {}
        self.up: Dict[str, Set[str]] = {}
        self.down: Dict[str, Set[str]] = {}

    def _add(self, table: Dict[str, Set[str]], left: str, right: str) -> None:
        if not left or not right or left == right or "_" in left or "_" in right or " " in right:
            return
        table.setdefault(left, set()).add(right)

    def add_row(self, row: dict) -> None:
        term = str(row.get("term") or row.get("word") or "").strip().lower()
        if not term or "_" in term:
            return
        keys = {term} | {str(alias).strip().lower() for alias in row.get("aliases") or [] if "_" not in str(alias)}
        for sense in row.get("senses") or []:
            for relation in sense.get("relations") or []:  # PT-LKB format
                other = str(relation.get("term", "")).strip().lower()
                name = relation.get("relation")
                for key in keys:
                    if name == "synonym":
                        self._add(self.syn, key, other)
                    elif name == "antonym":
                        self._add(self.ant, key, other)
                    elif name == "hyponym":
                        self._add(self.up, key, other)
                    elif name == "hypernym":
                        self._add(self.down, key, other)
            for field, table in (("synonyms", self.syn), ("antonyms", self.ant), ("hypernyms", self.up), ("hyponyms", self.down)):
                for other in sense.get(field) or []:
                    other = str(other).strip().lower()
                    for key in keys:
                        self._add(table, key, other)

    def drop_conflicting_directions(self) -> int:
        """Remove pairs listed as both hypernym and hyponym (about 6% of PT-LKB edges).

        PT-LKB merges several resources; some of them disagree on direction
        (e.g. ``cachorro``/``animal``). A wrong direction flips forward and
        reverse entailment, so ambiguous pairs are dropped rather than guessed.
        """
        removed = 0
        for key in set(self.up) & set(self.down):
            conflict = self.up[key] & self.down[key]
            if conflict:
                removed += len(conflict)
                self.up[key] -= conflict
                self.down[key] -= conflict
        return removed

    def __len__(self) -> int:
        return len(set(self.syn) | set(self.ant) | set(self.up) | set(self.down))

    @staticmethod
    def _hit(table: Dict[str, Set[str]], lefts: Sequence[str], rights: Sequence[str]) -> bool:
        right_set = set(rights)
        return any(table.get(left, set()) & right_set for left in lefts)

    def relation(self, left: str, right: str, language: str = "pt") -> str | None:
        lefts = lemma_candidates(left, language)
        rights = lemma_candidates(right, language)
        if self._hit(self.ant, lefts, rights) or self._hit(self.ant, rights, lefts):
            return "antonym"
        if self._hit(self.up, lefts, rights) or self._hit(self.down, rights, lefts):
            return "hypernym"
        if self._hit(self.down, lefts, rights) or self._hit(self.up, rights, lefts):
            return "hyponym"
        if self._hit(self.syn, lefts, rights) or self._hit(self.syn, rights, lefts):
            return "synonym"
        return None


_INDEX_CACHE: Dict[Tuple[str, ...], LexicalRelationIndex] = {}


def load_relation_index(paths: Iterable[Path | str | None]) -> LexicalRelationIndex:
    resolved = tuple(sorted(str(Path(p).resolve()) for p in paths if p and Path(p).exists()))
    if resolved in _INDEX_CACHE:
        return _INDEX_CACHE[resolved]
    index = LexicalRelationIndex()
    for path in resolved:
        files = sorted(Path(path).glob("*.jsonl")) if Path(path).is_dir() else [Path(path)]
        for file in files:
            with file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        index.add_row(json.loads(line))
    index.drop_conflicting_directions()
    _INDEX_CACHE[resolved] = index
    return index


def operator_type(word: str, language: str) -> str:
    lower = word.lower()
    lexicon = PT_OPERATORS if language == "pt" else EN_OPERATORS
    for name, words in lexicon.items():
        if lower in words:
            return name
    if NUMBER_RE.match(lower) or lower in (PT_NUMBER_WORDS if language == "pt" else EN_NUMBER_WORDS):
        return "number"
    return "none"


def number_value(word: str, language: str) -> float | None:
    lower = word.lower()
    words = PT_NUMBER_WORDS if language == "pt" else EN_NUMBER_WORDS
    if lower in words:
        return float(words[lower])
    if not NUMBER_RE.match(lower):
        return None
    if language == "pt":
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", lower):
            lower = lower.replace(".", "")
        lower = lower.replace(",", ".")
    else:
        lower = lower.replace(",", "")
    try:
        return float(lower)
    except ValueError:
        return None


def is_stopword(word: str, language: str) -> bool:
    return word.lower() in (PT_STOPWORDS if language == "pt" else EN_STOPWORDS)


def word_relation(left: str, right: str, language: str, index: LexicalRelationIndex | None) -> str | None:
    """Relation from a premise word (left) to a hypothesis word (right)."""
    lval, rval = number_value(left, language), number_value(right, language)
    if lval is not None and rval is not None:
        return "number_equal" if lval == rval else "number_conflict"
    if left.lower() == right.lower():
        return "identical"
    if PUNCTUATION_RE.match(left) or PUNCTUATION_RE.match(right):
        return None
    if is_stopword(left, language) or is_stopword(right, language):
        return None
    if index is not None and len(left) > 2 and len(right) > 2:
        relation = index.relation(left, right, language)
        if relation:
            return relation
    if len(left) > 3 and len(right) > 3 and light_stem(left, language) == light_stem(right, language):
        return "same_stem"
    return None


def build_token_graph(
    encoding,
    sentence1: str,
    sentence2: str,
    language: str,
    index: LexicalRelationIndex | None,
    max_len: int,
) -> Dict[str, torch.Tensor]:
    """Build token-level graph tensors aligned with a fast-tokenizer pair encoding.

    ``encoding`` must come from ``tokenizer(s1, s2, ..., return_offsets_mapping=True)``
    with exactly the same truncation/padding as the model input.
    """
    language = "pt" if str(language).lower().startswith("pt") else "en"
    texts = (sentence1, sentence2)
    seq_ids = encoding.sequence_ids(0)
    word_ids = encoding.word_ids(0)
    offsets = encoding["offset_mapping"][0].tolist()
    length = len(seq_ids)

    words: Dict[Tuple[int, int], List[int]] = {}
    spans: Dict[Tuple[int, int], List[int]] = {}
    for position, (seq, wid) in enumerate(zip(seq_ids, word_ids)):
        if seq is None or wid is None:
            continue
        key = (seq, wid)
        words.setdefault(key, []).append(position)
        start, end = offsets[position]
        span = spans.setdefault(key, [start, end])
        span[0], span[1] = min(span[0], start), max(span[1], end)
    ordered = sorted(words, key=lambda key: words[key][0])
    text_of = {key: texts[key[0]][spans[key][0]:spans[key][1]] for key in ordered}

    tg_seq = torch.full((max_len,), -1, dtype=torch.long)
    tg_node = torch.zeros(max_len, dtype=torch.bool)
    tg_op = torch.zeros(max_len, dtype=torch.long)
    rel = np.zeros((max_len, max_len), dtype=np.uint8)

    op_of: Dict[Tuple[int, int], str] = {}
    previous_word: Dict[int, str | None] = {0: None, 1: None}
    for key in ordered:
        positions = words[key]
        word = text_of[key]
        op = operator_type(word, language)
        if language == "pt" and word.lower() in PT_CLAUSE_INITIAL_CONDITIONALS:
            before = previous_word[key[0]]
            clause_start = before is None or bool(PUNCTUATION_RE.match(before)) or before.lower() in PT_CLAUSE_OPENERS
            op = "conditional" if clause_start else "none"
        previous_word[key[0]] = word
        op_of[key] = op
        for position in positions:
            tg_seq[position] = key[0]
            tg_op[position] = OP_TO_ID[op]
        is_punct = bool(PUNCTUATION_RE.match(word))
        if (op in NODE_OPS) or (not is_punct and op == "none" and not is_stopword(word, language)):
            tg_node[positions[0]] = True
        if len(positions) > 1:
            block = np.ix_(positions, positions)
            rel[block] = TOKGRAPH_REL_TO_ID["same_word"]

    def set_rel(left_key, right_key, name: str) -> None:
        rows, cols = words[left_key], words[right_key]
        rel[np.ix_(rows, cols)] = TOKGRAPH_REL_TO_ID[name]
        rel[np.ix_(cols, rows)] = TOKGRAPH_REL_TO_ID[INVERSE_RELATION.get(name, name)]

    # Within-sentence operator scope: operator -> next words until punctuation.
    for side in (0, 1):
        side_keys = [key for key in ordered if key[0] == side]
        for index_in_side, key in enumerate(side_keys):
            if op_of[key] not in SCOPED_OPS:
                continue
            for other in side_keys[index_in_side + 1:index_in_side + 1 + SCOPE_WINDOW]:
                if PUNCTUATION_RE.match(text_of[other]):
                    break
                set_rel(key, other, "scope_out")

    premise_keys = [key for key in ordered if key[0] == 0]
    hypothesis_keys = [key for key in ordered if key[0] == 1]
    for left_key in premise_keys:
        for right_key in hypothesis_keys:
            name = word_relation(text_of[left_key], text_of[right_key], language, index)
            if name:
                set_rel(left_key, right_key, name)

    return {
        "tg_seq": tg_seq,
        "tg_node": tg_node,
        "tg_op": tg_op,
        "tg_rel": torch.from_numpy(rel[:max_len, :max_len].copy()),
        "tg_length": torch.tensor(length, dtype=torch.long),
    }
