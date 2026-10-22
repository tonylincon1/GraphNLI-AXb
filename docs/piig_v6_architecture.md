# PIIG-NLI v6 Architecture

PIIG v6 extends PIIG v4 with a hybrid proof graph. The graph keeps the
sentence-level proof and Wiktionary nodes from v4, and adds a small set of
pair-conditioned word anchors. Typed edges describe why two nodes are
connected.

## Motivation

Sentence nodes preserve global scope and context, while word nodes make local
lexical and logical cues explicit. The hybrid design is useful for AX-b:
negation and quantifier scope require sentence context, but lexical bridges
and contradictions often begin with a small set of words.

## Graph contents

Every example starts with:

```text
[premise phrase] <-> [hypothesis phrase]
```

The constructor then selects at most four word anchors, normally two from each
side. It prioritizes words that participate in an identity, synonym, antonym,
hypernym, or hyponym relation, followed by logical cues such as `not`, `every`,
`some`, `if`, and `or`.

The graph can contain:

| Node type | Role |
| --- | --- |
| `premise_phrase` | compact representation of the premise |
| `hypothesis_phrase` | compact representation of the hypothesis |
| `premise_word` | selected word anchor from the premise |
| `hypothesis_word` | selected word anchor from the hypothesis |
| `wiktionary_entry` | cached lexical entry |
| `wiktionary_gloss` | cached sense definition |
| `wiktionary_relation` | cached lexical relation |
| logical evidence nodes | negation, conjunction, conditional, or generalization evidence |

## Typed edges

Edges are symmetric in the current implementation, but their labels preserve
the relation semantics used by the relation-aware GAT:

| Edge type | Meaning |
| --- | --- |
| `phrase_word` | word belongs to its sentence phrase |
| `word_alignment` | aligned lexical relation between premise and hypothesis words |
| `typed_support` | identity, synonym, hypernym, or hyponym evidence |
| `typed_contradiction` | antonym evidence |
| `word_lexicon` | word connected to a Wiktionary node |
| `word_evidence` | word connected to a logical evidence node |
| `phrase_lexicon` | phrase connected to Wiktionary evidence |
| `phrase_evidence` | phrase connected to logical evidence |

The implementation does not convert a lexical edge directly into an NLI
label. For example, a hypernym edge may support entailment in one quantifier
context and fail under a downward-entailing context. The Transformer state,
graph attention, and cross-attention jointly determine the final prediction.

## Information flow

1. The Transformer encodes the premise-hypothesis pair.
2. Each graph node is encoded with the Transformer input embeddings and a
   learned node-type embedding.
3. Relation-aware GAT layers propagate messages using the typed edge labels.
4. Cross-attention lets every text position retrieve graph information.
5. The classifier composes the pair representation, injected text state,
   pooled graph state, absolute differences, elementwise products, and node
   type counts.
6. An auxiliary proof head predicts which proof paths are present.

There is no learned fusion gate. The model receives the composed features and
learns how to use them in the classifier.

## Training

```bash
poetry run graphnli-train \
  --models piig_v6 \
  --model-name bert-base-multilingual-cased \
  --training-protocol mnli \
  --train-manifest data/manifests/mnli_train_2000_stratified.jsonl \
  --seeds 42 \
  --epochs 8 \
  --batch-size 32 \
  --eval-languages en \
  --wiktionary-cache data/cache/wiktionary_entries_mnli.jsonl \
  --selection-metric nli3/macro_f1 \
  --device mps \
  --output-dir runs
```

Explanations are saved under the usual experiment directory when
`--save-piig-explanations` is enabled. Word anchors appear as
`premise_word` and `hypothesis_word`, and their edge labels are included in
the JSON and SVG explanation files.

## Controlled comparison

The first ablation should keep the same fixed MNLI subset, seeds, and
hyperparameters:

```text
PIIG v4       sentence proof graph + Wiktionary
PIIG v6       PIIG v4 + typed word anchors
PIIG v6-no-W  PIIG v6 without Wiktionary nodes
```

This separates the contribution of the hybrid word graph from the external
lexical cache.
