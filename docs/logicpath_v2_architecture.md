# LogicPath v2: Wiktionary Evidence Transformation Graph

LogicPath v2 extends LogicPath v1 with local Wiktionary evidence. The
transformation graph remains pair-conditioned: it is built separately for
each premise-hypothesis pair and only uses cached entries selected from those
sentences.

## Graph structure

The graph contains the v1 nodes:

- `premise_state` and `hypothesis_state` for the original sentences;
- `premise_unit` and `hypothesis_unit` for selected lexical or logical units;
- `transformation` for operations such as synonymy, hypernymy, negation, or
  conditional alignment.

It additionally contains:

- `wiktionary_entry`, representing a matched dictionary entry;
- `wiktionary_gloss`, representing the first cached definition of that entry.

The new edge types are:

- `wiktionary_evidence`: connects a sentence unit to its entry and the entry
  to its gloss;
- `wiktionary_relation`: connects two entries when the local cache contains an
  explicit synonym, antonym, hypernym, or hyponym relation between them.

The graph does not add entries for arbitrary words. It uses the existing
`build_wiktionary_spec` selection logic, so only verified cache entries for
content words from the current pair are included. The number of entries is
controlled by `--max-wiktionary-entries`.

## Model

The Transformer encodes the original pair. Each graph node is encoded using
the Transformer's input embedding table and a learned node-type embedding.
Relation-aware Graph Transformer layers then propagate information along the
typed edges. The classifier receives:

```text
pair representation
graph representation
absolute difference(pair, graph)
elementwise product(pair, graph)
predicted transformation features
predicted inverse-transformation features
```

There is no explicit fusion gate. The classifier learns how to combine the
semantic pair representation with the transformation graph and Wiktionary
evidence.

## Auxiliary objectives

The model predicts which transformations are present and which inverse
transformations would reconstruct the source direction. These auxiliary
targets are the same as in v1. Wiktionary nodes provide evidence for the
lexical transformation, but they do not directly determine the NLI label.

## Training

```bash
poetry run graphnli-train \
  --models logicpath_v2 \
  --model-name bert-base-multilingual-cased \
  --training-protocol mnli \
  --train-manifest data/manifests/mnli_train_2000_stratified.jsonl \
  --seeds 42 \
  --epochs 8 \
  --batch-size 32 \
  --eval-languages en \
  --selection-metric nli3/macro_f1 \
  --wiktionary-cache data/cache/wiktionary_entries_mnli.jsonl \
  --max-wiktionary-entries 2 \
  --save-piig-explanations \
  --device mps \
  --output-dir runs
```

Despite the historical flag name `--save-piig-explanations`, the same option
saves LogicPath v2 explanations under each run's
`explanations/axb/` directory. The SVGs show Wiktionary nodes and typed edges;
edge width is based on the normalized Graph Transformer attention.

## Interpretation

The v2 is designed to test whether explicit lexical definitions and relations
improve the weak XNLI generalization observed with LogicPath v1, while keeping
the logical transformation mechanism unchanged for an ablation comparison.
