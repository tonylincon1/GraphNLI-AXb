# PIIG-NLI v5 Architecture

PIIG v5 extends v4 with explicit scope-aware evidence. It keeps the local
Wiktionary cache, relation-aware proof GAT, text-to-graph cross-attention,
feature composition, and auxiliary proof supervision. The new version does
not add a final fusion gate.

## New Evidence Nodes

The graph can now include deterministic candidate nodes for:

- `negation_scope_evidence`: compares premise and hypothesis polarity;
- `quantifier_polarity_evidence`: describes the polarity of quantifiers such
  as `every`, `no`, and `at most`;
- `monotonicity_direction`: records upward or downward reasoning context;
- `disjunction_scope_evidence`: records whether a hypothesis selects one
  branch of a disjunction;
- `numeric_relation_evidence`: exposes numbers found on each side;
- `directed_lexical_relation`: preserves the direction of lexical relations
  such as synonym, antonym, hypernym, and hyponym.

These are evidence candidates, not hard-coded labels. The GAT and
cross-attention still decide whether each node is relevant to the example.

## Example

For:

```text
Premise: No doctor treats every patient.
Hypothesis: A doctor treats every patient.
```

PIIG v5 can create nodes such as:

```text
evidence negation scope: premise is negated; hypothesis is affirmed
evidence quantifier polarity: premise downward quantifier: restrictor polarity is downward
evidence monotonicity direction: downward under every no
evidence polarity contrast from negation scope
```

For a numerical example:

```text
evidence numeric relation: premise numbers ['3']; hypothesis numbers ['5']
```

## Model Composition

The Transformer encodes the sentence pair. The proof nodes are encoded with
the same input embedding space, updated through the proof GAT, and queried by
the textual states through cross-attention. The classifier composes:

```text
pair representation
injected text representation
pooled proof representation
absolute text difference
text/proof products
proof-type counts
```

An auxiliary proof head predicts which evidence types are present. This
encourages the graph representation to preserve logical structure while the
main NLI head remains responsible for the final decision.

## Running

```bash
poetry run graphnli-train \
  --models piig_v5 \
  --seeds 1001,1002,1003,1004,1005 \
  --model-name bert-base-multilingual-cased \
  --training-protocol mnli \
  --train-manifest data/manifests/mnli_train_2000_stratified.jsonl \
  --batch-size 32 \
  --epochs 8 \
  --lr 1e-5 \
  --selection-metric nli3/macro_f1 \
  --patience 2 \
  --eval-languages en \
  --wiktionary-cache data/cache/wiktionary_entries_mnli.jsonl \
  --device mps \
  --output-dir runs
```
