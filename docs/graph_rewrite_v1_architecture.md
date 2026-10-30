# Graph Rewrite v1

`graph_rewrite_v1` is a new architecture for NLI diagnostics. It uses the
existing pair-conditioned evidence graph as its input, but performs learned
state transformations over that graph instead of only pooling its nodes.

## Main idea

```text
initial evidence graph
    -> relation message passing
    -> operation prediction
    -> operation-conditioned node update
    -> rewritten graph
    -> NLI prediction
```

The graph contains sentence states, lexical units, logical transformations,
and optional Wiktionary evidence nodes. The model predicts a soft distribution
over operations such as:

```text
lexical_identity, lexical_synonym, hypernym_generalization,
hyponym_specialization, antonym_contradiction, negation_flip,
quantifier_change, conjunction_change, conditional_alignment,
numeric_change, temporal_change
```

That operation representation is used to update the graph state at every
rewrite step. The model therefore learns a sequence of state changes rather
than treating the graph as a static feature vector.

## Classifier representation

The classifier receives the Transformer pair representation, the initial graph
state, the rewritten graph state, their difference, their interaction, and the
predicted forward and inverse operation distributions.

## Auxiliary objectives

Training combines NLI classification with forward-operation prediction,
inverse-operation prediction, and a small state-change objective for examples
where the graph builder detected an active transformation.

## Run

```bash
poetry run graphnli-train \
  --model-name roberta-base \
  --models graph_rewrite_v1 \
  --training-protocol mnli \
  --train-manifest data/manifests/mnli_train_2000_stratified.jsonl \
  --seeds 42 \
  --epochs 8 \
  --batch-size 16 \
  --selection-metric nli3/macro_f1 \
  --wiktionary-cache data/cache/wiktionary_entries_mnli.jsonl \
  --save-piig-explanations \
  --output-dir runs
```

The saved explanation files contain graph node importance, relation attention,
operation scores, and the detected operation targets.
