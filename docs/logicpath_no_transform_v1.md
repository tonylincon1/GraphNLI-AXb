# TRIAL-NLI without explicit transformations

`logicpath_no_transform_v1` is a controlled variant of TRIAL-NLI designed to
test the contribution of the transformation branch independently.

## Graph

Each example creates a typed graph with:

- one premise-state node;
- one hypothesis-state node;
- lexical unit nodes extracted from each sentence;
- Wiktionary entry and gloss nodes for cached lexical evidence;
- relation-aware edges for containment, lexical relations, support, and
  Wiktionary evidence/relations.

The graph does **not** create transformation nodes such as
`temporal_change`, `negation_flip`, or `hypernym_generalization`.

## Encoder and fusion

The premise and hypothesis are encoded by the Transformer. In parallel, the
typed graph is encoded by relation-aware graph-attention layers. The final
classifier receives the same four groups used by the graph branch:

```text
[h_pair, h_graph, |h_pair - h_graph|, h_pair * h_graph]
```

There is no transformation head, inverse-transformation head, transformation
probability feature, or auxiliary transformation loss. This makes the variant
directly comparable with `logicpath_v2` while isolating the effect of explicit
transformation prediction.

## Controlled experiment

`configs/experiment_0007.yaml` reproduces the Portuguese protocol of
`experiment_0003`: the model is trained and validated on the union of InferBR,
ASSIN, and ASSIN2, then evaluated on the official test split of each dataset.
It uses the Portuguese BERT model, ten seeds (`1001`--`1010`), the same batch
size, tuning space, early stopping, and evaluation metrics.
