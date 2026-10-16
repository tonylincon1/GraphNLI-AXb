# DiGE-NLI v7 Architecture

DiGE-NLI v7 is **Reliability-Calibrated Relation Routing**. It is a corrective
iteration over v6, motivated by experiment 010: v6 improved AX-b and English
XNLI over `pair_cls`, but its virtual `semantic_alignment` route dominated the
average gate and performance fell in 14 of 15 XNLI languages.

## Core Change

v6 independently used sigmoid gates for every route. Multiple routes could be
large at once, and the semantic route had no explicit reliability constraint.

v7 separates the evidence into:

```text
explicit diagnostic routes
  identity, synonym, antonym, quantifier, negation_scope, number_conflict, ...

virtual fallback route
  semantic_alignment
```

Explicit routes compete with a softmax. The virtual semantic route receives
only a confidence-limited residual budget when explicit evidence exists.

## Alignment Reliability

For every visible graph node, v7 finds its most similar node on the opposite
sentence side using normalized contextual node embeddings. Let `s1` and `s2`
be the best and second-best similarities. The local reliability is:

```text
c_i = sigmoid(4 * ((s1 - s2) - 0.15)) * sigmoid(3 * s1)
```

The graph-level semantic confidence `c_semantic` is the mean over visible
source nodes. Ambiguous alignments, where the best and second-best candidates
are similar, therefore receive lower confidence.

## Competitive Routing

For explicit routes, v7 computes contextual route scores and normalizes them
only over present explicit paths:

```text
a_explicit = softmax(score_r for present explicit r)
```

The virtual route receives:

```text
g_semantic = 0.35 * sigmoid(score_semantic) * c_semantic
```

when at least one explicit route exists. The explicit routes receive the
remaining mass:

```text
g_r = (1 - g_semantic) * a_explicit,r
```

If a graph has no explicit cross-sentence relation, semantic alignment becomes
the full fallback route. This avoids discarding multilingual examples whose
only available bridge is contextual similarity.

## Why This Addresses v6

| v6 issue | v7 correction |
|---|---|
| Semantic route had mean gate 0.722 on AX-b | It is capped at 0.35 when explicit evidence exists |
| Independent sigmoids could activate many routes | Explicit routes compete for the residual probability mass |
| Top-1 semantic matches could be ambiguous | Top-1/top-2 margin controls semantic confidence |
| AX-b calibration deteriorated | A bounded semantic contribution reduces a source of overconfident graph evidence |

v7 retains the v6 counterfactual sufficiency and necessity losses. The change
is specifically a routing and calibration intervention, so its ablation is
clean: compare v6 and v7 with the same graph, data, seed, batch size, and
training schedule.

## Suggested Comparison

Use the stronger experiment-009 regime, not the 2,000-example smoke regime of
experiment 010:

```bash
poetry run graphnli-train \
  --models pair_cls,dige_v2,dige_v6,dige_v7 \
  --seeds 42 \
  --training-protocol mnli \
  --train-limit 20000 \
  --batch-size 32 \
  --epochs 6 \
  --lr 2e-5 \
  --selection-metric nli3/macro_f1 \
  --patience 2 \
  --eval-languages all \
  --output-dir runs
```

Primary success criteria are: AX-b MCC not lower than v6, lower AX-b ECE and
Brier score than v6, and recovery of mean XNLI macro-F1 and MCC. Inspect the
`route_gate/*` columns in each v7 prediction CSV to confirm that
`semantic_alignment` is below 0.35 whenever explicit routes are present.
