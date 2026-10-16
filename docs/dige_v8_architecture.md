# DiGE-NLI v8 Architecture

DiGE-NLI v8 is **Counterfactual Logic Graph Composition**. It is a focused
extension of `dige_v2` for AX-b, where a small change to negation, a
quantifier, or a connective can reverse the NLI decision.

## Design Hypothesis

The lexical graph used by previous versions contains logical operators, but
they share the same pooled representation as ordinary lexical evidence. v8
keeps the v2 text/graph feature composition and adds an independent logic
pathway. Its graph attention only sees graph edges incident to an operator, so
the representation has to preserve the local scope neighbourhood of that
operator.

```text
premise + hypothesis
        |
Transformer pair encoder ------------------------------> v2 base features
        |
lexical evidence graph --> ordinary relation-aware GAT -> lexical graph pool
        |
        +--> operator-only subgraph --> small relation-aware GAT
                 |                       |
                 |                       +--> premise logic vector
                 |                       +--> hypothesis logic vector
                 |
                 +--> operator counts by side
                                      |
                      [P_logic, H_logic, |P-H|, P*H, counts]
                                      |
                       v8 feature composition --> NLI / binary heads
                                      |
                 signature + counterfactual auxiliary losses
```

## Logic Subgraph

The node types are:

```text
negation, quantifier, disjunction, conjunction,
conditional, downward-monotone, temporal
```

Only an edge whose source or destination is one of these nodes is visible to
the v8 logic GAT. That includes scope edges created around the operator, local
adjacency, and cross-sentence diagnostic edges. Ordinary lexical nodes may
provide the operator's local context, but they cannot form an unrelated
lexical-only path inside this branch.

The model pools logical nodes separately from the premise and hypothesis. It
then composes directional and symmetric features:

```text
L = [l_p, l_h, |l_p - l_h|, l_p * l_h, log(1 + operator_counts_by_side)]
```

`L` is appended to the original v2 composition; no v2 feature is removed.

## Structural Supervision

The graph builder supplies a 14-bit target: for each of the seven operator
families, whether it occurs in the premise and whether it occurs in the
hypothesis. This is a graph-derived target, not an inferred AX-b label.

For examples containing an operator, v8 also predicts one factual
configuration class:

| Class | Operator profile |
|---|---|
| 0 | no logical operator |
| 1 | matching premise/hypothesis profiles |
| 2 | premise-only |
| 3 | hypothesis-only |
| 4 | both sides present but profiles conflict |

The input to this head is directional:

```text
[l_p, l_h, l_p - l_h, l_p * l_h]
```

Counterfactually swapping premise and hypothesis swaps classes 2 and 3 while
leaving matching/conflicting configurations distinct. This makes direction and
operator asymmetry identifiable without inventing a false NLI label for a
synthetic rewritten sentence. During training, the configuration head receives
both the factual composition and the virtual swapped composition; its target is
also swapped between classes 2 and 3. The main NLI prediction is made only
from the factual ordering.

The auxiliary term is:

```text
L_aux = aux_weight * (L_evidence + L_binary
                      + 0.5 * L_signature
                      + 0.25 * L_counterfactual)
```

The two logic losses are masked out for examples without an identified logical
operator. This avoids rewarding a trivial all-zero detector on MultiNLI.

## What Is New Relative to v2 and v7

| Version | Main mechanism | v8 difference |
|---|---|---|
| v2 | feature composition over full lexical graph | adds a separately pooled operator-scope pathway |
| v7 | calibrated selection over lexical/semantic routes | v8 does not route all evidence; it isolates logic and supervises directional configurations |

This is intentionally not a claim that every AX-b example has a deterministic
rule label. The NLI prediction remains learned from MultiNLI. The structural
tasks make the model's graph features sensitive to the kinds of interventions
that AX-b diagnoses.

## First Comparison

Run v8 against the non-graph baseline and its direct v2 parent under the same
settings. `nli3/macro_f1` remains the appropriate checkpoint selector because
the held-out MultiNLI validation split has three-class labels; AX-b stays final
evaluation only.

```bash
poetry run graphnli-train \
  --models pair_cls,dige_v2,dige_v8 \
  --seeds 42 \
  --training-protocol mnli \
  --train-limit 20000 \
  --batch-size 32 \
  --epochs 6 \
  --lr 1e-5 \
  --selection-metric nli3/macro_f1 \
  --patience 2 \
  --eval-languages all \
  --output-dir runs
```

Primary result: AX-b binary MCC. Also compare AX-b F1, accuracy, ECE, and
Brier score, plus each AX-b `logic` group in the generated summaries. For a
causal ablation, train `dige_v8` once with `--aux-weight 0`; this removes the
evidence, binary, signature, and counterfactual auxiliary objectives together.
