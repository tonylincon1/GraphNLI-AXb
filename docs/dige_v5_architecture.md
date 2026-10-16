# DiGE-NLI v5 Architecture

DiGE-NLI v5 is the relation-aware GAT feature version of v2. The goal is to
test whether the model benefits from an explicit graph-attention feature branch,
without replacing the feature-composition recipe that made v2 stronger than v1,
v3, and v4 in the first AX-b runs.

## Design Principle

```text
v5 = v2 + relation-aware GAT feature pooling
```

v2 already uses a relation-aware GAT to encode graph nodes. In v2, however, the
GAT output is reduced to one global graph vector:

```text
h_graph = pool(GAT(nodes, edges))
```

v5 keeps that vector and adds a second graph feature:

```text
h_relation_gat = relation_attention_pool(GAT(nodes, edges), edge_types)
```

The final vector is additive:

```text
phi_v5 = [phi_v2; phi_relation_gat]
```

## Inputs

The input graph is the same diagnostic graph used by v2:

- premise and hypothesis anchor tokens;
- node types such as premise, hypothesis, negation, quantifier, number,
  disjunction, conjunction, conditional, temporal, and monotone;
- typed edges such as identity, synonym, antonym, hypernym, negation scope,
  number conflict, disjunction scope, conditional scope, temporal order,
  universal scope, and downward monotone scope.

## Base v2 Branch

The v2 branch produces:

```text
phi_v2 =
[
  h_premise;
  h_hypothesis;
  |h_premise - h_hypothesis|;
  h_premise * h_hypothesis;
  cos(h_premise, h_hypothesis);
  h_pair;
  h_graph;
  |h_pair - h_graph|;
  h_pair * h_graph;
  cos(h_pair, h_graph);
  cos(h_premise, h_graph);
  diagnostic_relation_histogram
]
```

This is preserved exactly in v5.

## Relation-Aware GAT Feature Branch

After the normal relation-aware GAT layers, v5 does not only pool all graph nodes
together. It asks a separate question for every relation type:

```text
For relation r:
  1. find graph nodes incident to edges of type r
  2. score those nodes with a learned query q_r
  3. attention-pool the nodes into h_r
```

This creates one relation-specific vector per diagnostic relation:

```text
H_relation = [h_identity, h_antonym, h_negation_scope, ...]
```

Then v5 learns which relation vectors matter for the current example:

```text
a_relation = softmax(score(h_r))
h_relation_gat = sum_r a_relation[r] * h_r
```

The relation attention distribution itself is also passed into the final
classifier as an interpretable feature:

```text
a_relation in R^(number_of_relations)
```

## Feature Composition

The new feature block is:

```text
phi_relation_gat =
[
  h_relation_gat;
  |h_pair - h_relation_gat|;
  h_pair * h_relation_gat;
  |h_graph - h_relation_gat|;
  cos(h_pair, h_relation_gat);
  cos(h_graph, h_relation_gat);
  a_relation
]
```

Final input:

```text
phi_v5 = [phi_v2; phi_relation_gat]
```

## Heads And Loss

v5 keeps the v2 training objective:

- 3-class NLI head;
- binary entailment/not_entailment AX-b head;
- diagnostic relation evidence head.

Loss:

```text
L = L_3class + lambda * (L_binary + L_evidence)
```

## Suggested Figure

Use four blocks:

```text
Premise + Hypothesis
        |
        v
Transformer Encoder ----------------------------+
        |                                       |
        v                                       |
h_pair, h_premise, h_hypothesis                |
                                                |
Diagnostic Graph Builder                       |
        |                                       |
        v                                       |
Relation-aware GAT                              |
        |                                       |
        +------------------+--------------------+
        |                  |
        v                  v
 global graph pool   relation-wise GAT pool
    h_graph              h_relation_gat + a_relation
        |                  |
        +------------------+--------------------+
                           |
                           v
                  Feature Composition v5
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
    3-Class NLI Head  Binary AX-b Head  Evidence Head
```

## Expected Benefit

v5 should help when the useful evidence is not just that a relation exists, but
which nodes participate in that relation. Good target slices are:

- antonym and contradiction examples;
- negation-scope examples;
- disjunction and conditional cases;
- number conflict cases;
- examples where several relation types are present but only one is decisive.

The main experimental question is whether `h_relation_gat` improves AX-b MCC and
calibration over v2 without overfitting to XNLI validation.
