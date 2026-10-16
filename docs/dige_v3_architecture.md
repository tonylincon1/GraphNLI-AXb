# DiGE-NLI v3 Architecture

DiGE-NLI v3 means Logic-Scope Diagnostic Evidence Graph Composition. It keeps
the successful feature-composition idea from v2, but changes the graph and the
composition layer to explicitly separate logical evidence from lexical evidence.

The motivation comes from the v1/v2 comparison: v2 improved global AX-b MCC, but
still struggled on disjunction, negation, temporal reasoning, and downward
monotonicity. v3 targets those phenomena directly.

## Core Change

```text
v2: one global graph vector h_graph
v3: semantic graph pool + logic graph pool + quantity/temporal graph pool
```

Instead of asking a single pooled graph vector to represent everything, v3
creates separate evidence views:

```text
h_semantic
h_logic
h_quantity_temporal
```

The classifier receives all of them through a GloSeGAT-style feature
composition block.

## Inputs

The input is still a premise-hypothesis NLI pair:

```text
premise
hypothesis
```

Training labels:

```text
label_3: entailment / neutral / contradiction
label_binary: entailment / not_entailment
```

## Transformer Branch

The pair is encoded jointly with mBERT/XLM-R:

```text
h_pair = TransformerPool(premise, hypothesis)
```

The model also obtains side-specific vectors:

```text
h_premise
h_hypothesis
```

For BERT-style encoders, token type IDs separate premise and hypothesis. For
encoders without token type IDs, the implementation falls back to a midpoint
split.

## Logic-Aware Diagnostic Graph

v3 enriches the graph builder with explicit logical node and edge types.

### Node Types

```text
premise
hypothesis
negation
quantifier
number
disjunction
conjunction
conditional
temporal
monotone
```

Examples:

- `not`, `no`, `never` -> `negation`
- `or`, `either`, `nor` -> `disjunction`
- `and`, `both` -> `conjunction`
- `if`, `then`, `unless`, `when` -> `conditional`
- `before`, `after`, `during`, `until` -> `temporal`
- `few`, `fewer`, `less`, `without` -> `monotone`

### Edge Types

v3 keeps the v2 relations:

```text
identity
synonym
antonym
hypernym
hyponym
negation_scope
quantifier
number_equal
number_conflict
```

and adds:

```text
disjunction_scope
conjunction_scope
conditional_scope
temporal_order
universal_scope
downward_monotone_scope
```

The new scope edges are local and typed: when a cue such as `or`, `if`, or `not`
is selected as a graph node, v3 connects it to nearby anchors in the same
sentence with the corresponding scope relation.

## Graph Encoder

The graph still uses relation-aware graph attention:

```text
score(i,j) = Q(z_i) · K(z_j) / sqrt(H) + bias(relation_type(i,j))
```

After the GAT layers, v3 does not only mean-pool all nodes. It creates typed
pools.

### Semantic Pool

```text
h_semantic = pool(nodes of type premise or hypothesis)
```

This pool carries ordinary lexical/semantic evidence.

### Logic Pool

```text
h_logic = pool(nodes of type negation, quantifier, disjunction,
               conjunction, conditional, monotone)
```

This pool is designed to preserve logical operators and their local scope.

### Quantity/Temporal Pool

```text
h_quantity_temporal = pool(nodes of type number or temporal)
```

This pool targets AX-b categories involving numeric intervals and temporal
relations.

If a pool has no matching nodes, it falls back to all graph nodes so the tensor
shape remains stable.

## Feature Composition

v3 starts with the v2 feature vector:

```text
φ_v2 =
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
  relation_histogram
]
```

Then it adds logic-scope composition:

```text
φ_logic =
[
  h_semantic;
  h_logic;
  h_quantity_temporal;
  |h_pair - h_logic|;
  h_pair * h_logic;
  cos(h_pair, h_logic);
  cos(h_hypothesis, h_logic)
]
```

Final feature vector:

```text
φ_v3 = [φ_v2; φ_logic]
```

## Prediction Heads

v3 keeps the v2 multi-head setup:

1. **3-class NLI head**
   ```text
   entailment / neutral / contradiction
   ```

2. **Binary NLI head**
   ```text
   entailment / not_entailment
   ```

3. **Evidence relation head**
   ```text
   active relation types in the graph
   ```

## Loss

```text
L = L_3class + λ * (L_binary + L_evidence)
```

where:

```text
L_3class = CrossEntropy(entailment, neutral, contradiction)
L_binary = CrossEntropy(entailment, not_entailment)
L_evidence = BCE(relation evidence labels)
```

## Suggested Figure

Use four horizontal lanes:

```text
Premise + Hypothesis
        |
        +----------------------+
        |                      |
        v                      v
Transformer Encoder     Logic-Aware Graph Builder
        |                      |
        |                      v
        |              Relation-Aware GAT
        |                      |
        |        +-------------+-------------+
        |        |             |             |
        v        v             v             v
 h_pair, h_p, h_h  h_semantic   h_logic   h_quantity_temporal
        |        |             |             |
        +--------+-------------+-------------+
                         |
                         v
              Feature Composition v3
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
     3-Class Head   Binary Head   Evidence Head
```

## Example Graph Image

An example SVG graph is available at:

```text
docs/dige_v3_graph_example.svg
```

The example uses:

```text
Premise:    All cats sleep or play before noon.
Hypothesis: Some cats do not sleep before noon.
```

It illustrates universal quantification, disjunction, negation scope, temporal
order, and lexical identity links.
