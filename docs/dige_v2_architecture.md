# DiGE-NLI v2 Architecture

DiGE-NLI v2 means Diagnostic Evidence Graph Feature Composition for natural
language inference. It keeps the same diagnostic graph idea from v1, but changes
the fusion mechanism.

The central design change is:

```text
v1: gated fusion between text and graph
v2: explicit feature composition between text, graph, and diagnostic relations
```

This is closer to the GloSeGAT feature-composition style: instead of forcing the
model to compress text and graph into one fused vector, the classifier receives
several comparison features and learns which interactions matter.

## High-Level Flow

1. Input pair: premise and hypothesis.
2. Transformer encoder: encodes the pair jointly.
3. Diagnostic graph builder: builds an evidence graph from anchors in the pair.
4. Relation-aware graph attention: encodes graph evidence.
5. Feature composition: combines text, graph, differences, products, cosine
   similarities, and relation counts.
6. Prediction heads:
   - 3-class NLI head;
   - binary entailment/not_entailment head;
   - evidence-relation auxiliary head.

## Inputs

The model receives:

- `premise`;
- `hypothesis`;
- `label_3`: entailment, neutral, contradiction;
- `label_binary`: entailment vs not_entailment.

The binary label is derived as:

```text
entailment -> entailment
neutral -> not_entailment
contradiction -> not_entailment
```

This makes v2 more aligned with AX-b/SuperGLUE's binary MCC evaluation.

## Transformer Branch

The pair is encoded jointly:

```text
Transformer(premise, hypothesis)
```

The branch produces:

```text
h_pair ∈ R^H
```

as the pooled pair representation.

It also derives side-specific text vectors:

```text
h_premise ∈ R^H
h_hypothesis ∈ R^H
```

For BERT-like models, token type IDs are used to pool premise and hypothesis.
For models without token type IDs, v2 falls back to a midpoint split over the
joint sequence.

## Diagnostic Evidence Graph

The graph builder is the same evidence-oriented module used by v1.

### Nodes

Nodes are selected anchors:

- content words;
- negation cues;
- quantifiers;
- numbers;
- premise anchors;
- hypothesis anchors.

Node types:

```text
premise
hypothesis
negation
quantifier
number
```

### Relations

Edges are typed diagnostic relations:

```text
self
intra_sentence
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

WordNet supplies lexical relations when possible. Rules supply negation,
quantifier, and number relations.

## Graph Encoder

Node text is embedded with the Transformer embedding table and enriched with a
node-type embedding:

```text
z_i^0 = mean(TokenEmbedding(node_i)) + TypeEmbedding(node_type_i)
```

Relation-aware graph attention updates nodes:

```text
score(i,j) = Q(z_i) · K(z_j) / sqrt(H) + bias(relation_type(i,j))
```

Only graph-visible neighbors are attended to.

After graph attention layers:

```text
h_graph = masked_mean({z_i^L})
```

## Relation Histogram

The model computes a relation count vector:

```text
r_k = log(1 + count(edge_type = k))
```

This gives the classifier direct access to which diagnostic phenomena were
activated.

## Feature Composition

The composed feature vector is:

```text
φ_text =
[
  h_premise;
  h_hypothesis;
  |h_premise - h_hypothesis|;
  h_premise * h_hypothesis;
  cos(h_premise, h_hypothesis)
]
```

```text
φ_graph =
[
  h_pair;
  h_graph;
  |h_pair - h_graph|;
  h_pair * h_graph;
  cos(h_pair, h_graph);
  cos(h_premise, h_graph)
]
```

```text
φ_diag = [r]
```

Final classifier input:

```text
φ = [φ_text; φ_graph; φ_diag]
```

Unlike v1, there is no gate:

```text
no h_fused = g*h_text + (1-g)*h_graph
```

The model keeps the components separate and lets the MLP classifier learn their
interactions.

## Prediction Heads

### 3-Class NLI Head

Receives `φ` and predicts:

```text
entailment / neutral / contradiction
```

### Binary NLI Head

Receives `φ` and predicts:

```text
entailment / not_entailment
```

This head is used for binary AX-b metrics when available. It is designed to
improve MCC by training the exact decision boundary used by AX-b.

### Evidence Head

Receives `h_graph` and predicts which diagnostic relation types are active:

```text
identity, synonym, antonym, negation_scope, quantifier, ...
```

## Loss

Training combines:

```text
L_3 = CrossEntropy(3-class NLI)
L_binary = CrossEntropy(binary NLI)
L_evidence = BCE(diagnostic relation labels)
```

The implemented objective is:

```text
L = L_3 + λ * (L_binary + L_evidence)
```

where `λ` is `--aux-weight`.

## Suggested Figure Layout

Use a left-to-right architecture diagram:

```text
Premise + Hypothesis
        |
        +--------------------+
        |                    |
        v                    v
Transformer Encoder   Diagnostic Graph Builder
        |                    |
        |                    v
        |             Relation-Aware GAT
        |                    |
        |                    v
 h_pair, h_premise,   h_graph + relation histogram r
 h_hypothesis                |
        |                    |
        +---------+----------+
                  v
         Feature Composition
                  |
        +---------+----------+
        |                    |
        v                    v
  3-Class NLI Head      Binary NLI Head
        |
        v
Evidence Head from h_graph
```

Visual suggestions:

- blue: Transformer/text branch;
- green: diagnostic graph branch;
- orange: feature composition block;
- red: binary AX-b head;
- purple: evidence auxiliary head.

## Difference From v1

v1 asks:

```text
How much should I trust text vs graph?
```

v2 asks:

```text
What interactions between text and graph explain the label?
```

That distinction matters because AX-b often depends on small contrasts:

- negation added or removed;
- quantifier swapped;
- antonym/synonym relation;
- numeric conflict;
- monotonicity pattern.

Feature composition exposes these contrasts more directly than a single fused
state.
