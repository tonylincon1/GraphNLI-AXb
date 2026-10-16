# DiGE-NLI v1 Architecture

DiGE-NLI v1 means Diagnostic Evidence Graph Adapter for NLI. The model is built
for premise-hypothesis classification and AX-b-style diagnostic analysis.

## High-Level Flow

1. Input pair: premise and hypothesis.
2. Text encoder: a pretrained Transformer encodes the pair jointly.
3. Diagnostic graph builder: a small per-example evidence graph is built from
   the same premise-hypothesis pair.
4. Graph encoder: relation-aware graph attention updates the evidence nodes.
5. Evidence adapter: text and graph representations are fused through a gate.
6. Prediction heads:
   - NLI classifier predicts entailment, neutral, or contradiction.
   - Evidence head predicts which diagnostic relation types appeared in the
     graph.
7. AX-b evaluation maps neutral/contradiction to not_entailment.

## Inputs

The model receives:

- `premise`: first sentence.
- `hypothesis`: second sentence.
- `label_3`: entailment, neutral, contradiction for XNLI-style training.
- `label_binary`: entailment vs not_entailment for AX-b evaluation.

## Transformer Branch

The pair is tokenized as:

```text
[CLS] premise [SEP] hypothesis [SEP]
```

For XLM-R-like models this uses the model's own separator scheme.

The Transformer returns contextual token states. The model takes the pooled pair
state:

```text
h_text ∈ R^H
```

This vector is the global semantic representation of the sentence pair.

## Diagnostic Evidence Graph Builder

DiGE v1 builds one graph per example.

### Nodes

Nodes are short textual anchors selected from premise and hypothesis:

- content words;
- negation cues: `not`, `no`, `never`, `without`, etc.;
- quantifiers: `all`, `some`, `few`, `most`, `every`, etc.;
- numbers;
- selected premise tokens;
- selected hypothesis tokens.

Each node has:

```text
node_text
node_type ∈ {premise, hypothesis, negation, quantifier, number}
```

### Edges

Edges are typed diagnostic evidence relations:

- `self`;
- `intra_sentence`;
- `identity`;
- `synonym`;
- `antonym`;
- `hypernym`;
- `hyponym`;
- `negation_scope`;
- `quantifier`;
- `number_equal`;
- `number_conflict`.

WordNet is used for lexical semantic relations when available. Rule-based
detectors are used for negation, quantifiers, and numbers.

The graph is represented as:

```text
V = evidence nodes
E = typed adjacency matrix
A ∈ N^(N×N), where each entry is a relation type id
```

## Node Encoder

Each node text is tokenized and embedded with the Transformer token embedding
matrix, then mean-pooled:

```text
x_i = mean(TokenEmbedding(node_text_i))
```

A node-type embedding is added:

```text
z_i^0 = x_i + TypeEmbedding(node_type_i)
```

## Relation-Aware Graph Attention

DiGE v1 uses stacked relation-aware GAT layers.

For nodes `i` and `j`:

```text
score(i,j) = Q(z_i) · K(z_j) / sqrt(H) + bias(relation_type(i,j))
```

Only visible graph edges are attended to. The updated node state is:

```text
z_i' = LayerNorm(z_i + Dropout(Σ_j α_ij V(z_j)))
```

After `L` graph layers, the graph representation is masked mean-pooled:

```text
h_graph = mean({z_i^L | node_i is valid})
```

## Relation Count Features

The model also computes a compact relation histogram:

```text
r = log(1 + count(edge_type_k)) for each relation type k
```

This gives the classifier explicit access to which diagnostic phenomena were
triggered.

## Evidence Fusion Gate

The adapter learns how much to trust text vs graph evidence:

```text
g = sigmoid(W [h_text ; h_graph ; r])
h_fused = g ⊙ h_text + (1 - g) ⊙ h_graph
```

This is one of the main differences from simple graph baselines: the graph does
not merely get concatenated; it competes with and complements the Transformer
representation feature by feature.

## Prediction Heads

### NLI Head

The NLI classifier receives:

```text
[h_text ; h_graph ; h_fused ; r]
```

and predicts:

```text
entailment / neutral / contradiction
```

### Evidence Head

The auxiliary evidence head receives `h_graph` and predicts a multi-label vector
over the diagnostic relation types.

Training loss:

```text
L = CrossEntropy(NLI label) + λ * BCE(evidence relation labels)
```

For AX-b, which is test-only in this setup, evaluation maps:

```text
entailment -> entailment
neutral or contradiction -> not_entailment
```

## Suggested Figure Layout

Use a left-to-right diagram with six blocks:

1. **Input Pair**
   - Premise
   - Hypothesis
2. **Transformer Encoder**
   - Output: `h_text`
3. **Diagnostic Graph Builder**
   - Nodes: words, negation, quantifiers, numbers
   - Edges: synonym, antonym, negation, number conflict, etc.
4. **Relation-Aware Graph Attention**
   - Output: `h_graph`
   - Side output: relation histogram `r`
5. **Diagnostic Evidence Adapter**
   - Gate: `h_fused = g*h_text + (1-g)*h_graph`
6. **Heads**
   - NLI prediction
   - Evidence prediction

Visual cue idea:

- blue path for Transformer text semantics;
- green path for graph evidence;
- orange gate for fusion;
- purple auxiliary evidence head.

## What Makes v1 Different

Compared with the literature-style baselines:

- K-BERT-style injection adds knowledge as text.
- KGAnet-style relation GAT uses graph attention but mainly concatenates graph
  and text evidence.
- KGNLI-style dual branch separates semantic and knowledge relation branches.
- DiGE v1 adds phenomenon-oriented evidence labels, relation histograms, and a
  learned gate between text and graph states.
