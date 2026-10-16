# DiGE-NLI v9 Architecture

DiGE-NLI v9 is **Sparse Latent-Relation Graph Composition**. It extends v8's
operator-scope pathway with a second graph whose cross-sentence edges and
relation types are learned from the pair itself.

## Motivation

The explicit graph provides useful priors such as identity, WordNet relations,
negation, and quantifier scope. Its relation inventory is necessarily
incomplete. v9 lets the model discover a small set of reusable relation
patterns, while constraining that flexibility so a dense contextual graph does
not overwhelm the diagnostic signal.

## Dynamic Relation Graph

For every valid cross-sentence node pair `(i, j)`, v9 constructs a contextual
edge representation:

```text
e_ij = [n_i, n_j, |n_i - n_j|, n_i * n_j, h_pair]
```

Two heads predict:

```text
g_ij = sigmoid(gate(e_ij))                 edge relevance
p_ij = softmax(relation_head(e_ij))        distribution over 8 latent relations
```

For each source node, only the three targets with the highest gated attention
score remain. The gate is included in the attention logit, so it controls both
whether an edge survives and how much message it contributes.

```text
a_ij = softmax_j(q_i . k_j / sqrt(d) + log(g_ij))   over top-3 candidates
m_i = sum_j a_ij [W_v n_j + sum_k p_ijk r_k]
n_i_prime = LayerNorm(n_i + ELU(m_i))
```

`r_k` is the embedding of latent relation `k`. The dynamic graph pool is then
combined with the explicit lexical graph pool, pair representation, v2
features, and the v8 logic features.

## Feature Composition

The final v9 addition is:

```text
D = [h_dynamic, |h_dynamic - h_explicit|,
     h_dynamic * h_explicit, h_dynamic * h_pair,
     cos(h_dynamic, h_pair), mean_gate, mean_latent_relation_mass]
```

The NLI and binary heads receive `[v2_features, v8_logic_features, D]`.
Thus the learned graph augments rather than replaces lexical and logical
evidence.

## Constraints Against Relation Collapse

v9 uses three regularizers inside its existing auxiliary loss:

```text
0.05 * mean(g_ij)                         sparse-edge pressure
0.05 * mean((cos(r_k, r_l) - I_kl)^2)     diverse relation prototypes
0.10 * KL(p_ij || prior(edge_type_ij))    soft explicit-relation guidance
```

The last term applies only where an explicit cross-sentence diagnostic edge
already exists. `prior(edge_type_ij)` is learned, so WordNet/logic relations
guide the latent inventory without forcing any latent relation to equal a fixed
symbol. The v8 evidence, binary, logical-signature, and virtual
premise/hypothesis-swap losses remain active.

## Inspecting Learned Relations

The prediction CSVs for v9 contain:

```text
latent_relation/0 ... latent_relation/7
```

Each is the mean attention-weighted mass of one latent relation for that
example. These identifiers have no predefined linguistic label. To interpret a
prototype, group examples with high mass for it and inspect their AX-b `logic`
tag, label, and active explicit evidence. The weights in
`latent_relation_embedding` can also be projected after training, but example
inspection is the more defensible first analysis.

## First Experiment

Use the same seed, data cap, model, and checkpoint protocol for all models:

```bash
poetry run graphnli-train \
  --models pair_cls,dige_v2,dige_v8,dige_v9 \
  --seeds 42 \
  --model-name bert-base-multilingual-cased \
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

Judge v9 first on AX-b binary MCC, then AX-b calibration and the logic-group
breakdown. Compare its 15-language XNLI mean with v8: the learned relations
should help multilingual transfer only if they avoid relying on English-only
WordNet patterns.
