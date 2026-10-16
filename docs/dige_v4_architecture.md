# DiGE-NLI v4 Architecture

DiGE-NLI v4 is the conservative Wikidata extension of v2. The goal is to keep
the feature-composition design that worked best, while adding an entity evidence
branch for examples involving named entities, places, organizations, events, or
taxonomic concepts.

## Design Principle

```text
v4 = v2 + Wikidata entity evidence branch
```

v4 deliberately does not inherit the heavier v3 logic pools, because v3 improved
some phenomenon slices but hurt AX-b overall. The v4 branch is additive:

```text
φ_v4 = [φ_v2; φ_wikidata]
```

## Offline Wikidata Cache

Training does not query Wikidata online.

The optional cache path is:

```text
data/cache/wikidata_entities.jsonl
```

Each row can contain:

```json
{
  "label": "Paris",
  "qid": "Q90",
  "description": "capital and most populous city of France",
  "instance_of": ["capital city", "city"],
  "subclass_of": []
}
```

If the cache does not exist, v4 still runs. It falls back to simple capitalized
entity spans such as `Paris`, `Amazon`, or `Barack Obama`.

An optional helper can populate the cache:

```bash
PYTHONPATH=src poetry run python scripts/build_wikidata_cache.py \
  --languages en \
  --out data/cache/wikidata_entities.jsonl
```

## Inputs

The base input is the same premise-hypothesis pair:

```text
premise
hypothesis
```

The Wikidata branch extracts entity candidates from both sides.

Example:

```text
Premise:    Barack Obama was born in Hawaii.
Hypothesis: Obama was born in the United States.
```

Entity candidates:

```text
Barack Obama
Hawaii
Obama
United States
```

## Wikidata Evidence Text

For each entity, v4 builds a short evidence text:

```text
wikidata entity: Barack Obama Q76 former president of the United States instance of human
```

The evidence text is tokenized and embedded with the same Transformer embedding
table used by the main model.

## Wikidata Relation Histogram

v4 also computes a compact relation vector:

```text
wikidata_entity_present
wikidata_same_entity
wikidata_premise_only
wikidata_hypothesis_only
wikidata_has_description
wikidata_has_type
```

This is intentionally small. It tells the classifier whether the example has
entity evidence without making the model depend on large Wikidata subgraphs.

## Wikidata Encoder

For each entity evidence text:

```text
e_i = mean(TokenEmbedding(wikidata_text_i))
```

Then entity vectors are pooled:

```text
h_wikidata = mean({e_i | entity_i exists})
```

If no entity exists, `h_wikidata` becomes a zero-like fallback vector after mask
pooling.

## Feature Composition

v4 starts from the full v2 feature vector:

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
  diagnostic_relation_histogram
]
```

Then it adds Wikidata composition:

```text
φ_wikidata =
[
  h_wikidata;
  |h_pair - h_wikidata|;
  h_pair * h_wikidata;
  cos(h_pair, h_wikidata);
  wikidata_relation_histogram
]
```

Final input:

```text
φ_v4 = [φ_v2; φ_wikidata]
```

## Heads And Loss

v4 keeps the v2 heads:

- 3-class NLI head;
- binary entailment/not_entailment head;
- diagnostic relation evidence head.

Loss:

```text
L = L_3class + λ * (L_binary + L_evidence)
```

## Suggested Figure

Use three branches:

```text
Premise + Hypothesis
        |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
Transformer Encoder   Diagnostic Graph Builder   Wikidata Entity Branch
        |                      |                      |
        v                      v                      v
 h_pair, h_p, h_h       h_graph + r_diag       h_wikidata + r_wiki
        |                      |                      |
        +----------------------+----------------------+
                               |
                               v
                    Feature Composition v4
                               |
             +-----------------+-----------------+
             |                 |                 |
             v                 v                 v
       3-Class NLI Head   Binary AX-b Head   Evidence Head
```

## Expected Benefit

v4 should help most on examples where entailment depends on entities or entity
types:

- people and aliases;
- countries, cities, and locations;
- organizations;
- events;
- taxonomic identity or type compatibility.

It is not expected to solve pure negation or monotonicity by itself; those are
still mostly handled by the diagnostic graph branch.
