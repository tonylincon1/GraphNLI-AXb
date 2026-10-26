# PIIG-PT-LKB v1

`piig_pt_lkb_v1` is the Portuguese lexical-knowledge variant of PIIG. It
keeps the PIIG proof graph, graph attention, and Transformer cross-attention,
but replaces the Wiktionary evidence source with typed relations from
PT-LexicalSemantics.

## Graph content

The graph contains the usual PIIG phrase and logical evidence nodes, plus
lexical nodes retrieved from the PT-LKB cache. A lexical relation is rendered
as a compact node such as:

```text
PT-LKB entry: árvore pt-lkb PT-LKB lexical relations
PT-LKB relation: árvore hypernym faia
```

The relation inventory includes synonymy, hypernymy, part-of, membership,
origin, purpose, causation, description, similarity, and related typed
relations. The original `Recursos` count and BERT/GPT scores are preserved in
the cache for later analysis; the first implementation uses the relation
text as the Transformer node input and the resource itself as the retrieval
filter.

## Training protocol

The cache should be built only from the InferBR training split when it is used
in the Portuguese experiment:

```bash
poetry run python scripts/build_pt_lkb_cache.py \
  --input data/raw/pt_lexical_semantics/triplos_pesados_norm.tsv.zip \
  --manifests data/raw/portuguese/inferbr_train.jsonl \
  --out data/cache/pt_lkb_relations.jsonl \
  --max-terms 8 \
  --min-resources 1
```

Then include the variant in the Portuguese YAML pipeline:

```yaml
variants: [pair_cls, piig_v4, piig_pt_lkb_v1, logicpath_v2, logicpath_pt_lkb_v1]
training:
  wiktionary_cache: data/cache/wiktionary_entries_pt.jsonl
  pt_lkb_cache: data/cache/pt_lkb_relations.jsonl
```

The PT-LKB cache is selected only by `piig_pt_lkb_v1`; `piig_v4` continues to
use the Wiktionary cache. This makes the comparison isolate the lexical
resource rather than changing the entire PIIG architecture.

## Interpretation

PT-LKB is useful for lexical entailment patterns such as hypernymy and
synonymy, but it does not directly encode negation, quantifier scope, or
conditional validity. Therefore the logical nodes remain in the graph and the
PT-LKB nodes act as complementary lexical evidence.
