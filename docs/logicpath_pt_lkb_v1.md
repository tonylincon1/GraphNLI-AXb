# LogicPath PT-LKB v1

`logicpath_pt_lkb_v1` is the Portuguese version of `logicpath_v2` with a
different lexical evidence source. The logical transformation graph remains
the same, but the external lexical nodes are retrieved from
PT-LexicalSemantics rather than Wiktionary.

## What changes

The model still creates nodes for the premise, hypothesis, lexical units and
detected transformations such as negation, conjunction, conditionals,
quantifier changes, numeric changes and temporal changes. It also adds nodes
such as:

```text
PT-LKB entry: médico
PT-LKB gloss: médico hypernym profissional_da_saúde resources=3
```

The PT-LKB relation types include synonymy, hypernymy, part-of, membership,
origin, purpose, causation and related lexical relations. The graph encoder,
typed edges, auxiliary transformation objectives and final feature fusion are
the same as in `logicpath_v2`; only the lexical cache is changed.

## Build the cache

Create the cache from the Portuguese training split to avoid using lexical
resources derived from evaluation examples:

```bash
poetry run python scripts/build_pt_lkb_cache.py \
  --input data/raw/pt_lexical_semantics/triplos_pesados_norm.tsv.zip \
  --manifests data/raw/portuguese/inferbr_train.jsonl \
  --out data/cache/pt_lkb_relations.jsonl \
  --max-terms 8 \
  --min-resources 1
```

Then add both variants to the Portuguese pipeline:

```yaml
variants: [pair_cls, logicpath_v2, logicpath_pt_lkb_v1]
training:
  wiktionary_cache: data/cache/wiktionary_entries_pt.jsonl
  pt_lkb_cache: data/cache/pt_lkb_relations.jsonl
```

`logicpath_v2` uses Wiktionary and `logicpath_pt_lkb_v1` uses PT-LKB. This
keeps the architectural comparison controlled while measuring the effect of
the Portuguese lexical resource.
