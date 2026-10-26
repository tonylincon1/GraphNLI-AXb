# YAML Training Pipelines

The project now supports three reproducible pipeline families through
`graphnli-pipeline`. The YAML filename is the experiment identifier:

```bash
poetry run python scripts/download_sick.py
poetry run graphnli-pipeline configs/experiment_001.yaml
```

This creates `runs/experiment_001/`. Running the same command again resumes
the named experiment: completed `study.json` files and `result.json` files are
reused.

## Pipeline protocol

`english` trains and validates on English SICK and evaluates on English XNLI
and AX-b. SICK is a compact three-way English NLI corpus with about 10,000
pairs and official TRAIN, TRIAL and TEST partitions. It is used here as a
smaller training source; AX-b remains test-only.

`english_multilingual` uses the same English SICK train/validation data and
evaluates on all 15 official XNLI languages. The multilingual pipeline uses
`xlm-roberta-base` by default, while the English pipeline uses `roberta-base`.

The batch size is configured independently for each pipeline with the
top-level `batch_size` field:

```yaml
pipelines:
  english:
    batch_size: 32
  english_multilingual:
    batch_size: 16
  portuguese:
    batch_size: 32
```

This value overrides `training.batch_size`; the global value is only a
fallback for older configurations. The default Optuna search space does not
tune batch size, so this resource choice stays fixed during tuning and final
runs. To tune it, add `batch_size` to that pipeline's own
`tuning.search_space`.

`portuguese` consumes normalized JSONL manifests for ASSIN, ASSIN2 and
InferBR, concatenating their official training and validation partitions and
keeping their test partitions separate. Each row must contain either
`sentence1`/`sentence2` or `premise`/`hypothesis`, plus a label field. The
loader accepts `label`, `gold_label`, `entailment_judgment` and
`entailment_label`. ASSIN's `paraphrase` is mapped to non-entailment; ASSIN2
and InferBR numeric labels are mapped according to their dataset conventions.

The Portuguese paths in the example YAML are deliberately explicit. Put the
official splits at those paths, or change the paths without changing the
runner.

### PT-LKB variant

`piig_pt_lkb_v1` keeps the PIIG proof graph and Transformer cross-attention,
but uses typed relations from PT-LexicalSemantics instead of the Portuguese
Wiktionary cache. The cache preserves relation type, number of supporting
resources, and the BERT/GPT lexical scores. Build a cache filtered by the
InferBR training split with:

```bash
poetry run python scripts/build_pt_lkb_cache.py \
  --input data/raw/pt_lexical_semantics/triplos_pesados_norm.tsv.zip \
  --manifests data/raw/portuguese/inferbr_train.jsonl \
  --out data/cache/pt_lkb_relations.jsonl \
  --max-terms 8 \
  --min-resources 1
```

The Portuguese pipeline includes this variant alongside `piig_v4`, allowing
the effect of the lexical resource to be measured without changing the
logical evidence graph.

`logicpath_pt_lkb_v1` provides the analogous controlled comparison for
LogicPath. It keeps the transformation graph of `logicpath_v2` and replaces
its Wiktionary evidence nodes with PT-LKB evidence nodes. Use it alongside
`logicpath_v2` to compare Wiktionary and PT-LexicalSemantics with the same
encoder and auxiliary objectives.

To create the manifests from the available dataset mirrors, run:

```bash
poetry run python scripts/download_portuguese_nli.py
```

## Wiktionary caches by language

The cache builder queries the corresponding Wiktionary edition and stores a
separate JSONL cache for each language. For the Portuguese experiment, the
cache is intentionally built only from the InferBR training split, avoiding
lexical information from validation or test examples:

```bash
poetry run python scripts/build_wiktionary_cache.py \
  --languages pt \
  --sources "" \
  --manifests data/raw/portuguese/inferbr_train.jsonl \
  --max-examples 0 \
  --max-entities 8 \
  --sleep 0.10 \
  --out data/cache/wiktionary_entries_pt.jsonl
```

For XNLI, create one file per language in a directory. The `{language}`
placeholder is expanded by the script:

```bash
poetry run python scripts/build_wiktionary_cache.py \
  --languages ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh \
  --sources xnli \
  --max-examples 10000 \
  --max-entities 8 \
  --out data/cache/wiktionary_multilingual/wiktionary_{language}.jsonl
```

The dataset loader accepts a directory as `wiktionary_cache` and reads all its
JSONL files. This is the value used by the multilingual YAML pipeline:
`data/cache/wiktionary_multilingual`.

For English, use the regular sources and an English-specific file:

```bash
poetry run python scripts/build_wiktionary_cache.py \
  --languages en --sources mnli,axb,xnli \
  --max-examples 10000 --max-entities 8 --sleep 0.10 \
  --out data/cache/wiktionary_entries_en.jsonl
```

Use an identifiable `--user-agent` when querying Wiktionary and respect the
request delay. The cache builder is incremental: rerunning it skips terms
already present in the target file. The local full-dump workflow remains
available for English; the API workflow above is the practical route for
Portuguese and the other Wiktionary editions.

## Optuna

Tuning happens independently for every `model_name` and architecture
`variant`. The default is 10 trials using a three-epoch tuning run and the
validation `nli3/macro_f1`. The best hyperparameters are then reused for all
final seeds (10 by default in the example). Each tuning directory also has a
persistent `optuna.db` SQLite study, so an interrupted tuning run continues
with the remaining trials instead of restarting from trial 0. Set `tuning.trials` and
`tuning.epochs` to change the cost, or replace `tuning.search_space` to tune
other parameters.

The artifacts are organized as:

```text
runs/experiment_001/
  config.yaml
  english/
    tuning/roberta-base/pair_cls/study.json
    tuning/roberta-base/pair_cls/optuna.db
    roberta-base/pair_cls/seed_42/result.json
    summary.json
  english_multilingual/
    xlm-roberta-base/piig_v4/seed_42/result.json
  portuguese/
    neuralmind__bert-base-portuguese-cased/pair_cls/seed_42/result.json
  pipeline_summary.json
```

The tuning stage is intentionally per variant: graph variants can have
different optimization behavior from the pair classifier, so a single global
learning rate would make the comparison less controlled.

## Dataset references

- SICK: Marelli et al., *A SICK cure for the evaluation of compositional
  distributional semantic models* (LREC 2014); official data record at
  [Zenodo](https://zenodo.org/records/2787612).
- XNLI: Conneau et al., *XNLI: Evaluating Cross-lingual Sentence
  Representations* (EMNLP 2018); [official repository](https://github.com/facebookresearch/XNLI).
- ASSIN2: Real, Fonseca and Gonçalo Oliveira, *The ASSIN 2 shared task: a
  quick overview* (STIL 2019); [official task page](https://sites.google.com/view/assin2).
- InferBR: Bencke et al., *InferBR: A Natural Language Inference Dataset in
  Portuguese* (LREC-COLING 2024); [ACL Anthology](https://aclanthology.org/2024.lrec-main.793/).
