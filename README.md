# GraphNLI-AXb

Workspace for a graph/Transformer architecture aimed at AX-b-style broad-coverage
NLI diagnostics.

## Initial Data

The first data layer is deliberately conservative and comparison-friendly:

- `superglue_axb`: official SuperGLUE AX-b Broadcoverage Diagnostics split.
- `glue_ax`: original GLUE diagnostic set with fine-grained linguistic tags.
- `xnli`: official XNLI 1.0 multilingual NLI dev/test data across 15 languages.

Run:

```bash
poetry run python scripts/download_datasets.py
```

Download MultiNLI for the reference-compatible training protocol:

```bash
poetry run python scripts/download_datasets.py --with-mnli
```

`--with-optional` downloads both MultiNLI and XNLI-MT. XNLI-MT is only needed
for later translate-train/translate-test comparisons.

## Direction

GloSeGAT starts from a target word, retrieves WordNet/OMW candidate senses, and
uses a small GAT over candidate synsets/glosses. This project starts from a
premise-hypothesis pair instead: the graph should represent lexical, semantic,
entity, polarity, quantifier, and syntax-level evidence that can explain an
entailment or contradiction decision.

## Training

For a Google Colab run with the heavier English ALBERT XXLarge checkpoint, see
[notebooks/colab_albert_xxlarge.ipynb](notebooks/colab_albert_xxlarge.ipynb).
It installs the Poetry project, downloads the official data, trains
`pair_cls,piig_v3`, saves PIIG explanation graphs, and packages the incremental
experiment directory for download.

Initial one-seed comparison:

```bash
poetry run graphnli-train \
  --models pair_cls,bi_encoder,kbert_injection,kganet_rel_gat,kgnli_dual,dige_v1,dige_v2,dige_v3,dige_v4,dige_v5,dige_v6,dige_v7,dige_v8,dige_v9,piig_v1,piig_v2,piig_v3,piig_v4 \
  --seeds 42 \
  --epochs 1 \
  --selection-metric binary/mcc \
  --patience 2 \
  --output-dir runs
```

The default `--training-protocol mnli` follows the reference-compatible split:
MultiNLI `train` for optimization, MultiNLI `dev_matched` for checkpoint
selection and early stopping, then XNLI `test` and SuperGLUE AX-b only for
final evaluation. AX-b is never used to select a checkpoint. To reproduce the
older exploratory runs that trained on a split of XNLI dev, pass
`--training-protocol xnli_dev_legacy`.

For practical iteration on a local machine, training defaults to a deterministic
stratified subset of 60,000 MultiNLI examples (20,000 per NLI class). Use
`--train-limit 0` for the complete 392,702-example MultiNLI training split.

By default, evaluation writes independent XNLI test metrics for all 15 official
languages: Arabic, Bulgarian, German, Greek, English, Spanish, French, Hindi,
Russian, Swahili, Thai, Turkish, Urdu, Vietnamese, and Chinese. This evaluates
75,150 additional sentence pairs; pass `--eval-languages en` for a quicker
English-only iteration, or any comma-separated subset such as `en,es,fr,de`.
MultiNLI, AX-b, and the original GLUE AX diagnostic are English-only; XNLI is
the multilingual evaluation dataset already included in this workspace.

Each run creates a new incremental experiment folder:

```text
runs/
|-- experiment_001/
|   |-- run_config.json
|   |-- summary.json
|   |-- pair_cls/
|   |   `-- seed_42/
|   |       |-- result.json
|   |       |-- predictions_axb.csv
|   |       `-- predictions_xnli_test_en.csv
|   `-- dige_v1/
|       `-- seed_42/
|           |-- result.json
|           |-- predictions_axb.csv
|           `-- predictions_xnli_test_en.csv
`-- experiment_002/
```

The default architecture variants are:

- `pair_cls`: joint Transformer sentence-pair classifier.
- `bi_encoder`: separate premise/hypothesis encoder baseline.
- `kbert_injection`: K-BERT-style baseline that verbalizes retrieved graph evidence and injects it into the text input.
- `kganet_rel_gat`: KGAnet-style relation-aware graph attention baseline with relation-importance bias.
- `kgnli_dual`: KGNLI-style baseline with separate semantic-relation and knowledge-relation branches.
- `dige_v1`: Diagnostic Evidence Graph Adapter, version 1.
- `dige_v2`: Diagnostic Evidence Graph Feature Composition, version 2.
- `dige_v3`: Logic-Scope Diagnostic Evidence Graph Composition, version 3.
- `dige_v4`: Diagnostic Feature Composition with Wikidata entity evidence, version 4.
- `dige_v5`: Diagnostic Feature Composition with relation-aware GAT feature pooling, version 5.
- `dige_v6`: Counterfactual multilingual relation-path graph composition, version 6.
- `dige_v7`: Reliability-calibrated relation routing, version 7.
- `dige_v8`: Counterfactual logic graph composition for AX-b operators, version 8.
- `dige_v9`: Sparse latent-relation graph composition guided by explicit evidence, version 9.
- `piig_v1`: Proof-Informed Information Injection Graph with cross-attention, version 1.
- `piig_v2`: PIIG with verified and typed Wikidata fact-token injection, version 2.
- `piig_v3`: PIIG with scope-conditioned conditional, quantifier, and disjunction evidence, version 3.
- `piig_v4`: PIIG with cached Wiktionary gloss and lexical-relation evidence, version 4.

There are also two simple graph ablations available:

- `lexical_gcn`: lexical evidence GCN.
- `lexical_gat`: lexical relation-aware GAT.

For statistical testing later, pass multiple seeds:

```bash
poetry run graphnli-train \
  --models pair_cls,bi_encoder,kbert_injection,kganet_rel_gat,kgnli_dual,dige_v1,dige_v2,dige_v3,dige_v4,dige_v5,dige_v6,dige_v7,dige_v8,dige_v9,piig_v1,piig_v2,piig_v3 \
  --seeds 1000,1001,1002,1003,1004 \
  --epochs 6 \
  --lr 1e-5 \
  --selection-metric binary/mcc \
  --patience 2 \
  --output-dir runs
```

Flatten the JSON results to CSV:

```bash
poetry run python scripts/summarize_runs.py runs/experiment_001/summary.json
```

Metrics include accuracy, macro/weighted F1, macro precision/recall, MCC,
expected calibration error, Brier score, AX-b binary metrics, and AX-b metrics
grouped by `logic`.

For `piig_v1`, `piig_v2`, and `piig_v3`, each run also saves an attention-attribution graph
for the first example of every AX-b logical subtype under
`piig_v*/seed_*/explanations/axb/`, plus exact values in JSON and an
`index.json` manifest. Use `--no-save-piig-explanations` to disable it or
`--piig-explain-split xnli_test_en` to change the source split.

Checkpoints are selected by `binary/mcc` by default, which matches the AX-b
binary entailment/not-entailment objective. You can override it with
`--selection-metric nli3/macro_f1` when you want XNLI-style 3-class selection.
Early stopping is disabled by default; set `--patience 2` to stop a
model/seed after two validation epochs without improvement.
