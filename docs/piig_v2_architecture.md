# PIIG-NLI v2 Architecture

PIIG-NLI v2 retains the phrase, proof, and cross-attention architecture of
`piig_v1`, but replaces the unverified entity fallback with **verified typed
Wikidata fact nodes**.

## Why v2

In experiment 013, no Wikidata cache was present. The v1 fallback still added
an entity-name node for capitalized spans, so that experiment measured
phrase/proof injection rather than factual knowledge injection.

v2 has a strict rule:

```text
no verified cache entry -> no external knowledge node
verified cache entry    -> external entity and fact nodes
```

This makes the factual ablation meaningful.

## Cache Builder v2

`scripts/build_wikidata_cache.py` now:

1. Uses an explicit `User-Agent`, which avoids Wikimedia rejecting Python's
   anonymous `urllib` request with HTTP 403.
2. Retries temporary network failures and 429/5xx responses with backoff.
3. Retrieves claims for `instance of` (P31), `subclass of` (P279), `award
   received` (P166), `occupation` (P106), and `country` (P17).
4. Resolves labels for linked entity IDs.
5. Caches aliases, description, typed facts, and a `verified` marker.
6. Can collect entity candidates from MultiNLI training, AX-b, and XNLI.

Build a practical first cache, including entities seen during training:

```bash
poetry run python scripts/build_wikidata_cache.py \
  --sources mnli,axb,xnli \
  --languages en \
  --max-examples 10000 \
  --max-entities 4 \
  --sleep 0.10
```

The command is resumable: entities already present under any cached alias are
skipped. For a network that still rejects the default identifier, pass your
own descriptive value:

```bash
poetry run python scripts/build_wikidata_cache.py \
  --user-agent "GraphNLI-AXb/0.2 (Antonio research cache)"
```

## Verified Fact Injection

Given a cached entry for an entity, v2 adds a node for the entity description
and individual fact nodes such as:

```text
external knowledge: Marie Curie, physicist and chemist
external fact: Marie Curie award received Nobel Prize
external fact: Nobel Prize instance of award
```

Each fact connects to both phrase anchors through `phrase_knowledge` edges and
to other knowledge/fact nodes through `evidence_knowledge` edges. The existing
proof GAT turns them into feature tokens. The text states query those tokens by
cross-attention, without a fusion gate.

`external_fact_path` is an added graph-derived auxiliary target. It indicates
that at least one verified factual path was available; it is not an NLI label.

## Experiment

```bash
poetry run graphnli-train \
  --models pair_cls,piig_v1,piig_v2 \
  --seeds 42 \
  --model-name bert-base-multilingual-cased \
  --training-protocol mnli \
  --train-limit 20000 \
  --batch-size 24 \
  --epochs 6 \
  --lr 1e-5 \
  --selection-metric nli3/macro_f1 \
  --patience 2 \
  --eval-languages all \
  --output-dir runs
```

Compare v1 and v2 only after recording cache coverage. A simple check is:

```bash
wc -l data/cache/wikidata_entities.jsonl
```

The key v2 success criterion is not merely higher AX-b MCC: it must improve or
maintain calibration while showing a nonzero count of `external_fact_path`
nodes in training and evaluation.

## Evaluation Attribution Graph

`piig_v2` also writes an SVG and JSON for the first AX-b example of every
logical subtype under `explanations/axb/`, with an `index.json` manifest.
External entity and fact nodes are visible only when a verified cache entry was
used, making the visualization a direct coverage check as well as an
explanation artifact.
Node attribution combines text-to-proof cross-attention with relation-aware
GAT attention; edge width represents GAT attention. It is attention-based
inspection rather than a post-hoc GNNExplainer optimization.

## Local Wikidata Strategy

The task-specific cache is already local: after it is built, training and
evaluation make no HTTP calls. A full Wikidata JSON dump is possible, but is
far too large and slow to parse for this project on a typical laptop. Prefer a
local cache populated from the entities in the selected training/evaluation
sets, then pass its path with `--wikidata-cache`.

The official [Wikidata database-download page](https://www.wikidata.org/wiki/Wikidata:Database_download)
does provide weekly JSON dumps and documents line-oriented parsing, but even a
recent compressed full dump is listed there at roughly 130 GiB. That is useful
for a server-side SQLite/indexing pipeline, not as the default PIIG workflow.
