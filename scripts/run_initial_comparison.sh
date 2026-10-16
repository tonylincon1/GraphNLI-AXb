#!/usr/bin/env bash
set -euo pipefail

poetry run graphnli-train \
  --models pair_cls,bi_encoder,kbert_injection,kganet_rel_gat,kgnli_dual,dige_v1,dige_v2,dige_v3,dige_v4,dige_v5,dige_v6,dige_v7,dige_v8,dige_v9,piig_v1,piig_v2,piig_v3 \
  --seeds 42 \
  --epochs 1 \
  --train-languages en \
  --eval-languages en \
  --output-dir runs
