#!/bin/bash
# Waits for machine A's main experiment_0014 process to finish, then runs the cross-lingual
# lexicon ablation (added 2026-11-25). Run inside tmux.
cd ~/GraphNLI-AXb
echo "$(date) waiting for PID 2157185 (experiment_0014 main queue on A)"
while kill -0 2157185 2>/dev/null; do sleep 60; done
echo "$(date) main queue finished; starting cross-lingual lexicon ablation"
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
poetry run graphnli-pipeline configs/experiment_0014.yaml \
  --only xling_large_en_to_pt_lex_no_wiktionary,xling_large_en_to_pt_lex_no_wordnet_ptlkb,xling_large_pt_to_en_lex_no_wiktionary,xling_large_pt_to_en_lex_no_wordnet_ptlkb \
  2>&1 | tee -a logs/experiment_0014_A_xllex.log
