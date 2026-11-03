# LogicPath ablation on InferBR

This experiment measures the contribution of the LogicPath components in the
same Portuguese setting used by the previous project comparison. It uses one
fixed seed (`1007`), the same Portuguese encoder, and the same training
hyperparameters. Training and validation use the union of InferBR, ASSIN, and
ASSIN2, while testing is performed separately on the three test manifests and
also aggregated for the overall comparison.

## Conditions

| Variant | Transformer | State graph | Transformations | Wiktionary |
| --- | ---: | ---: | ---: | ---: |
| `logicpath_ablation_transformer` | yes | no | no | no |
| `logicpath_ablation_states` | yes | premise/hypothesis states | no | no |
| `logicpath_ablation_transformations` | yes | premise/hypothesis states | yes | no |
| `logicpath_v2` | yes | full LogicPath graph | yes | yes |

The state-only condition contains the two sentence-level nodes. The
transformation condition adds operation nodes detected from the pair. The full
condition restores lexical units, operation nodes, and Wiktionary entry/gloss
evidence. The classifier and auxiliary transformation heads remain unchanged,
so the comparison isolates graph information rather than changing the model
capacity.

## Surrogate analysis

For the final `logicpath_v2` model, the test-set fusion vector is exposed as
six named blocks:

1. Transformer pair representation.
2. Pooled graph representation.
3. Absolute pair/graph difference.
4. Pair/graph element-wise product.
5. Transformation probabilities.
6. Inverse-transformation probabilities.

A logistic-regression surrogate is fitted to reproduce the neural model's
predicted classes. Each block is permuted independently on the surrogate test
partition. The drop in surrogate fidelity is the block's importance. Results
are saved to `surrogate/surrogate_analysis.json` and
`surrogate/surrogate_analysis.csv` inside the `logicpath_v2/seed_1007` run.

## Run

From the repository root:

```bash
poetry run graphnli-pipeline configs/experiment_0006.yaml 2>&1 | tee logs/experiment_0006.log
```

The output is written to:

```text
runs/experiment_0006/portuguese_inferbr_ablation/
```

The four model results are under the Portuguese model directory, followed by
their variant and `seed_1007`. The full model's surrogate files are created
only after its InferBR test evaluation finishes.
