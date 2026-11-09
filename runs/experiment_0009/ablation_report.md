# Ablation report — experiment_0009

## cover_ablation

Full model: `coverage_pl_v1`. Δ = variant − full model, paired by seed (95% bootstrap CI; Wilcoxon signed-rank p). Negative Δ means the removed component helped.

### inferbr_test · nli3/macro_f1

| variant | n | mean ± std | Δ vs full | 95% CI | p | Δ vs pair_cls |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `coverage_pl_v1` | 10 | 0.9223 ± 0.0047 | — | — | — | +0.0047 |
| `coverage_pl_v1_no_symbolic` | 10 | 0.9187 ± 0.0039 | -0.0036 | [-0.0066, -0.0004] | 0.064 | +0.0011 |
| `coverage_pl_v1_mlp_head` | 10 | 0.9171 ± 0.0127 | -0.0052 | [-0.0132, +0.0024] | 0.232 | -0.0005 |
| `coverage_pl_v1_no_conflict` | 10 | 0.9145 ± 0.0094 | -0.0079 | [-0.0139, -0.0025] | 0.037 | -0.0032 |
| `coverage_pl_v1_no_coverage` | 10 | 0.9204 ± 0.0036 | -0.0020 | [-0.0060, +0.0021] | 0.432 | +0.0027 |
| `coverage_pl_v1_no_graph` | 10 | 0.9175 ± 0.0095 | -0.0048 | [-0.0125, +0.0019] | 0.492 | -0.0001 |
| `pair_cls` | 10 | 0.9176 ± 0.0057 | -0.0047 | [-0.0099, +0.0000] | 0.084 | — |

### inferbr_test · binary/mcc

| variant | n | mean ± std | Δ vs full | 95% CI | p | Δ vs pair_cls |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `coverage_pl_v1` | 10 | 0.9054 ± 0.0066 | — | — | — | +0.0155 |
| `coverage_pl_v1_no_symbolic` | 10 | 0.8950 ± 0.0059 | -0.0104 | [-0.0160, -0.0043] | 0.014 | +0.0051 |
| `coverage_pl_v1_mlp_head` | 10 | 0.8981 ± 0.0149 | -0.0073 | [-0.0180, +0.0023] | 0.375 | +0.0082 |
| `coverage_pl_v1_no_conflict` | 10 | 0.8942 ± 0.0117 | -0.0111 | [-0.0174, -0.0055] | 0.002 | +0.0043 |
| `coverage_pl_v1_no_coverage` | 10 | 0.9016 ± 0.0072 | -0.0038 | [-0.0105, +0.0022] | 0.322 | +0.0116 |
| `coverage_pl_v1_no_graph` | 10 | 0.8922 ± 0.0119 | -0.0132 | [-0.0225, -0.0050] | 0.020 | +0.0022 |
| `pair_cls` | 10 | 0.8899 ± 0.0087 | -0.0155 | [-0.0218, -0.0104] | 0.002 | — |

Critical-difference report: `cover_ablation/critical_difference`
