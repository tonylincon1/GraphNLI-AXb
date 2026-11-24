#!/usr/bin/env python3
"""Slide tables for experiment 0014 (partial results allowed), next to the 0011 references.

    poetry run python scripts/make_presentation_0014_tables.py   # writes presentation/tables_0014/*.tex

Same conventions as scripts/make_paper_tables.py: binary MCC x100, mean with the seed SD in small
type, \\best on the best mean per column, dagger = paired Wilcoxon p < .05 against the row marked
as reference (same seeds). Systems with fewer than 10 seeds get their n in the label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import make_paper_tables as mpt

ROOT = mpt.ROOT
OUT = ROOT / "presentation" / "tables_0014"
E11, E14 = ROOT / "runs" / "experiment_0011", ROOT / "runs" / "experiment_0014"
mpt.N_SEEDS = 1  # accept partial seed sets

EN, PT = mpt.EN_TESTS, mpt.PT_TESTS
EN_OOD = EN[1:]


def load(exp: Path, pipeline: str, variant: str, tests):
    mpt.EXP = exp
    return mpt.scores([pipeline], variant, tests)


def table(rows, tests, ref: int, midrules=()) -> str:
    """rows: (label, exp, pipeline, variant); ref: index of the reference row for the dagger."""
    data = [load(e, p, v, tests) for _, e, p, v in rows]
    n = len(tests)
    lines = [f"\\begin{{tabular}}{{@{{}}l{'c' * n}@{{}}}}", "\\toprule",
             "Sistema & " + " & ".join(h for _, h, _ in tests) + " \\\\", "\\midrule"]
    cols = {}
    for key, _, _ in tests:
        means = {i: np.mean([s[key] for s in d.values()]) for i, d in enumerate(data)
                 if d and all(s.get(key) is not None for s in d.values())}
        best = max(means, key=means.get) if means else None
        for i, d in enumerate(data):
            if i not in means:
                cols.setdefault(i, []).append("--")
                continue
            mark = ""
            if i > ref and data[ref]:
                p = mpt.paired_p(d, data[ref], key)
                mark = "$^{\\dagger}$" if p is not None and p < 0.05 else ""
            cols.setdefault(i, []).append(mpt.cell([s[key] for s in d.values()], best == i, mark))
    for i, (label, *_rest) in enumerate(rows):
        if i in midrules:
            lines.append("\\midrule")
        k = len(data[i]) if data[i] else 0
        label = label + (f" {{\\tiny($n{{=}}{k}$)}}" if 0 < k < 10 else "")
        lines.append(f"{label} & " + " & ".join(cols[i]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


ALB = "PORTULAN__albertina-900m-portuguese-ptbr-encoder"
XLR = "FacebookAI__xlm-roberta-large"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "lex_en": table([
            ("\\cls{}", E11, "sick_en_ablation", "pair_cls"),
            ("WordNet + Wiktionary (0011)", E11, "sick_en_ablation", "coverage_pl_v1"),
            ("só WordNet", E14, "sick_en_lex_wordnet", "coverage_pl_v1"),
            ("só Wiktionary", E14, "sick_en_lex_wiktionary", "coverage_pl_v1"),
        ], EN, ref=0, midrules=(1, 2)),
        "pt_large": table([
            ("BERTimbau-base \\cls{} (0011)", E11, "inferbr_pt_ablation", "pair_cls"),
            ("BERTimbau-base \\model{} (0011)", E11, "inferbr_pt_ablation", "coverage_pl_v1"),
            ("Albertina-900M \\cls{}", E14, "inferbr_pt_large_ablation", "pair_cls"),
            ("Albertina-900M \\model{}", E14, "inferbr_pt_large_ablation", "coverage_pl_v1"),
            ("Albertina-900M \\model{} $-$ conflito", E14, "inferbr_pt_large_ablation", "coverage_pl_v1_no_conflict"),
            ("Albertina-900M \\model{} $-$ léxico", E14, "inferbr_pt_large_ablation", "coverage_pl_v1_no_symbolic"),
            ("Albertina-900M \\model{} $-$ grafo", E14, "inferbr_pt_large_ablation", "coverage_pl_v1_no_graph"),
        ], PT, ref=2, midrules=(2,)),
        "xl_large_en": table([
            ("XLM-R-base \\cls{} (0011)", E11, "xling_en_to_pt_ablation", "pair_cls"),
            ("XLM-R-base \\model{} (0011)", E11, "xling_en_to_pt_ablation", "coverage_pl_v1"),
            ("XLM-R-large \\cls{}", E14, "xling_large_en_to_pt_ablation", "pair_cls"),
            ("XLM-R-large \\model{}", E14, "xling_large_en_to_pt_ablation", "coverage_pl_v1"),
            ("XLM-R-large $-$ conflito", E14, "xling_large_en_to_pt_ablation", "coverage_pl_v1_no_conflict"),
            ("XLM-R-large $-$ léxico", E14, "xling_large_en_to_pt_ablation", "coverage_pl_v1_no_symbolic"),
        ], EN, ref=2, midrules=(2,)),
        "xl_large_pt": table([
            ("XLM-R-large \\cls{}", E14, "xling_large_en_to_pt_ablation", "pair_cls"),
            ("XLM-R-large \\model{}", E14, "xling_large_en_to_pt_ablation", "coverage_pl_v1"),
            ("XLM-R-large $-$ conflito", E14, "xling_large_en_to_pt_ablation", "coverage_pl_v1_no_conflict"),
            ("XLM-R-large $-$ léxico", E14, "xling_large_en_to_pt_ablation", "coverage_pl_v1_no_symbolic"),
            ("XLM-R-large $-$ grafo", E14, "xling_large_en_to_pt_ablation", "coverage_pl_v1_no_graph"),
            ("XLM-R-large $-$ cobertura", E14, "xling_large_en_to_pt_ablation", "coverage_pl_v1_no_coverage"),
            ("XLM-R-large MLP no lugar da regra", E14, "xling_large_en_to_pt_ablation", "coverage_pl_v1_mlp_head"),
        ], PT, ref=0, midrules=(1,)),
    }
    for name, tex in tables.items():
        (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
        print(OUT / f"{name}.tex")


if __name__ == "__main__":
    main()
