#!/usr/bin/env python3
"""Build the LaTeX result tables of the paper from runs/experiment_0011.

    python scripts/make_paper_tables.py            # writes paper/tables/*.tex

Cells are filled only when all seeds of a system are finished (10 by default); otherwise
they are left blank, so the script can be rerun after every pull.

Metric (x100): Matthews correlation (MCC) of the binary decision entailment vs. non-entailment,
for every test set. It is the only metric defined for all of them (AX-b, ASSIN and ASSIN 2 are
binary, and models trained on SICK label most Breaking NLI contradictions as neutral, following
SICK's annotation convention), and it is robust to label imbalance. For models trained on SICK,
ASSIN 2 is scored on the pairs that are not translated from SICK TRAIN/TRIAL
(data/raw/challenge/assin2_test_sick_provenance.json).

Main table: mean, with the seed standard deviation in small type; best per column in bold;
dagger = paired exact Wilcoxon p < 0.05 against pair_cls (same seeds).
Ablation table: asterisk = Holm-corrected Wilcoxon p < 0.05 against the full model, over
the five ablations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "runs" / "experiment_0011"
OUT = ROOT / "paper" / "tables"
N_SEEDS = 10
PROVENANCE = ROOT / "data" / "raw" / "challenge" / "assin2_test_sick_provenance.json"

# (key, header, metric); key "assin2_clean" is computed from predictions.
EN_TESTS = [("sick_test", "SICK", "binary/mcc"), ("axb", "AX-b", "binary/mcc"),
            ("breaking_nli", "Break.", "binary/mcc"), ("conjnli_dev", "Conj.", "binary/mcc"),
            ("fracas_en", "FraCaS", "binary/mcc")]
PT_TESTS = [("inferbr_test", "InferBR", "binary/mcc"), ("assin_test", "ASSIN", "binary/mcc"),
            ("assin2_test", "ASSIN\\,2", "binary/mcc"), ("sick_br_test", "SICK-BR", "binary/mcc")]

METRIC_LABEL = {"nli3/macro_f1": "macro-F1", "nli3/accuracy": "acc.", "binary/mcc": "MCC"}

MAIN_SYSTEMS = [("pair_cls", "\\cls{}"), ("kbert_injection", "K-BERT"), ("kganet_rel_gat", "KGAnet"),
                ("kgnli_dual", "KGNLI"), ("coverage_pl_v1", "\\model{}")]
ABLATIONS = [("coverage_pl_v1", "\\model{} (full)"), ("coverage_pl_v1_no_conflict", "$-$ conflict"),
             ("coverage_pl_v1_no_symbolic", "$-$ lexicon"), ("coverage_pl_v1_no_graph", "$-$ graph"),
             ("coverage_pl_v1_no_coverage", "$-$ coverage"), ("coverage_pl_v1_mlp_head", "MLP head"),
             ("pair_cls", "\\cls{}")]

# Cross-lingual ablation: only transfer to the other language is reported.
SAME_LANGUAGE = {"Train SICK (EN)": {k for k, _, _ in EN_TESTS},
                 "Train InferBR (PT)": {k for k, _, _ in PT_TESTS}}

# setting -> list of (row-group label, [pipelines that may hold a system], tests)
SETTINGS = {
    "en": [(None, ["sick_en_ablation", "sick_en_graph_baselines", "sick_en_kbert"], EN_TESTS)],
    "pt": [(None, ["inferbr_pt_ablation", "inferbr_pt_graph_baselines"], PT_TESTS)],
    "xl": [("Train SICK (EN)", ["xling_en_to_pt_ablation", "xling_en_to_pt_graph_baselines"], EN_TESTS + PT_TESTS),
           ("Train InferBR (PT)", ["xling_pt_to_en_ablation", "xling_pt_to_en_graph_baselines"], EN_TESTS + PT_TESTS)],
}


def load_provenance() -> dict:
    if not PROVENANCE.exists():
        return {}
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


PROV = load_provenance()


def assin2_clean_mcc(seed_dir: Path, metric: str = "binary/mcc") -> float | None:
    path = seed_dir / "predictions_assin2_test.csv"
    if not path.exists() or not PROV:
        return None
    gold, pred = [], []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if PROV.get(row["uid"].rsplit("-", 1)[1]) in {"TEST", "nomatch"}:
                gold.append(row["label_binary"])
                pred.append(row["prediction_binary"])
    if metric == "binary/accuracy":
        return float(np.mean([g == q for g, q in zip(gold, pred)]))
    return matthews_corrcoef(gold, pred)


def scores(pipelines: list[str], variant: str, tests) -> dict | None:
    """seed -> {test: value} for the first pipeline holding the variant; None if incomplete."""
    for pipeline in pipelines:
        seed_dirs = sorted((EXP / pipeline).glob(f"*/{variant}/seed_*"))
        done = [d for d in seed_dirs if (d / "result.json").exists()]
        if not done:
            continue
        if len(done) < N_SEEDS:
            return None
        out = {}
        sick_trained = "sick" in pipeline or "en_to_pt" in pipeline
        for d in done:
            result = json.loads((d / "result.json").read_text(encoding="utf-8"))
            values = {}
            for key, _, metric in tests:
                if key == "assin2_test" and sick_trained:
                    values[key] = assin2_clean_mcc(d, metric)
                else:
                    values[key] = result["eval"].get(key, {}).get(metric)
            out[int(result["seed"])] = values
        return out
    return None


def cell(values: list[float], bold: bool, mark: str) -> str:
    mean, sd = 100 * np.mean(values), 100 * np.std(values, ddof=1)
    text = f"{mean:.1f}"
    if bold:
        text = f"\\textbf{{{text}}}"
    return f"{text}{mark}\\,{{\\tiny$\\pm${sd:.1f}}}"


def delta_cell(variant: dict, full: dict, key: str, mark: str) -> str:
    """Mean paired difference to the full model (x100); negative values in red."""
    seeds = sorted(set(variant) & set(full))
    delta = 100 * np.mean([variant[s][key] - full[s][key] for s in seeds])
    if round(delta, 1) < 0:
        return f"\\textcolor{{red}}{{$-${abs(delta):.1f}{mark}}}"
    return f"$+${delta:.1f}{mark}"


def paired_p(a: dict, b: dict, key: str) -> float | None:
    seeds = sorted(set(a) & set(b))
    x = np.array([a[s][key] for s in seeds], dtype=float)
    y = np.array([b[s][key] for s in seeds], dtype=float)
    if len(seeds) < 5 or np.isnan(x).any() or np.isnan(y).any():
        return None
    return float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0


def holm(ps: dict, m: int | None = None) -> dict:
    """Holm adjustment; m = family size (missing members count as p = 1, i.e. conservative)."""
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    m, running, out = max(m or 0, len(items)), 0.0, {}
    for i, (p, k) in enumerate(items):
        running = max(running, (m - i) * p)
        out[k] = min(1.0, running)
    return out


MAIN_METRICS = [("binary/accuracy", "Accuracy"), ("binary/mcc", "MCC")]


def with_metric(tests, metric: str):
    return [(key, header, metric) for key, header, _ in tests]


def main_cells(pipelines, tests, systems) -> dict:
    """(variant, metric) -> list of cells; bold = best per column, dagger = Wilcoxon vs pair_cls."""
    out = {}
    for metric, _ in MAIN_METRICS:
        mt = with_metric(tests, metric)
        data = {v: scores(pipelines, v, mt) for v, _ in systems}
        for key, _, _ in mt:
            means = {v: np.mean([s[key] for s in d.values()]) for v, d in data.items()
                     if d and all(s[key] is not None for s in d.values())}
            best = max(means, key=means.get) if means else None
            for v, _ in systems:
                d = data[v]
                if not d or any(s[key] is None for s in d.values()):
                    out.setdefault((v, metric), []).append("")
                    continue
                mark = ""
                if v != "pair_cls" and data["pair_cls"]:
                    p = paired_p(d, data["pair_cls"], key)
                    mark = "$^{\\dagger}$" if p is not None and p < 0.05 else ""
                out.setdefault((v, metric), []).append(cell([s[key] for s in d.values()], best == v, mark))
    return out


def main_table(groups, tests, systems) -> str:
    """One row per system; column blocks MCC | accuracy over the same test sets."""
    n = len(tests)
    header = " & ".join(h for _, h, _ in tests)
    lines = [f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}l{'c' * n}{'c' * n}@{{}}}}",
             "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{label}}}" for _, label in MAIN_METRICS) + " \\\\",
             f"\\cmidrule(lr){{2-{n + 1}}}\\cmidrule(l){{{n + 2}-{2 * n + 1}}}",
             f"System & {header} & {header} \\\\",
             "\\midrule"]
    for i, (group, pipelines) in enumerate(groups):
        if i:
            lines.append("\\midrule")
        if group:
            lines.append(f"\\multicolumn{{{2 * n + 1}}}{{@{{}}l}}{{\\textit{{{group}}}}} \\\\")
        cells = main_cells(pipelines, tests, systems)
        for v, label in systems:
            row = [c for metric, _ in MAIN_METRICS for c in cells[(v, metric)]]
            lines.append(f"{label} & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def block(group, pipelines, tests, systems, mode: str) -> list[str]:
    data = {v: scores(pipelines, v, tests) for v, _ in systems}
    ref = "pair_cls" if mode == "main" else "coverage_pl_v1"
    marks = {v: {} for v, _ in systems}
    for key, _, _ in tests:
        if mode == "main":
            for v, _ in systems:
                if v != ref and data[v] and data[ref]:
                    p = paired_p(data[v], data[ref], key)
                    marks[v][key] = "$^{\\dagger}$" if p is not None and p < 0.05 else ""
        else:
            raw = {v: paired_p(data[v], data[ref], key) for v, _ in systems
                   if v not in (ref, "pair_cls") and data[v] and data[ref]}
            family = sum(v not in (ref, "pair_cls") for v, _ in systems)
            for v, p in holm(raw, family).items():
                marks[v][key] = "$^{*}$" if p < 0.05 else ""
            if data.get("pair_cls") and data[ref]:
                p = paired_p(data["pair_cls"], data[ref], key)
                marks["pair_cls"][key] = "$^{\\dagger}$" if p is not None and p < 0.05 else ""
    best = {}
    for key, _, _ in tests:
        means = {v: np.mean([s[key] for s in d.values()]) for v, d in data.items()
                 if d and all(s[key] is not None for s in d.values())}
        best[key] = max(means, key=means.get) if means else None
    lines = []
    if group:
        lines.append(f"\\multicolumn{{{len(tests) + 1}}}{{@{{}}l}}{{\\textit{{{group}}}}} \\\\")
    for v, label in systems:
        if mode == "ablation" and v == "pair_cls":
            lines.append("\\midrule")
        cells = []
        for key, _, _ in tests:
            d = data[v]
            if key in SAME_LANGUAGE.get(group, set()):
                cells.append("--")
            elif not d or any(s[key] is None for s in d.values()):
                cells.append("")
            elif mode == "ablation" and v != ref:
                full = data[ref]
                if not full or any(s[key] is None for s in full.values()):
                    cells.append("")
                else:
                    cells.append(delta_cell(d, full, key, marks[v].get(key, "")))
            elif mode == "ablation":
                cells.append(cell([s[key] for s in d.values()], False, ""))
            else:
                cells.append(cell([s[key] for s in d.values()], best[key] == v, marks[v].get(key, "")))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return lines


def table(setting: str, systems, mode: str) -> str:
    groups = SETTINGS[setting]
    tests = groups[0][2]
    header = " & ".join(h for _, h, _ in tests)
    off = 1
    if setting == "xl":
        header_rows = [" & " * off + f"\\multicolumn{{{len(EN_TESTS)}}}{{c}}{{English tests}} & "
                       f"\\multicolumn{{{len(PT_TESTS)}}}{{c}}{{Portuguese tests}} \\\\",
                       f"\\cmidrule(lr){{{off + 1}-{off + len(EN_TESTS)}}}"
                       f"\\cmidrule(l){{{off + len(EN_TESTS) + 1}-{off + len(tests)}}}"]
    else:
        header_rows = []
    # Full text width for every panel, so all panels of a table have the same size.
    extra = ""
    lead = "System & "
    lines = [f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}l{extra}{'c' * len(tests)}@{{}}}}",
             "\\toprule", *header_rows,
             f"{lead}{header} \\\\",
             "\\midrule"]
    for i, (group, pipelines, group_tests) in enumerate(groups):
        if i:
            lines.append("\\midrule")
        lines.extend(block(group, pipelines, group_tests, systems, mode))
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- error analysis
AXB_CATEGORIES = ROOT / "data" / "raw" / "challenge" / "axb_categories.json"
ANALYSIS_SYSTEMS = [("pair_cls", "\\cls{}"), ("coverage_pl_v1", "\\model{}")]
LABEL_NAMES = ["\\E{}", "\\N{}", "\\C{}"]


def predictions(pipeline: str, variant: str, test: str) -> list[list[dict]] | None:
    """Per-seed prediction rows; None unless all seeds are finished."""
    seed_dirs = sorted((EXP / pipeline).glob(f"*/{variant}/seed_*"))
    done = [d for d in seed_dirs if (d / "result.json").exists() and (d / f"predictions_{test}.csv").exists()]
    if len(done) < N_SEEDS:
        return None
    out = []
    for d in done:
        with (d / f"predictions_{test}.csv").open(encoding="utf-8") as handle:
            out.append(list(csv.DictReader(handle)))
    return out


AXB_COARSE = [("Lexical Semantics", "Lexical semantics"), ("Predicate-Argument Structure", "Predicate--argument structure"),
              ("Logic", "Logic"), ("Knowledge", "Knowledge")]


def binary_mcc(rows: list[dict]) -> float:
    return matthews_corrcoef([r["label_binary"] for r in rows], [r["prediction_binary"] for r in rows])


def axb_table(pipeline: str) -> str:
    """AX-b by coarse category and its fine-grained phenomena: n, [CLS], COVER, paired difference.
    A pair can carry several phenomena. Binary MCC x100, mean over seeds."""
    cats = json.loads(AXB_CATEGORIES.read_text(encoding="utf-8")) if AXB_CATEGORIES.exists() else {}
    preds = {v: predictions(pipeline, v, "axb") for v, _ in ANALYSIS_SYSTEMS}
    complete = all(p is not None for p in preds.values())
    lines = ["\\begin{tabular}{@{}l r ccc@{}}", "\\toprule",
             "Phenomenon & $n$ & " + " & ".join(label for _, label in ANALYSIS_SYSTEMS) + " & $\\Delta$ \\\\"]

    def row(label: str, member) -> str:
        cells, values = [], {}
        n = sum(1 for idx in cats if member(cats[idx]))
        for v, _ in ANALYSIS_SYSTEMS:
            if preds[v] is None:
                cells.append("")
                continue
            per_seed = [binary_mcc([r for r in rows if member(cats.get(r["uid"].split("-", 1)[1], {}))])
                        for rows in preds[v]]
            values[v] = np.array(per_seed) * 100
            mean = values[v].mean()
            cells.append(f"$-${abs(mean):.1f}" if round(mean, 1) < 0 else f"{abs(round(mean, 1)):.1f}")
        if complete:
            delta = float(np.mean(values["coverage_pl_v1"] - values["pair_cls"]))
            cells.append(f"\\textcolor{{red}}{{$-${abs(delta):.1f}}}" if round(delta, 1) < 0 else f"$+${delta:.1f}")
        else:
            cells.append("")
        return f"{label} & {n} & " + " & ".join(cells) + " \\\\"

    for key, label in AXB_COARSE:
        lines.append("\\midrule")
        lines.append(row(f"\\textbf{{{label}}}", lambda c, k=key: k in c))
        fine = sorted({f for c in cats.values() if key in c for f in c[key].split(";")})
        for f in fine:
            lines.append(row(f"\\quad {f}", lambda c, k=key, f=f: f in c.get(k, "").split(";")))
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def confusion_table(panels: list[tuple[str, str, str]]) -> str:
    """panels: (title, pipeline, test). Mean confusion counts per seed for [CLS], and COVER minus [CLS]
    next to it (red where COVER is worse)."""
    systems = [("pair_cls", "\\cls{}"), ("coverage_pl_v1", "\\model{}")]
    lines = ["\\begin{tabular}{@{}ll ccc c ccc@{}}", "\\toprule",
             " & & \\multicolumn{3}{c}{" + systems[0][1] + "} & & \\multicolumn{3}{c}{" + systems[1][1]
             + " $-$ " + systems[0][1] + "} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(l){7-9}",
             " & Gold & " + " & ".join(LABEL_NAMES) + " & & " + " & ".join(LABEL_NAMES) + " \\\\"]
    for title, pipeline, test in panels:
        lines.append("\\midrule")
        mats = {}
        for v, _ in systems:
            preds = predictions(pipeline, v, test)
            if preds is None:
                continue
            cms = []
            for rows in preds:
                cm = np.zeros((3, 3))
                for r in rows:
                    g, q = int(r["label_3"]), int(r["prediction_3"])
                    if 0 <= g < 3 and 0 <= q < 3:
                        cm[g, q] += 1
                cms.append(cm)
            mats[v] = np.mean(cms, axis=0)
        for i, gold in enumerate(LABEL_NAMES):
            first = f"\\multirow{{3}}{{*}}{{{title}}}" if i == 0 else ""
            cells = []
            for j, (v, _) in enumerate(systems):
                if j:
                    cells.append("")
                if v in mats and j == 0:
                    cells.extend(f"{x:.0f}" if k != i else f"\\textbf{{{x:.0f}}}" for k, x in enumerate(mats[v][i]))
                elif v in mats and systems[0][0] in mats:
                    # COVER minus [CLS]; red where COVER is worse (fewer correct on the diagonal,
                    # more errors off the diagonal).
                    for k in range(3):
                        d = int(round(mats[v][i][k] - mats[systems[0][0]][i][k]))
                        text = "0" if d == 0 else (f"$+${d}" if d > 0 else f"$-${-d}")
                        worse = d < 0 if k == i else d > 0
                        if k == i:
                            text = f"\\textbf{{{text}}}"
                        cells.append(f"\\textcolor{{red}}{{{text}}}" if worse else text)
                else:
                    cells.extend(["", "", ""])
            lines.append(f"{first} & {gold} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for setting in SETTINGS:
        if setting != "xl":
            (group, pipelines, tests), = SETTINGS[setting]
            (OUT / f"main_{setting}.tex").write_text(main_table([(group, pipelines)], tests, MAIN_SYSTEMS), encoding="utf-8")
        (OUT / f"ablation_{setting}.tex").write_text(table(setting, ABLATIONS, "ablation"), encoding="utf-8")
    # Cross-lingual: main table = transfer to the other language; appendix = same-language tests.
    (_, en_to_pt, _), (_, pt_to_en, _) = SETTINGS["xl"]
    for name, pipelines, tests in [("main_xl_en_to_pt", en_to_pt, PT_TESTS), ("main_xl_pt_to_en", pt_to_en, EN_TESTS),
                                   ("appendix_xl_sick_en", en_to_pt, EN_TESTS),
                                   ("appendix_xl_inferbr_pt", pt_to_en, PT_TESTS)]:
        (OUT / f"{name}.tex").write_text(main_table([(None, pipelines)], tests, MAIN_SYSTEMS), encoding="utf-8")
    # Presentation: cross-lingual ablation, one direction and only the other language's tests.
    for name, (group, pipelines, _), tests in [("pres_ablation_xl_en_to_pt", SETTINGS["xl"][0], PT_TESTS),
                                               ("pres_ablation_xl_pt_to_en", SETTINGS["xl"][1], EN_TESTS)]:
        body = block(None, pipelines, tests, ABLATIONS, "ablation")
        head = [f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}l{'c' * len(tests)}@{{}}}}",
                "\\toprule", "System & " + " & ".join(h for _, h, _ in tests) + " \\\\", "\\midrule"]
        (OUT / f"{name}.tex").write_text("\n".join(head + body + ["\\bottomrule", "\\end{tabular*}"]) + "\n",
                                         encoding="utf-8")
    (OUT / "analysis_axb_en.tex").write_text(axb_table("sick_en_ablation"), encoding="utf-8")
    (OUT / "analysis_confusion.tex").write_text(confusion_table([
        ("SICK (EN)", "sick_en_ablation", "sick_test"),
        ("FraCaS (EN)", "sick_en_ablation", "fracas_en"),
        ("InferBR (PT)", "inferbr_pt_ablation", "inferbr_test")]), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
