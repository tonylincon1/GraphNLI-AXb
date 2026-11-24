#!/usr/bin/env python3
"""Build the LaTeX result tables of the paper.

    python scripts/make_paper_tables.py            # writes paper/tables/*.tex

Sources: English = runs/experiment_0011 (RoBERTa-large); Portuguese = runs/experiment_0014
(Albertina-900M); cross-lingual = runs/experiment_0014 (XLM-R-large).
Systems with fewer than 10 finished seeds are reported on the seeds available, and their label
carries the seed count (n=k); systems with no finished seed are left blank. Rerun after every pull.

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
EXP = ROOT / "runs"  # pipelines below are "<experiment>/<pipeline>"
OUT = ROOT / "paper" / "tables"
N_SEEDS = 10
MIN_SEEDS = 1  # partial seed sets are reported, with their n in the row label
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
    "en": [(None, ["experiment_0011/sick_en_ablation", "experiment_0011/sick_en_graph_baselines",
                   "experiment_0011/sick_en_kbert"], EN_TESTS)],
    "pt": [(None, ["experiment_0014/inferbr_pt_large_ablation", "experiment_0014/inferbr_pt_large_graph_baselines"],
            PT_TESTS)],
    "xl": [("Train SICK (EN)", ["experiment_0014/xling_large_en_to_pt_ablation",
                                "experiment_0014/xling_large_en_to_pt_graph_baselines"], EN_TESTS + PT_TESTS),
           ("Train InferBR (PT)", ["experiment_0014/xling_large_pt_to_en_ablation",
                                   "experiment_0014/xling_large_pt_to_en_graph_baselines"], EN_TESTS + PT_TESTS)],
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
        if len(done) < MIN_SEEDS:
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


def with_n(label: str, data: dict | None) -> str:
    """Row label, with the seed count when a system has fewer than N_SEEDS finished seeds."""
    if data and len(data) < N_SEEDS:
        return f"{label}\\textsuperscript{{\\,$n{{=}}{len(data)}$}}"
    return label


def cell(values: list[float], bold: bool, mark: str) -> str:
    mean, sd = 100 * np.mean(values), 100 * np.std(values, ddof=1)
    text = f"{mean:.1f}"
    if bold:
        text = f"\\best{{{text}}}"
    return f"{text}{mark}\\,{{\\tiny$\\pm${sd:.1f}}}"


def delta_cell(variant: dict, full: dict, key: str, mark: str) -> str:
    """Mean paired difference to the full model (x100); negative values in red."""
    seeds = sorted(set(variant) & set(full))
    delta = 100 * np.mean([variant[s][key] - full[s][key] for s in seeds])
    if round(delta, 1) < 0:
        return f"\\textcolor{{red}}{{$-${abs(delta):.1f}{mark}}}"
    if round(delta, 1) == 0:
        return f"0.0{mark}"
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
            lines.append(f"{with_n(label, scores(pipelines, v, tests))} & " + " & ".join(row) + " \\\\")
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
        lines.append(f"{with_n(label, data[v])} & " + " & ".join(cells) + " \\\\")
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
    if len(done) < MIN_SEEDS:
        return None
    out = []
    for d in done:
        with (d / f"predictions_{test}.csv").open(encoding="utf-8") as handle:
            out.append(list(csv.DictReader(handle)))
    return out


AXB_TOP_GAINS, AXB_TOP_LOSSES = 8, 5
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

    # Compact view: the four coarse categories, then the fine-grained phenomena with the largest
    # gains and losses of COVER over [CLS] (AXB_TOP_GAINS / AXB_TOP_LOSSES of them).
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{\\textit{Categories}} \\\\")
    for key, label in AXB_COARSE:
        lines.append(row(label, lambda c, k=key: k in c))
    if complete:
        fine = []
        for key, _ in AXB_COARSE:
            for f in sorted({f for c in cats.values() if key in c for f in c[key].split(";")}):
                member = lambda c, k=key, f=f: f in c.get(k, "").split(";")
                deltas = []
                for v, _ in ANALYSIS_SYSTEMS:
                    deltas.append(np.mean([binary_mcc([r for r in rows if member(cats.get(r["uid"].split("-", 1)[1], {}))])
                                           for rows in preds[v]]))
                fine.append((deltas[1] - deltas[0], f, member))
        fine.sort(key=lambda t: -t[0])
        for title, part in (("Largest gains", fine[:AXB_TOP_GAINS]), ("Largest losses", fine[::-1][:AXB_TOP_LOSSES])):
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
            for _, f, member in part:
                lines.append(row(f, member))
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


# ----------------------------------------------------------------------------- combined tables
# One table* with the five English and the four Portuguese test sets side by side. Each half of a row
# comes from the model trained for that half: monolingual rows read the English half from the English
# setting and the Portuguese half from the Portuguese one; cross-lingual rows read the English half from
# the model trained on InferBR and the Portuguese half from the model trained on SICK.
E11, E14 = "experiment_0011", "experiment_0014"
HALVES = {
    "mono": (SETTINGS["en"][0][1], SETTINGS["pt"][0][1]),
    "xl": (SETTINGS["xl"][1][1], SETTINGS["xl"][0][1]),
}
TESTS = EN_TESTS + PT_TESTS
PARTIAL = "$^{\\S}$"  # fewer than N_SEEDS seeds finished


def half_scores(src, tests):
    """src: (pipelines, variant) or None."""
    return scores(*src, tests) if src else None


def row_data(row):
    """row: (label, en_src, pt_src) -> {test key: {seed: value}}."""
    _, en, pt = row
    out = {}
    for src, tests in ((en, EN_TESTS), (pt, PT_TESTS)):
        d = half_scores(src, tests)
        for key, _, _ in tests:
            if d and all(v.get(key) is not None for v in d.values()):
                out[key] = {seed: {key: v[key]} for seed, v in d.items()}
    return out


def header() -> list[str]:
    n_en, n_pt = len(EN_TESTS), len(PT_TESTS)
    return [f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}l{'c' * (n_en + n_pt)}@{{}}}}",
            "\\toprule",
            f" & \\multicolumn{{{n_en}}}{{c}}{{English tests}} & \\multicolumn{{{n_pt}}}{{c}}{{Portuguese tests}} \\\\",
            f"\\cmidrule(lr){{2-{n_en + 1}}}\\cmidrule(l){{{n_en + 2}-{n_en + n_pt + 1}}}",
            "System & " + " & ".join(h for _, h, _ in TESTS) + " \\\\"]


def group_title(text: str) -> str:
    return f"\\multicolumn{{{len(TESTS) + 1}}}{{@{{}}l}}{{\\textit{{{text}}}}} \\\\"


def combined_main(groups) -> str:
    """groups: (title, rows); row = (label, en_src, pt_src); the first row of a group is \\cls{}."""
    lines = header()
    for title, rows in groups:
        lines += ["\\midrule", group_title(title)]
        data = [row_data(r) for r in rows]
        for i, (label, _, _) in enumerate(rows):
            cells = []
            for key, _, _ in TESTS:
                means = {j: np.mean([v[key] for v in d[key].values()]) for j, d in enumerate(data) if key in d}
                if key not in data[i]:
                    cells.append("")
                    continue
                d = data[i][key]
                mark = ""
                if i and key in data[0]:
                    p = paired_p(d, data[0][key], key)
                    mark = "$^{\\dagger}$" if p is not None and p < 0.05 else ""
                if len(d) < N_SEEDS:
                    mark += PARTIAL
                cells.append(cell([v[key] for v in d.values()], max(means, key=means.get) == i, mark))
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def combined_ablation(groups) -> str:
    """groups: (title, full_row, families, order).
    families: list of (rows, marker); marker "holm": Holm over the family (asterisk), "raw": uncorrected
    (dagger). order: display items ("row", family, index, indent) or ("rule",)."""
    lines = header()
    for title, full_row, families, order in groups:
        lines += ["\\midrule", group_title(title)]
        full = row_data(full_row)
        cells = []
        for key, _, _ in TESTS:
            if key not in full:
                cells.append("")
                continue
            mark = PARTIAL if len(full[key]) < N_SEEDS else ""
            cells.append(cell([v[key] for v in full[key].values()], False, mark))
        lines.append(f"{full_row[0]} & " + " & ".join(cells) + " \\\\")
        data, marks = [], []
        for rows, marker in families:
            fd = [row_data(r) for r in rows]
            fm = [{} for _ in rows]
            for key, _, _ in TESTS:
                raw = {j: paired_p(d[key], full[key], key) for j, d in enumerate(fd) if key in d and key in full}
                adj = holm(raw, len(rows)) if marker == "holm" else raw
                for j, p in adj.items():
                    if p is not None and p < 0.05:
                        fm[j][key] = "$^{*}$" if marker == "holm" else "$^{\\dagger}$"
            data.append(fd)
            marks.append(fm)
        for item in order:
            if item[0] == "rule":
                lines.append("\\cmidrule(l){1-%d}" % (len(TESTS) + 1))
                continue
            if item[0] == "sub":
                lines.append(f"\\multicolumn{{{len(TESTS) + 1}}}{{@{{}}l}}{{{item[1]}}} \\\\")
                continue
            _, f, j, indent, *override = item
            label = override[0] if override else families[f][0][j][0]
            row = []
            for key, _, _ in TESTS:
                d = data[f][j]
                if key not in d or key not in full:
                    row.append("")
                    continue
                mark = marks[f][j].get(key, "") + (PARTIAL if len(d[key]) < N_SEEDS else "")
                row.append(delta_cell(d[key], full[key], key, mark))
            lines.append(("\\quad " if indent else "") + f"{label} & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def rows_for(half, systems):
    en, pt = HALVES[half]
    return [(label, (en, v), (pt, v)) for v, label in systems]


LEXICON_SOURCES = [
    # Read as removals from the full model (WordNet or PT-LKB, + Wiktionary): removing Wiktionary leaves
    # WordNet / PT-LKB only, and vice versa. The ConceptNet variants of experiment 0014
    # (sick_en_lex_conceptnet/_all, inferbr_pt_large_lex_conceptnet/_all) are left out of the paper for now.
    ("Wiktionary", "sick_en_lex_wordnet", "inferbr_pt_large_lex_ptlkb"),
    ("WordNet / PT-LKB", "sick_en_lex_wiktionary", "inferbr_pt_large_lex_wiktionary"),
]


def write_combined() -> None:
    main_systems = [MAIN_SYSTEMS[0], *MAIN_SYSTEMS[1:]]  # [CLS] first: reference of the daggers
    (OUT / "main.tex").write_text(combined_main([
        ("Monolingual: RoBERTa-large on SICK (EN) / Albertina-900M on InferBR (PT)", rows_for("mono", main_systems)),
        ("Cross-lingual, XLM-R-large: trained on InferBR (EN tests) / on SICK (PT tests)", rows_for("xl", main_systems)),
    ]), encoding="utf-8")
    abl = [a for a in ABLATIONS if a[0] not in ("coverage_pl_v1", "pair_cls")]
    cls = [a for a in ABLATIONS if a[0] == "pair_cls"]
    lex = [(label, ([f"{E14}/{en}"], "coverage_pl_v1"), ([f"{E14}/{pt}"], "coverage_pl_v1"))
           for label, en, pt in LEXICON_SOURCES]
    # Display: ablations, with the lexicon-source variants right below "- lexicon"; Holm stays per family.
    lex_pos = [v for v, _ in abl].index("coverage_pl_v1_no_symbolic")
    mono_order = ([("row", 0, j, False) for j in range(lex_pos)]
                  + [("sub", "$-$ lexicon"),
                     ("row", 0, lex_pos, True, "all")]
                  + [("row", 1, j, True) for j in range(len(lex))]
                  + [("row", 0, j, False) for j in range(lex_pos + 1, len(abl))]
                  + [("rule",), ("row", 2, 0, False)])
    # Cross-lingual lexicon ablation: EN half from the model trained on InferBR, PT half from SICK.
    xl_lex = [(label, ([f"{E14}/xling_large_pt_to_en_lex_{name}"], "coverage_pl_v1"),
               ([f"{E14}/xling_large_en_to_pt_lex_{name}"], "coverage_pl_v1"))
              for label, name in (("Wiktionary", "no_wiktionary"), ("WordNet / PT-LKB", "no_wordnet_ptlkb"))]
    xl_order = ([("row", 0, j, False) for j in range(lex_pos)]
                + [("sub", "$-$ lexicon"), ("row", 0, lex_pos, True, "all")]
                + [("row", 2, j, True) for j in range(len(xl_lex))]
                + [("row", 0, j, False) for j in range(lex_pos + 1, len(abl))]
                + [("rule",), ("row", 1, 0, False)])
    (OUT / "ablation.tex").write_text(combined_ablation([
        ("Monolingual: RoBERTa-large (EN) / Albertina-900M (PT)", rows_for("mono", [ABLATIONS[0]])[0],
         [(rows_for("mono", abl), "holm"), (lex, "holm"), (rows_for("mono", cls), "raw")], mono_order),
        ("Cross-lingual, XLM-R-large", rows_for("xl", [ABLATIONS[0]])[0],
         [(rows_for("xl", abl), "holm"), (rows_for("xl", cls), "raw"), (xl_lex, "holm")], xl_order),
    ]), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_combined()
    (OUT / "analysis_axb_en.tex").write_text(axb_table("experiment_0011/sick_en_ablation"), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
