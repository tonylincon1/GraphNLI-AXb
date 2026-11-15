#!/usr/bin/env python3
"""Explanation figures from experiment_0013 (COVER decisions on pairs the [CLS] baseline gets wrong).

For each pipeline and each COVER variant (full model, no conflict) the script picks up to
``--per-model`` out-of-distribution test pairs that the variant classifies correctly and pair_cls
(same seed) classifies wrongly, on the binary decision entailment vs. non-entailment used in the
paper. One TikZ standalone figure is written per pair, in the style of
paper/figures/interpretability_example.tex, and compiled to PDF and PNG:

  premise / hypothesis words (content words, i.e. graph nodes, framed; hypothesis nodes shaded)
  top premise alignment of every hypothesis node (last graph layer; width ~ attention, label =
      lexical relation and weight; green = supports, red = conflicts, blue dashed = no lexicon
      relation, the alignment was learned)
  support s_j / conflict c_j bars and weight w_j under each hypothesis node
  the logical aggregation (log P_cov, log P_noconf), the structured, [CLS] and fused
      probabilities, and the prediction of the separately trained pair_cls model

Selection favours short pairs (readable figures), confident correct COVER decisions and confident
wrong pair_cls decisions, and at most ``--max-per-test`` pairs from the same test set.

    poetry run python scripts/make_explanation_figures.py                # experiment_0013
    poetry run python scripts/make_explanation_figures.py --no-compile   # .tex only
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"coverage_pl_v1": "COVER", "coverage_pl_v1_no_conflict": "COVER $-$conflict"}
OOD_TESTS = {
    "sick_en_explain": ["axb", "conjnli_dev", "fracas_en", "breaking_nli"],
    "inferbr_pt_explain": ["assin_test", "assin2_test", "sick_br_test"],
}
TEST_NAMES = {
    "axb": "AX-b", "conjnli_dev": "ConjNLI", "fracas_en": "FraCaS", "breaking_nli": "Breaking NLI",
    "assin_test": "ASSIN", "assin2_test": "ASSIN 2", "sick_br_test": "SICK-BR", "sick_test": "SICK",
    "inferbr_test": "InferBR",
}
LABELS = ["entailment", "neutral", "contradiction"]
BINARY = ["entailment", "non-entailment"]
LEXICAL = {"identical", "same_stem", "synonym", "antonym", "hypernym", "hyponym", "number_equal",
           "number_conflict"}
CONFLICT_RELATIONS = {"antonym", "number_conflict"}
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)  # same split as the encoder's pre-tokenizer (hyphens apart)


# ---------------------------------------------------------------- data

def read_predictions(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return {row["uid"]: row for row in csv.DictReader(handle)}


def read_explanations(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return {rec["uid"]: rec for rec in map(json.loads, handle)}


def run_dir(pipeline_dir: Path, variant: str, seed: int) -> Path:
    hits = sorted(pipeline_dir.glob(f"*/{variant}/seed_{seed}"))
    if not hits:
        raise FileNotFoundError(f"no run for {variant} seed {seed} under {pipeline_dir}")
    return hits[0]


def candidates(pipeline_dir: Path, variant: str, seed: int, tests: list[str], max_words: int) -> list[dict]:
    cover_dir, cls_dir = run_dir(pipeline_dir, variant, seed), run_dir(pipeline_dir, "pair_cls", seed)
    out = []
    for test in tests:
        expl_path = cover_dir / "cover_explanations" / f"{test}.jsonl"
        if not expl_path.exists():
            continue
        cover_pred = read_predictions(cover_dir / f"predictions_{test}.csv")
        cls_pred = read_predictions(cls_dir / f"predictions_{test}.csv")
        expl = read_explanations(expl_path)
        for uid, rec in expl.items():
            c, b = cover_pred.get(uid), cls_pred.get(uid)
            if c is None or b is None:
                continue
            gold = int(c["label_binary"])
            if int(c["prediction_binary"]) != gold or int(b["prediction_binary"]) == gold:
                continue
            n_p, n_h = len(WORD_RE.findall(rec["premise"])), len(WORD_RE.findall(rec["hypothesis"]))
            if n_p > max_words or n_h > max_words or not rec["hypothesis_words"]:
                continue
            p_ent = rec["probs"][0]
            confidence = p_ent if gold == 0 else 1.0 - p_ent      # COVER confidence in the gold side
            rec = dict(rec, test=test, gold_binary=gold, cls_prediction_3=int(b["prediction_3"]),
                       cls_prediction_binary=int(b["prediction_binary"]), cover_confidence=confidence)
            rec["score"] = confidence - 0.02 * max(0, n_p + n_h - 18)
            out.append(rec)
    return out


def select(cands: list[dict], k: int, max_per_test: int) -> list[dict]:
    chosen, per_test, seen_h = [], {}, set()
    for rec in sorted(cands, key=lambda r: -r["score"]):
        if per_test.get(rec["test"], 0) >= max_per_test or rec["hypothesis"] in seen_h:
            continue
        chosen.append(rec)
        per_test[rec["test"]] = per_test.get(rec["test"], 0) + 1
        seen_h.add(rec["hypothesis"])
        if len(chosen) == k:
            break
    return chosen


# ---------------------------------------------------------------- TikZ

def tex(text: str) -> str:
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(repl.get(ch, ch) for ch in text)


def num(x: float) -> str:
    """.93 style (no leading zero), as in the paper figure."""
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0") else ("1.0" if s == "1.00" else s)


def layout(sentence: str, nodes: list[dict]) -> list[tuple[str, int | None]]:
    """Split a sentence into words and attach graph-node positions to the matching words."""
    words = WORD_RE.findall(sentence)
    out, i = [], 0
    pending = list(nodes)
    for word in words:
        if not re.search(r"\w", word):      # punctuation is not drawn
            continue
        pos = None
        if pending and (pending[0]["word"] == word or pending[0]["word"].startswith(word)
                        or word.startswith(pending[0]["word"])):
            pos = pending.pop(0)["pos"]
        out.append((word, pos))
        i += 1
    return out


def x_positions(words: list[tuple[str, int | None]], gap: float = 0.12) -> list[float]:
    xs, x = [], 0.0
    for word, _ in words:
        width = 0.105 * len(word) + 0.22
        xs.append(x + width / 2)
        x += width + gap
    return xs


def figure_tex(rec: dict, model_label: str, use_conflict: bool, title: str) -> str:
    hyp_nodes = {h["pos"]: h for h in rec["hypothesis_words"]}
    prem_words = layout(rec["premise"], rec["premise_words"])
    hyp_words = layout(rec["hypothesis"], rec["hypothesis_words"])
    px, hx = x_positions(prem_words), x_positions(hyp_words)
    # hypothesis words need room for their bars
    for i in range(1, len(hx)):
        if hyp_words[i][1] is not None:
            prev = max((hx[j] for j in range(i) if hyp_words[j][1] is not None), default=-9)
            if hx[i] - prev < 0.62:
                shift = 0.62 - (hx[i] - prev)
                for j in range(i, len(hx)):
                    hx[j] += shift
    width = max(px[-1] if px else 0, hx[-1] if hx else 0) + 0.6
    prem_x = {pos: x for (w, pos), x in zip(prem_words, px) if pos is not None}

    lines = []
    lines.append(r"\node[font=\scriptsize\itshape, anchor=east] at (-0.35,2.45) {$P$};")
    lines.append(r"\node[font=\scriptsize\itshape, anchor=east] at (-0.35,1.15) {$H$};")
    for i, ((word, pos), x) in enumerate(zip(prem_words, px)):
        style = "tok" if pos is not None else "stok"
        name = f"p{pos}" if pos is not None else f"pw{i}"
        lines.append(rf"\node[{style}] ({name}) at ({x:.2f},2.45) {{{tex(word)}}};")
    for i, ((word, pos), x) in enumerate(zip(hyp_words, hx)):
        style = "htok" if pos is not None else "stok"
        name = f"h{pos}" if pos is not None else f"hw{i}"
        lines.append(rf"\node[{style}] ({name}) at ({x:.2f},1.15) {{{tex(word)}}};")

    # alignment edges: top premise word of every hypothesis node
    labels: list[str] = []
    # labels sit above their hypothesis node, staggered over two rows so crossing edges stay legible
    label_row = 0
    for (word, pos), x in zip(hyp_words, hx):
        if pos is None:
            continue
        h = hyp_nodes[pos]
        top = (h.get("align") or [None])[0]
        if not top or top["attn"] < 0.1 or top["pos"] not in prem_x:
            continue
        rel = top["relation"]
        line = 0.4 + 1.6 * min(top["attn"], 1.0)
        conflicting = use_conflict and (h["conflict"] >= 0.5 or rel in CONFLICT_RELATIONS)
        style = "cedge" if conflicting else ("aedge" if rel in LEXICAL else "ledge")
        label = (rel.replace("_", " ") if rel in LEXICAL else "learned") + r"\," + num(top["attn"])
        lines.append(rf"\draw[{style}={line:.2f}pt] (h{pos}.north) -- (p{top['pos']}.south);")
        labels.append(rf"\node[rlab, anchor=south] at ({x:.2f},{1.47 + 0.36 * (label_row % 2):.2f}) {{{label}}};")
        label_row += 1

    lines.extend(labels)  # drawn after all edges so their white fill masks crossings

    # support / conflict bars
    lines.append(rf"\draw[black!40] (-0.1,0.05) -- ({width - 0.3:.2f},0.05);")
    # labels sit above their hypothesis node, staggered over two rows so crossing edges stay legible
    label_row = 0
    for (word, pos), x in zip(hyp_words, hx):
        if pos is None:
            continue
        h = hyp_nodes[pos]
        if use_conflict:
            lines.append(rf"\bars{{{x:.2f}}}{{{num(h['support'])}}}{{{num(h['conflict'])}}}{{{h['weight']:.2f}}}")
        else:
            lines.append(rf"\sbar{{{x:.2f}}}{{{num(h['support'])}}}{{{h['weight']:.2f}}}")

    p_cov = 2.718281828 ** rec["log_p_cov"]
    p_nc = 2.718281828 ** rec["log_p_noconf"]
    probs, cls_probs, struct = rec["probs"], rec["cls_probs"], rec["struct_probs"]
    pred3 = LABELS[max(range(3), key=lambda i: probs[i])]
    gold = rec["gold_3"] or BINARY[rec["gold_binary"]]
    triple = lambda p: r"(" + r",\ ".join(num(v) for v in p) + r")"
    agg = [rf"$\log\Pcov=\sum_j\omega_j\log s_j={rec['log_p_cov']:.2f}$\quad($\Pcov={num(p_cov)}$)"]
    if use_conflict:
        agg.append(rf"$\log\Pnc=\sum_j\omega_j\log(1-c_j)={rec['log_p_noconf']:.2f}$\quad($\Pnc={num(p_nc)}$)")
    else:
        agg.append(r"conflict head removed: $z_\E=\log\Pcov$, \N{} and \C{} share $\log(1-\Pcov)$")
    agg.append(rf"structured $p(\E,\N,\C)={triple(struct)}$;\ \ \textsc{{[cls]}} term ${triple(cls_probs)}$")
    agg.append(rf"fused $p(\E,\N,\C)={triple(probs)}$ $\Rightarrow$ \textbf{{{pred3}}}"
               rf"\quad gold: {tex(gold)}")
    cls_side = BINARY[rec["cls_prediction_binary"]]
    agg.append(rf"\textcolor{{relneg}}{{\textsc{{[cls]}} baseline model: {LABELS[rec['cls_prediction_3']]}"
               rf" ({cls_side}) \texttimes}}")
    box_width = max(width - 0.1, 8.6)
    lines.append(
        r"\node[draw=logic, fill=logicbg, rounded corners=3pt, line width=0.8pt, font=\scriptsize, "
        rf"align=left, anchor=north west, text width={box_width:.2f}cm, inner sep=4pt] (agg) at (-0.3,-0.85) {{%"
    )
    lines.append("  " + r"\\[1pt]".join(agg) + "};")
    lines.append(rf"\node[font=\scriptsize\bfseries, anchor=south west] at (-0.75,2.85) {{{title}}};")

    return TEMPLATE.replace("%BODY%", "\n".join(lines)).replace("%MODEL%", model_label)


TEMPLATE = r"""% Generated by scripts/make_explanation_figures.py from experiment_0013 (do not edit by hand).
% Model: %MODEL%
\documentclass[tikz,border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usetikzlibrary{positioning,arrows.meta,calc}

\definecolor{neural}{HTML}{2F6DB5}
\definecolor{neuralbg}{HTML}{E8F0FA}
\definecolor{logic}{HTML}{2E8B57}
\definecolor{logicbg}{HTML}{E6F4EC}
\definecolor{relneg}{HTML}{C0392B}

\tikzset{
  tok/.style={draw=neural!80, fill=white, rounded corners=1.5pt, inner sep=1.6pt, font=\scriptsize,
              minimum height=4mm, anchor=base},
  htok/.style={tok, fill=neural!12},
  stok/.style={tok, draw=black!25, text=black!45},
  aedge/.style={line width=#1, draw=logic},
  cedge/.style={line width=#1, draw=relneg},
  ledge/.style={line width=#1, draw=neural, densely dashed},
  rlab/.style={font=\tiny, fill=white, inner sep=0.6pt},
}
\newcommand{\bars}[4]{%
  \fill[logic]  (#1-0.17,0.05) rectangle ++(0.15,{0.8*#2});
  \fill[relneg] (#1+0.02,0.05) rectangle ++(0.15,{0.8*#3});
  \node[font=\tiny, anchor=north] at (#1-0.095,0.03) {$s$};
  \node[font=\tiny, anchor=north] at (#1+0.095,0.03) {$c$};
  \node[font=\tiny, anchor=north] at (#1,-0.22) {$#2\,/\,#3$};
  \node[font=\tiny, anchor=north, text=black!60] at (#1,-0.47) {$\omega{=}#4$};
}
\newcommand{\sbar}[3]{%
  \fill[logic]  (#1-0.075,0.05) rectangle ++(0.15,{0.8*#2});
  \node[font=\tiny, anchor=north] at (#1,0.03) {$s$};
  \node[font=\tiny, anchor=north] at (#1,-0.22) {$#2$};
  \node[font=\tiny, anchor=north, text=black!60] at (#1,-0.47) {$\omega{=}#3$};
}
\providecommand{\Pcov}{P_{\mathrm{cov}}}
\providecommand{\Pnc}{P_{\neg\mathrm{conf}}}
\providecommand{\E}{\textsc{e}}
\providecommand{\N}{\textsc{n}}
\providecommand{\C}{\textsc{c}}

\begin{document}
\begin{tikzpicture}
%BODY%
\end{tikzpicture}
\end{document}
"""


def compile_figure(path: Path) -> None:
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", path.name], cwd=path.parent,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["latexmk", "-c", path.name], cwd=path.parent, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if shutil.which("gs"):
        subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r220",
                        f"-sOutputFile={path.with_suffix('.png').name}", path.with_suffix(".pdf").name],
                       cwd=path.parent, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", default=str(ROOT / "runs" / "experiment_0013"))
    parser.add_argument("--pipelines", default=",".join(OOD_TESTS))
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--per-model", type=int, default=5)
    parser.add_argument("--max-per-test", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=16, help="longest premise/hypothesis (words)")
    parser.add_argument("--tests", default="", help="override the out-of-distribution test list")
    parser.add_argument("--out", default=str(ROOT / "paper" / "figures" / "explanations"))
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for pipeline in [p for p in args.pipelines.split(",") if p]:
        pipeline_dir = Path(args.experiment_dir) / pipeline
        if not pipeline_dir.exists():
            print(f"skip {pipeline}: not run yet")
            continue
        tests = args.tests.split(",") if args.tests else OOD_TESTS.get(pipeline, [])
        setting = "en" if pipeline.startswith("sick") else "pt"
        for variant, label in VARIANTS.items():
            try:
                cands = candidates(pipeline_dir, variant, args.seed, tests, args.max_words)
            except FileNotFoundError as err:
                print(f"skip {pipeline}/{variant}: {err}")
                continue
            chosen = select(cands, args.per_model, args.max_per_test)
            print(f"{pipeline}/{variant}: {len(cands)} pairs right where [CLS] is wrong; {len(chosen)} drawn")
            short = "cover" if variant == "coverage_pl_v1" else "cover_no_conflict"
            for k, rec in enumerate(chosen, 1):
                name = f"{setting}_{short}_{k}"
                title = f"{label} --- {TEST_NAMES.get(rec['test'], rec['test'])}"
                path = out_dir / f"{name}.tex"
                path.write_text(figure_tex(rec, label, variant != "coverage_pl_v1_no_conflict", title),
                                encoding="utf-8")
                if not args.no_compile:
                    compile_figure(path)
                index.append({
                    "figure": path.name, "pipeline": pipeline, "variant": variant, "seed": args.seed,
                    "test": rec["test"], "uid": rec["uid"], "premise": rec["premise"],
                    "hypothesis": rec["hypothesis"], "gold": rec["gold_3"] or BINARY[rec["gold_binary"]],
                    "cover_prediction": LABELS[max(range(3), key=lambda i: rec["probs"][i])],
                    "cls_prediction": LABELS[rec["cls_prediction_3"]],
                    "cover_confidence": round(rec["cover_confidence"], 3),
                })
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(index)} figures to {out_dir}")


if __name__ == "__main__":
    main()
