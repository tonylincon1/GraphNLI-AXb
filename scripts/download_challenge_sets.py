#!/usr/bin/env python3
"""Download the NLI challenge/OOD test sets used by experiment_0011 and convert them to
the JSONL manifest format read by the pipeline (premise, hypothesis, label, language, logic).

Written to data/raw/challenge/:

  English
    breaking_nli.jsonl   Breaking NLI (Glockner et al., 2018), 8,193 pairs, 3-way.
                         Single-word substitutions (antonyms, co-hyponyms, hypernyms...).
                         logic = substitution category.
    hans.jsonl           HANS (McCoy et al., 2019), evaluation set, 30,000 pairs, binary.
                         logic = heuristic:subcase.
    conjnli_dev.jsonl    ConjNLI (Saha et al., 2020), dev set, 623 pairs, 3-way (the test
                         set is released without labels, so the dev set is the usual test).
    med.jsonl            MED (Yanaka et al., 2019), 5,382 pairs, binary (entailment/neutral).
                         logic = genre tag (source:phenomenon:upward_monotone/downward_monotone...).
    fracas_en.jsonl      FraCaS (Cooper et al., 1996), English, 3-way; problems with an
                         undefined answer are dropped; multiple premises are joined with a
                         space. logic = section|n_premises.
  Portuguese
    sick_br_test.jsonl   SICK-BR (Real et al., 2018), SemEval TEST partition, 4,906 pairs,
                         3-way. Translation of the SICK test set: id = SICK pair_ID, so every
                         pair is aligned with sick_test.
  Annotations
    axb_categories.json  For each AX-b idx, the four coarse GLUE diagnostic categories (Lexical
                         Semantics, Predicate-Argument Structure, Logic, Knowledge) and their
                         fine-grained labels, from the GLUE diagnostic-full.tsv (same pairs, same
                         order as SuperGLUE AX-b; checked on load).
  Provenance
    assin2_test_sick_provenance.json
                         For each ASSIN 2 test id, the SICK partition of its source pair in
                         SICK-BR (TRAIN / TRIAL / TEST / nomatch), by token Jaccard >= 0.6 on
                         both sentences. ~36% of ASSIN 2 test comes from SICK TRAIN/TRIAL, so
                         models trained on SICK (en_to_pt) must be scored on the clean subset.

Binary sets store non-entailment as "neutral" (as ASSIN/ASSIN 2): only binary/* metrics are
meaningful for hans.jsonl and med.jsonl.

    poetry run python scripts/download_challenge_sets.py
"""
from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "challenge"
RAW = OUT / "source"

URLS = {
    "breaking_nli.zip": "https://raw.githubusercontent.com/BIU-NLP/Breaking_NLI/master/breaking_nli_dataset.zip",
    "hans.txt": "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt",
    "conj_dev.tsv": "https://raw.githubusercontent.com/swarnaHub/ConjNLI/master/data/NLI/conj_dev.tsv",
    "med.tsv": "https://raw.githubusercontent.com/verypluming/MED/master/MED.tsv",
    "fracas_en_fr.tsv": "https://huggingface.co/datasets/maximoss/fracas/resolve/main/fracas_in_english_and_french.tsv",
    "glue_diagnostic_full.tsv": "https://dl.fbaipublicfiles.com/glue/data/diagnostic-full.tsv",
    "sick_br.txt": "https://raw.githubusercontent.com/livyreal/SICK-BR/master/SICK_BR.txt",
}


def fetch(name: str) -> bytes:
    path = RAW / name
    if not path.exists():
        print(f"downloading {URLS[name]}")
        with urllib.request.urlopen(URLS[name], timeout=120) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def tsv(data: bytes) -> list[dict]:
    text = data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))


def write(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} pairs, labels {dict(Counter(r['label'] for r in rows))}")


def row(idx, premise, hypothesis, label, language, logic) -> dict:
    label = {"non-entailment": "neutral"}.get(label.strip().lower(), label.strip().lower())
    assert label in {"entailment", "neutral", "contradiction"}, label
    return {"id": idx, "premise": " ".join(premise.split()), "hypothesis": " ".join(hypothesis.split()),
            "label": label, "language": language, "logic": logic}


def breaking_nli() -> None:
    with zipfile.ZipFile(io.BytesIO(fetch("breaking_nli.zip"))) as archive:
        name = next(n for n in archive.namelist() if n.endswith("dataset.jsonl"))
        lines = archive.read(name).decode("utf-8").splitlines()
    data = [json.loads(line) for line in lines if line.strip()]
    write("breaking_nli.jsonl", [row(d["pairID"], d["sentence1"], d["sentence2"], d["gold_label"], "en", d["category"]) for d in data])


def hans() -> None:
    data = tsv(fetch("hans.txt"))
    write("hans.jsonl", [row(d["pairID"], d["sentence1"], d["sentence2"], d["gold_label"], "en",
                             f"{d['heuristic']}:{d['subcase']}") for d in data])


def conjnli() -> None:
    data = tsv(fetch("conj_dev.tsv"))
    write("conjnli_dev.jsonl", [row(i, d["Premise"], d["Hypothesis"], d["Label"], "en", "conjnli") for i, d in enumerate(data)])


def med() -> None:
    data = tsv(fetch("med.tsv"))
    write("med.jsonl", [row(d["pairID"], d["sentence1"], d["sentence2"], d["gold_label"], "en", d["genre"]) for d in data])


def fracas() -> None:
    labels = {"0": "entailment", "1": "neutral", "2": "contradiction"}
    rows = []
    for d in tsv(fetch("fracas_en_fr.tsv")):
        if d["label"] not in labels:
            continue
        premises = [d[f"premise{k}_original"] for k in range(1, 6) if d.get(f"premise{k}_original")]
        rows.append(row(d["id"], " ".join(premises), d["hypothesis_original"], labels[d["label"]], "en",
                        f"{d['topic'].strip().lower()}|{len(premises)}p"))
    write("fracas_en.jsonl", rows)


def axb_categories() -> None:
    axb_path = ROOT / "data" / "raw" / "superglue_axb" / "AX-b" / "AX-b.jsonl"
    if not axb_path.exists():
        print(f"skipping AX-b categories: {axb_path} not found")
        return
    axb = [json.loads(line) for line in axb_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    diag = tsv(fetch("glue_diagnostic_full.tsv"))
    assert len(axb) == len(diag), (len(axb), len(diag))
    coarse = ["Lexical Semantics", "Predicate-Argument Structure", "Logic", "Knowledge"]
    out = {}
    for a, d in zip(axb, diag):
        assert a["sentence1"].strip() == d["Premise"].strip() and a["sentence2"].strip() == d["Hypothesis"].strip()
        out[str(a["idx"])] = {c: d[c].strip() for c in coarse if d[c].strip()}
    (OUT / "axb_categories.json").write_text(json.dumps(out, indent=0), encoding="utf-8")
    print(f"axb_categories.json: {dict(Counter(c for v in out.values() for c in v))}")


def sick_br() -> list[dict]:
    data = tsv(fetch("sick_br.txt"))
    test = [row(int(d["pair_ID"]), d["sentence_A"], d["sentence_B"], d["entailment_label"], "pt", "sick_br")
            for d in data if d["SemEval_set"] == "TEST"]
    write("sick_br_test.jsonl", test)
    return data


def assin2_provenance(sick_br_rows: list[dict]) -> None:
    path = ROOT / "data" / "raw" / "portuguese" / "assin2_test.jsonl"
    if not path.exists():
        print(f"skipping ASSIN 2 provenance: {path} not found")
        return
    tok = lambda s: frozenset(re.findall(r"\w+", s.lower()))
    jac = lambda a, b: len(a & b) / max(1, len(a | b))
    index = defaultdict(list)
    sides = [(tok(r["sentence_A"]), tok(r["sentence_B"])) for r in sick_br_rows]
    for i, (a, b) in enumerate(sides):
        for word in a | b:
            index[word].append(i)
    provenance = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        p, h = tok(x["premise"]), tok(x["hypothesis"])
        best, best_i = 0.0, None
        for i, _ in Counter(i for w in p | h for i in index[w]).most_common(50):
            a, b = sides[i]
            score = max(min(jac(p, a), jac(h, b)), min(jac(p, b), jac(h, a)))
            if score > best:
                best, best_i = score, i
        provenance[str(x["id"])] = sick_br_rows[best_i]["SemEval_set"] if best >= 0.6 else "nomatch"
    (OUT / "assin2_test_sick_provenance.json").write_text(json.dumps(provenance, indent=0), encoding="utf-8")
    print(f"assin2_test_sick_provenance.json: {dict(Counter(provenance.values()))}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    breaking_nli()
    hans()
    conjnli()
    med()
    fracas()
    axb_categories()
    assin2_provenance(sick_br())


if __name__ == "__main__":
    main()
