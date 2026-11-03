#!/usr/bin/env python3
"""Build a comparable Portuguese five-model summary from completed runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCES = {
    "pair_cls": [
        "runs/experiment_0001/portuguese/neuralmind__bert-base-portuguese-cased/pair_cls",
        "runs/experiment_0002/portuguese/neuralmind__bert-base-portuguese-cased/pair_cls",
    ],
    "logicpath_v2": [
        "runs/experiment_0001/portuguese/neuralmind__bert-base-portuguese-cased/logicpath_v2",
        "runs/experiment_0004/portuguese/neuralmind__bert-base-portuguese-cased/logicpath_v2",
    ],
    "kbert_injection": [
        "runs/experiment_0003/portuguese/neuralmind__bert-base-portuguese-cased/kbert_injection",
    ],
    "kganet_rel_gat": [
        "runs/experiment_0003/portuguese/neuralmind__bert-base-portuguese-cased/kganet_rel_gat",
    ],
    "kgnli_dual": [
        "runs/experiment_0003/portuguese/neuralmind__bert-base-portuguese-cased/kgnli_dual",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/experiment_0003/portuguese/five_model_comparison/summary.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results = []
    missing = []
    for variant, source_dirs in SOURCES.items():
        found = {}
        for source in source_dirs:
            for result_path in (root / source).glob("seed_*/result.json"):
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                found[int(payload["seed"])] = payload
        for seed in range(1001, 1011):
            if seed not in found:
                missing.append(f"{variant}/seed_{seed}")
            else:
                results.append(found[seed])

    if missing:
        raise SystemExit("Missing results: " + ", ".join(missing))

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison": "Portuguese five-model comparison",
        "protocol": {
            "seeds": list(range(1001, 1011)),
            "datasets": ["assin_test", "assin2_test", "inferbr_test"],
            "metric_family": "binary",
            "sources": SOURCES,
            "note": "Pair classifier and LogicPath v2 are merged from prior compatible Portuguese runs; the other three are from experiment_0003.",
        },
        "results": sorted(results, key=lambda row: (row["variant"], int(row["seed"]))),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
