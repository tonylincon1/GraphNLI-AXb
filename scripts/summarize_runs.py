#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", help="Path to a run summary.json.")
    parser.add_argument("--out", default=None, help="Output CSV path.")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    flat = []
    for run in rows:
        base = {
            "variant": run["variant"],
            "seed": run["seed"],
            "model_name": run["model_name"],
            "best_epoch": run["best_epoch"],
            "selection_metric": run.get("selection_metric", "nli3/macro_f1"),
            "best_validation_metric": run["best_validation_metric"],
            "early_stopped": run.get("early_stopped", False),
            "stopped_epoch": run.get("stopped_epoch", 0),
            "elapsed_seconds": run["elapsed_seconds"],
        }
        for split, metrics in run["eval"].items():
            row = dict(base)
            row["split"] = split
            row.update(metrics)
            flat.append(row)

    out_path = Path(args.out) if args.out else summary_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in flat for key in row})
    preferred = [
        "variant",
        "seed",
        "split",
        "model_name",
        "best_epoch",
        "selection_metric",
        "best_validation_metric",
        "early_stopped",
        "stopped_epoch",
        "elapsed_seconds",
    ]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
