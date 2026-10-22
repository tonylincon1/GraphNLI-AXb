#!/usr/bin/env python3
"""Create a fixed, label-stratified MultiNLI training manifest."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from graphnli_axb.data import load_mnli, project_root


def select_stratified(rows, limit: int, seed: int):
    if limit <= 0 or limit > len(rows):
        raise ValueError(f"limit must be between 1 and {len(rows)}")
    by_label = {}
    for row in rows:
        by_label.setdefault(row.label_3, []).append(row)
    rng = random.Random(seed)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)

    labels = sorted(by_label)
    base, remainder = divmod(limit, len(labels))
    selected = []
    for index, label in enumerate(labels):
        quota = base + (1 if index < remainder else 0)
        selected.extend(by_label[label][:quota])
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/manifests/mnli_train_2000_stratified.jsonl")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Seed used only to construct the manifest.")
    args = parser.parse_args()

    root = project_root()
    data_root = root / args.data_dir
    rows = load_mnli(data_root, "train")
    selected = select_stratified(rows, args.limit, args.seed)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    counts = Counter(row.label_3 for row in selected)
    metadata = {
        "source": "MultiNLI train",
        "limit": len(selected),
        "selection_seed": args.seed,
        "label_3_counts": dict(sorted(counts.items())),
        "manifest": str(output),
    }
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {metadata_path}")
    print(f"label_3_counts={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
