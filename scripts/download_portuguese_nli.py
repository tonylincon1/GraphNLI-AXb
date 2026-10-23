#!/usr/bin/env python3
"""Download Portuguese NLI mirrors and write the manifests used by the YAML pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "portuguese"


def label_name(value, dataset: str) -> str:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"entailment", "entails", "1"}:
            return "entailment"
        if value in {"contradiction", "contradicts", "0"} and dataset == "inferbr":
            return "contradiction"
        return "neutral"
    value = int(value)
    if dataset == "assin":
        return {0: "neutral", 1: "entailment", 2: "neutral"}[value]
    if dataset == "assin2":
        return {0: "neutral", 1: "entailment"}[value]
    return {0: "contradiction", 1: "entailment", 2: "neutral"}[value]


def write_rows(frame: pd.DataFrame, path: Path, dataset: str) -> None:
    columns = {str(column).lower(): column for column in frame.columns}
    premise = columns.get("premise") or columns.get("sentence1") or columns.get("sentence_a")
    hypothesis = columns.get("hypothesis") or columns.get("sentence2") or columns.get("sentence_b")
    label = next((columns[name] for name in ("label", "entailment_judgment", "entailment_label") if name in columns), None)
    if not premise or not hypothesis or not label:
        raise ValueError(f"Could not identify NLI columns in {list(frame.columns)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, row in frame.iterrows():
            record = {
                "id": int(index),
                "premise": str(row[premise]),
                "hypothesis": str(row[hypothesis]),
                "label": label_name(row[label], dataset),
                "language": "pt",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(frame)} rows)")


def download_table(repo: str, filename: str, split: str, dataset: str) -> pd.DataFrame:
    path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename)
    frame = pd.read_parquet(path) if filename.endswith(".parquet") else pd.read_csv(path)
    frame.attrs["dataset"] = dataset
    print(f"downloaded {repo}:{split} ({len(frame)} rows)")
    return frame


def main() -> None:
    # ASSIN combines its official Brazilian and European Portuguese partitions.
    for split in ("train", "validation", "test"):
        frames = [
            download_table("nilc-nlp/assin", f"ptbr/{split}-00000-of-00001.parquet", split, "assin"),
            download_table("nilc-nlp/assin", f"ptpt/{split}-00000-of-00001.parquet", split, "assin"),
        ]
        write_rows(pd.concat(frames, ignore_index=True), OUT / f"assin_{'dev' if split == 'validation' else split}.jsonl", "assin")

    for split in ("train", "validation", "test"):
        frame = download_table("nilc-nlp/assin2", f"data/{split}-00000-of-00001.parquet", split, "assin2")
        write_rows(frame, OUT / f"assin2_{'dev' if split == 'validation' else split}.jsonl", "assin2")

    for split in ("train", "val", "test"):
        frame = download_table("hapaxlegomenon/InferBR", f"{split}.csv", split, "inferbr")
        write_rows(frame, OUT / f"inferbr_{'dev' if split == 'val' else split}.jsonl", "inferbr")


if __name__ == "__main__":
    main()
