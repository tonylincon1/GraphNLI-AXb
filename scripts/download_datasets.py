#!/usr/bin/env python3
"""Download the initial NLI/diagnostic datasets for GraphNLI-AXb."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    filename: str
    description: str
    source: str
    split_note: str


DATASETS = [
    DatasetSpec(
        name="superglue_axb",
        url="https://dl.fbaipublicfiles.com/glue/superglue/data/v2/AX-b.zip",
        filename="AX-b.zip",
        description="SuperGLUE Broadcoverage Diagnostics, binary entailment.",
        source="https://super.gluebenchmark.com/diagnostics",
        split_note="Official test-only diagnostic split; 1104 examples.",
    ),
    DatasetSpec(
        name="glue_ax",
        url="https://dl.fbaipublicfiles.com/glue/data/AX.tsv",
        filename="AX.tsv",
        description="Original GLUE diagnostic set with fine-grained linguistic tags.",
        source="https://gluebenchmark.com/diagnostics",
        split_note="Official diagnostic/test split; 1104 sentence pairs.",
    ),
    DatasetSpec(
        name="xnli",
        url="https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip",
        filename="XNLI-1.0.zip",
        description="XNLI 1.0 multilingual NLI evaluation corpus.",
        source="https://github.com/facebookresearch/XNLI",
        split_note="Official dev/test splits in 15 languages; English training comes from MultiNLI.",
    ),
]

OPTIONAL_DATASETS = [
    DatasetSpec(
        name="xnli_mt",
        url="https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip",
        filename="XNLI-MT-1.0.zip",
        description="Machine-translated XNLI train/test data for translate-train/translate-test baselines.",
        source="https://github.com/facebookresearch/XNLI",
        split_note="Optional large archive released by XNLI; useful for translate-train comparisons.",
    ),
    DatasetSpec(
        name="mnli",
        url="https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
        filename="MNLI.zip",
        description="MultiNLI, the standard English training source for XNLI-style transfer.",
        source="https://cims.nyu.edu/~sbowman/multinli/",
        split_note="Official train, validation_matched, and validation_mismatched splits.",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"skip existing {destination}")
        return
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    print(f"download {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)


def extract_zip(path: Path, destination: Path) -> None:
    marker = destination / ".extracted"
    if marker.exists():
        print(f"skip extracted {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"extract {path.name}")
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)
    marker.write_text(path.name + "\n", encoding="utf-8")


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def count_tsv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)
    return max(0, len(rows) - 1)


def summarize_extracted(raw_dir: Path) -> dict:
    summary: dict[str, object] = {}

    axb = raw_dir / "superglue_axb" / "AX-b" / "AX-b.jsonl"
    if axb.exists():
        summary["superglue_axb"] = {"test": count_jsonl(axb), "path": str(axb)}

    glue_ax = raw_dir / "glue_ax" / "AX.tsv"
    if glue_ax.exists():
        summary["glue_ax"] = {"test": count_tsv_rows(glue_ax), "path": str(glue_ax)}

    xnli_dir = raw_dir / "xnli"
    xnli_files = sorted(xnli_dir.rglob("*.jsonl")) + sorted(xnli_dir.rglob("*.txt"))
    if xnli_files:
        summary["xnli_files"] = [str(path) for path in xnli_files]
        split_summary = {}
        for path in sorted(xnli_dir.rglob("xnli.*.jsonl")):
            split = path.stem.split(".")[-1]
            languages: dict[str, int] = {}
            labels: dict[str, int] = {}
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    language = row.get("language", "unknown")
                    label = row.get("gold_label", "unknown")
                    languages[language] = languages.get(language, 0) + 1
                    labels[label] = labels.get(label, 0) + 1
            split_summary[split] = {
                "total": sum(languages.values()),
                "languages": dict(sorted(languages.items())),
                "labels": dict(sorted(labels.items())),
            }
        summary["xnli"] = split_summary

    mnli_dir = raw_dir / "mnli" / "MNLI" / "original"
    mnli_splits = {
        "train": mnli_dir / "multinli_1.0_train.jsonl",
        "dev_matched": mnli_dir / "multinli_1.0_dev_matched.jsonl",
        "dev_mismatched": mnli_dir / "multinli_1.0_dev_mismatched.jsonl",
    }
    mnli_summary = {name: count_jsonl(path) for name, path in mnli_splits.items() if path.exists()}
    if mnli_summary:
        summary["mnli"] = mnli_summary

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory, relative to project root.")
    parser.add_argument("--with-mnli", action="store_true", help="Also download MultiNLI, required by the reference-compatible training protocol.")
    parser.add_argument("--with-xnli-mt", action="store_true", help="Also download the optional XNLI-MT archive.")
    parser.add_argument("--with-optional", action="store_true", help="Also download both MNLI and XNLI-MT archives.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = (project_root / args.data_dir).resolve()
    optional_names = set()
    if args.with_optional or args.with_mnli:
        optional_names.add("mnli")
    if args.with_optional or args.with_xnli_mt:
        optional_names.add("xnli_mt")
    specs = DATASETS + [spec for spec in OPTIONAL_DATASETS if spec.name in optional_names]

    manifest = {"datasets": [], "summary": {}}
    for spec in specs:
        dataset_dir = raw_dir / spec.name
        archive_path = dataset_dir / spec.filename
        download(spec.url, archive_path)
        if archive_path.suffix == ".zip":
            extract_zip(archive_path, dataset_dir)

        manifest["datasets"].append(
            {
                "name": spec.name,
                "description": spec.description,
                "url": spec.url,
                "source": spec.source,
                "split_note": spec.split_note,
                "archive": str(archive_path),
                "sha256": sha256(archive_path),
                "bytes": archive_path.stat().st_size,
            }
        )

    manifest["summary"] = summarize_extracted(raw_dir)
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
