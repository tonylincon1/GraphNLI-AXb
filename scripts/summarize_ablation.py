#!/usr/bin/env python3
"""Paired ablation report for experiment_0009-style YAML experiments.

For every ablation pipeline (``*_ablation``) the script compares each variant
with the full model of that pipeline (the first variant listed in the YAML,
e.g. ``natlog_ot_v1``) and with the ``pair_cls`` control, pairing runs by seed:

* mean ± std per variant and test split;
* paired delta vs. the full model, 95% bootstrap CI over seeds, and a
  two-sided Wilcoxon signed-rank p-value (exact for small n);
* a merged ``summary.json`` per pipeline (ablations + control) and the
  Friedman/Nemenyi critical-difference report from ``critical_difference.py``.

    poetry run python scripts/summarize_ablation.py runs/experiment_0009
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphnli_axb.critical_difference import generate_critical_difference_reports  # noqa: E402

DEFAULT_METRICS = {
    "assin_test": ["binary/mcc"],
    "assin2_test": ["binary/mcc"],
    "inferbr_test": ["nli3/macro_f1", "binary/mcc"],
}


def load_results(pipeline_dir: Path) -> dict[str, dict[int, dict]]:
    found: dict[str, dict[int, dict]] = {}
    for path in pipeline_dir.glob("*/*/seed_*/result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        found.setdefault(payload["variant"], {})[int(payload["seed"])] = payload
    return found


def bootstrap_ci(values: np.ndarray, rounds: int = 10000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(rounds, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired(results, variant, reference, split, metric):
    seeds = sorted(set(results.get(variant, {})) & set(results.get(reference, {})))
    a = np.array([results[variant][s]["eval"][split][metric] for s in seeds])
    b = np.array([results[reference][s]["eval"][split][metric] for s in seeds])
    return seeds, a, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--control-pipeline", default="control")
    parser.add_argument("--control-variant", default="pair_cls")
    parser.add_argument("--cd-metric", default="binary/mcc")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    config = yaml.safe_load((experiment_dir / "config.yaml").read_text(encoding="utf-8"))
    control = load_results(experiment_dir / args.control_pipeline).get(args.control_variant, {})
    lines = [f"# Ablation report — {experiment_dir.name}", ""]
    if not control:
        lines.append(f"> Control `{args.control_variant}` not found in `{args.control_pipeline}`; deltas vs. control omitted.\n")

    for name, pipeline in config["pipelines"].items():
        if not name.endswith("_ablation") or not pipeline.get("enabled", True):
            continue
        pipeline_dir = experiment_dir / name
        results = load_results(pipeline_dir)
        if not results:
            lines.append(f"## {name}\n\n(no results yet)\n")
            continue
        variants = [v for v in pipeline["variants"] if v in results]
        full = pipeline["variants"][0]
        if control:
            results[args.control_variant] = control
            variants.append(args.control_variant)
        lines += [f"## {name}", "", f"Full model: `{full}`. Δ = variant − full model, paired by seed "
                  "(95% bootstrap CI; Wilcoxon signed-rank p). Negative Δ means the removed component helped.", ""]
        present = {split for runs in results.values() for r in runs.values() for split in r["eval"]}
        for split, metrics in DEFAULT_METRICS.items():
            if split not in present:
                continue
            for metric in metrics:
                lines += [f"### {split} · {metric}", "",
                          "| variant | n | mean ± std | Δ vs full | 95% CI | p | Δ vs pair_cls |",
                          "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
                for variant in variants:
                    values = [r["eval"][split][metric] for r in results[variant].values() if split in r["eval"]]
                    if not values:
                        continue
                    row = f"| `{variant}` | {len(values)} | {np.mean(values):.4f} ± {np.std(values):.4f} |"
                    if variant == full:
                        row += " — | — | — |"
                    else:
                        _, a, b = paired(results, variant, full, split, metric)
                        delta = a - b
                        low, high = bootstrap_ci(delta)
                        p = wilcoxon(a, b).pvalue if len(delta) >= 2 and np.any(delta != 0) else float("nan")
                        row += f" {delta.mean():+.4f} | [{low:+.4f}, {high:+.4f}] | {p:.3f} |"
                    if control and variant != args.control_variant:
                        _, a, b = paired(results, variant, args.control_variant, split, metric)
                        row += f" {(a - b).mean():+.4f} |" if len(a) else " — |"
                    else:
                        row += " — |"
                    lines.append(row)
                lines.append("")
        merged = {"comparison": name, "results": [r for v in variants for r in results[v].values()]}
        merged_path = pipeline_dir / "ablation_summary.json"
        merged_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        try:
            generate_critical_difference_reports(merged_path, pipeline_dir / "critical_difference", metric=args.cd_metric)
            lines.append(f"Critical-difference report: `{(pipeline_dir / 'critical_difference').relative_to(experiment_dir)}`\n")
        except Exception as error:  # small n or incomplete seeds should not hide the table
            lines.append(f"Critical-difference report skipped: {error}\n")

    report = experiment_dir / "ablation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
