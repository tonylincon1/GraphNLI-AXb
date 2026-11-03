#!/usr/bin/env python3
"""Plot LogicPath ablation metrics and surrogate feature importance."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


LABELS = {
    "logicpath_ablation_transformer": "Transformer only",
    "logicpath_ablation_states": "States graph",
    "logicpath_ablation_transformations": "States + transformations",
    "logicpath_v2": "Full LogicPath v2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    candidates = list(root.glob("portuguese_inferbr_ablation/*/logicpath_v*/seed_*/result.json"))
    reference_result = max(
        candidates,
        key=lambda path: int(path.parent.name.removeprefix("seed_")),
        default=None,
    )
    if reference_result is None:
        raise SystemExit(
            f"No completed ablation results found under {root}. "
            "Run the experiment before plotting."
        )
    model_root = reference_result.parents[2]
    seed_dir = reference_result.parent.name

    results = {}
    for result_path in model_root.glob(f"*/{seed_dir}/result.json"):
        variant = result_path.parent.parent.name
        results[variant] = json.loads(result_path.read_text(encoding="utf-8"))
    if len(results) != 4:
        raise SystemExit(f"Expected four completed variants, found {sorted(results)}")

    test_names = ["inferbr_test", "assin_test", "assin2_test"]
    aggregate_metrics = {}
    for variant in results:
        labels, binary_predictions = [], []
        complete = True
        for test_name in test_names:
            prediction_path = model_root / variant / seed_dir / f"predictions_{test_name}.csv"
            if not prediction_path.exists():
                complete = False
                break
            with prediction_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            labels.extend(int(row["label_binary"]) for row in rows)
            binary_predictions.extend(int(row["prediction_binary"]) for row in rows)
        if complete and labels:
            aggregate_metrics[variant] = {
                "binary/mcc": float(matthews_corrcoef(labels, binary_predictions)),
                "binary/accuracy": float(accuracy_score(labels, binary_predictions)),
                "binary/macro_f1": float(f1_score(labels, binary_predictions, average="macro", zero_division=0)),
            }
    aggregate = len(aggregate_metrics) == len(results)

    surrogate_path = model_root / f"logicpath_v2/{seed_dir}/surrogate/surrogate_analysis.json"
    surrogate = json.loads(surrogate_path.read_text(encoding="utf-8"))
    groups = surrogate["groups"]
    group_labels = {
        "transformer_pair": "Transformer pair (h_pair)",
        "graph_pool": "Graph pooling (h_graph)",
        "pair_graph_abs_difference": "|Pair - graph| (|h_pair - h_graph|)",
        "pair_graph_product": "Pair × graph (h_pair ⊙ h_graph)",
        "transformation_probabilities": "Transform probabilities (t_pred)",
        "inverse_transformation_probabilities": "Inverse transform probabilities (t_inv)",
    }
    groups = sorted(groups, key=lambda row: row["permutation_fidelity_drop_mean"])
    y = np.arange(len(groups))
    raw_values = np.asarray([row["permutation_fidelity_drop_mean"] for row in groups], dtype=float)
    raw_errors = np.asarray([row["permutation_fidelity_drop_std"] for row in groups], dtype=float)
    normalizer = float(np.abs(raw_values).sum())
    values = 100.0 * raw_values / normalizer if normalizer else raw_values
    errors = 100.0 * raw_errors / normalizer if normalizer else raw_errors

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(y, values, xerr=errors, color="#477fa8", alpha=0.9, capsize=4)
    ax.set_yticks(y, [group_labels.get(row["feature_group"], row["feature_group"]) for row in groups])
    ax.set_xlabel("Relative importance (%) | normalized to 100")
    ax.set_title("LogicPath v2 feature-fusion importance | InferBR test")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    label_offset = 1.0
    for index, value, error in zip(range(len(values)), values, errors):
        ax.text(value + error + label_offset, index, f"{value:.1f}%", va="center", fontsize=9)
    ax.set_xlim(left=min(0, float(np.min(values - errors)) - 1), right=float(np.max(values + errors)) + 8)
    fig.tight_layout()
    fig.savefig(root / "surrogate_feature_importance.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "surrogate_feature_importance.svg", bbox_inches="tight")
    plt.close(fig)

    metric_specs = {
        "binary/mcc": ("Binary MCC", "binary_mcc"),
        "binary/accuracy": ("Binary accuracy", "binary_accuracy"),
        "binary/macro_f1": ("Binary Macro-F1", "binary_macro_f1"),
    }
    cascade_variants = [
        "logicpath_ablation_transformer",
        "logicpath_ablation_states",
        "logicpath_ablation_transformations",
        "logicpath_v2",
    ]
    cascade_labels = ["Transformer only", "States graph", "Transformations", "Full + Wiktionary"]
    for metric, (metric_title, filename) in metric_specs.items():
        if aggregate:
            scores = [aggregate_metrics[variant][metric] for variant in cascade_variants]
            split_title = "Portuguese test aggregate"
        else:
            scores = [results[variant]["eval"]["inferbr_test"][metric] for variant in cascade_variants]
            split_title = "InferBR test"
        deltas = [None] + [scores[index] - scores[index - 1] for index in range(1, len(scores))]
        x = np.arange(len(scores))
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        lower = min(scores) - 0.035
        upper = max(scores) + 0.045

        # First bar is the absolute baseline; subsequent bars show each
        # component's incremental change from the preceding condition.
        ax.bar(x[0], scores[0] - lower, bottom=lower, color="#477fa8", width=0.62, label="Absolute score")
        for index in range(1, len(scores)):
            previous, current = scores[index - 1], scores[index]
            bottom = min(previous, current)
            color = "#4d9466" if current >= previous else "#c56a4b"
            ax.bar(x[index], abs(current - previous), bottom=bottom, color=color, width=0.62)
        ax.axhline(scores[0], color="#477fa8", linestyle="--", linewidth=1, alpha=0.35)
        for index, score in enumerate(scores):
            ax.text(index, score + 0.004, f"{score:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            if index > 0:
                delta = deltas[index]
                ax.text(index, max(scores[index - 1], score) + 0.011, f"{delta:+.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, cascade_labels, rotation=10, ha="right")
        ax.set_ylim(lower, upper)
        ax.set_ylabel("Score")
        ax.set_title(f"LogicPath ablation cascade | {metric_title} | {split_title}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.03, "Dashed line = Transformer-only baseline", transform=ax.transAxes, fontsize=9, color="#52616b")
        fig.tight_layout()
        fig.savefig(root / f"ablation_{filename}.png", dpi=220, bbox_inches="tight")
        fig.savefig(root / f"ablation_{filename}.svg", bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {root / 'surrogate_feature_importance.png'}")
    print(f"wrote ablation cascade charts under {root}")


if __name__ == "__main__":
    main()
