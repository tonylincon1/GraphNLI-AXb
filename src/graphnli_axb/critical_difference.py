"""Critical-difference reports for completed multi-seed experiments."""
from __future__ import annotations

import csv
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank import autorank, create_report, plot_stats
from scipy.stats import rankdata, studentized_range


DISPLAY_NAMES = {
    "kbert_injection": "K-BERT",
    "kganet_rel_gat": "KGAnet",
    "kgnli_dual": "KGNLI",
    "pair_cls": "Pair classifier",
    "piig_v4": "Our approach",
    "piig_v5": "Our approach v5",
}


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _svg_escape(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _write_autorank_plot(path: Path, result, title: str) -> None:
    """Save autorank's standard Demšar-style CD diagram as SVG."""
    figure, axis = plt.subplots(figsize=(10, 4.8))
    plot_stats(result, allow_insignificant=True, ax=axis)
    axis.set_title(title, pad=18, fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _nemenyi_cd(model_count: int, block_count: int, alpha: float) -> float:
    q_alpha = float(studentized_range.ppf(1.0 - alpha, model_count, np.inf) / math.sqrt(2.0))
    return q_alpha * math.sqrt(model_count * (model_count + 1) / (6.0 * block_count))


def _rank_groups(ordered: list[tuple[str, float]], cd: float) -> list[list[tuple[str, float]]]:
    groups = []
    for start in range(len(ordered) - 1):
        for end in range(start + 1, len(ordered)):
            interval = ordered[start : end + 1]
            if interval[-1][1] - interval[0][1] <= cd:
                groups.append(interval)
    return [
        group for group in groups
        if not any(
            len(other) > len(group)
            and other[0][1] <= group[0][1]
            and other[-1][1] >= group[-1][1]
            for other in groups
        )
    ]


def _draw_exploratory_axis(axis, ranks: dict[str, float], cd: float, blocks: int) -> None:
    ordered = sorted(ranks.items(), key=lambda item: (item[1], item[0]))
    axis.set_xlim(0.8, len(ordered) + 0.2)
    axis.set_ylim(-0.55, 0.75)
    axis.set_yticks([])
    axis.set_xlabel("Mean rank (1 = best)")
    axis.set_xticks(range(1, len(ordered) + 1))
    axis.axhline(0, color="#253746", linewidth=1.5)
    for rank in range(1, len(ordered) + 1):
        axis.plot([rank, rank], [-0.04, 0.04], color="#253746", linewidth=1)
    groups = _rank_groups(ordered, cd)
    for index, group in enumerate(groups):
        y = 0.24 + index * 0.1
        axis.plot([group[0][1], group[-1][1]], [y, y], color="#253746", linewidth=5, solid_capstyle="round")
    for index, (name, rank) in enumerate(ordered):
        y = -0.17 if index % 2 == 0 else -0.32
        axis.plot(rank, 0, "o", color="#2874a6", markersize=7)
        axis.plot([rank, rank], [-0.04, y + 0.035], color="#7f8c8d", linewidth=0.8)
        axis.text(rank, y, f"{name}\n{rank:.2f}", ha="center", va="top", fontsize=8)
    axis.text(0.02, 0.98, f"Exploratory: {blocks} seeds; autorank requires >= 5", transform=axis.transAxes, fontsize=8, va="top", color="#9c2f2f")


def _write_exploratory_plot(path: Path, ranks: dict[str, float], cd: float, title: str, blocks: int) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.8))
    _draw_exploratory_axis(axis, ranks, cd, blocks)
    axis.set_title(title, pad=18, fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)


def _autorank_text_report(result) -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        create_report(result)
    return stream.getvalue().strip()


def _report_for_split(
    runs: list[dict],
    split: str,
    metric: str,
    alpha: float,
    output_dir: Path,
    plot_results: dict | None = None,
) -> dict:
    values: dict[str, dict[str, float]] = {}
    for run in runs:
        metric_value = run.get("eval", {}).get(split, {}).get(metric)
        if metric_value is None:
            continue
        variant = run["variant"]
        block = f"seed_{run['seed']}"
        values.setdefault(variant, {})[block] = float(metric_value)
    if not values:
        return {"split": split, "status": "no_metric", "metric": metric}
    common_blocks = sorted(set.intersection(*(set(blocks) for blocks in values.values())))
    variants = sorted(variant for variant, blocks in values.items() if set(common_blocks) <= set(blocks))
    if len(variants) < 2 or len(common_blocks) < 2:
        return {
            "split": split,
            "metric": metric,
            "status": "insufficient_blocks",
            "models": variants,
            "blocks": common_blocks,
            "message": "At least two paired seeds and two models are needed for a critical-difference test.",
        }
    if len(common_blocks) < 5:
        display_variants = [DISPLAY_NAMES.get(variant, variant) for variant in variants]
        matrix = np.array([[values[variant][block] for variant in variants] for block in common_blocks], dtype=float)
        rank_matrix = np.vstack([rankdata(-row, method="average") for row in matrix])
        average_ranks = {
            display: float(rank)
            for display, rank in zip(display_variants, rank_matrix.mean(axis=0))
        }
        cd = _nemenyi_cd(len(variants), len(common_blocks), alpha)
        ordered_ranks = sorted(average_ranks.items(), key=lambda item: (item[1], item[0]))
        groups = _rank_groups(ordered_ranks, cd)
        report = {
            "split": split,
            "metric": metric,
            "status": "exploratory_insufficient_blocks",
            "alpha": alpha,
            "blocks": common_blocks,
            "models": display_variants,
            "friedman": {"statistic": None, "p_value": None},
            "autorank": {"available": False, "message": "autorank requires at least five paired seeds."},
            "nemenyi_critical_difference": cd,
            "average_ranks": average_ranks,
            "non_significant_groups": [[variant for variant, _ in group] for group in groups],
            "block_values": {
                block: {display: float(matrix[i, j]) for j, display in enumerate(display_variants)}
                for i, block in enumerate(common_blocks)
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(split)
        (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with (output_dir / f"{stem}_ranks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "average_rank"])
            for variant, rank in ordered_ranks:
                writer.writerow([variant, rank])
        _write_exploratory_plot(output_dir / f"{stem}.svg", average_ranks, cd, f"Critical Difference: {split}", len(common_blocks))
        if plot_results is not None:
            plot_results[split] = ("exploratory", average_ranks, cd, len(common_blocks))
        return report
    display_variants = [DISPLAY_NAMES.get(variant, variant) for variant in variants]
    data = pd.DataFrame(
        {display: [values[variant][block] for block in common_blocks] for variant, display in zip(variants, display_variants)},
        index=common_blocks,
    )
    result = autorank(
        data,
        alpha=float(alpha),
        order="descending",
        force_mode="nonparametric",
    )
    average_ranks = {variant: float(result.rankdf.loc[variant, "meanrank"]) for variant in result.rankdf.index}
    cd = float(result.cd) if result.cd is not None else float("nan")
    ordered_ranks = sorted(average_ranks.items(), key=lambda item: (item[1], item[0]))
    groups = []
    if not math.isnan(cd):
        for start in range(len(ordered_ranks) - 1):
            for end in range(start + 1, len(ordered_ranks)):
                interval = ordered_ranks[start : end + 1]
                if interval[-1][1] - interval[0][1] <= cd:
                    groups.append(interval)
        groups = [
            group for group in groups
            if not any(
                len(other) > len(group)
                and other[0][1] <= group[0][1]
                and other[-1][1] >= group[-1][1]
                for other in groups
            )
        ]
    report = {
        "split": split,
        "metric": metric,
        "status": "ok",
        "alpha": alpha,
        "blocks": common_blocks,
        "models": display_variants,
        "friedman": {"statistic": None, "p_value": float(result.pvalue)},
        "autorank": {
            "omnibus": result.omnibus,
            "posthoc": result.posthoc,
            "all_normal": bool(result.all_normal),
            "homoscedastic": bool(result.homoscedastic) if result.homoscedastic is not None else None,
            "effect_size": result.effect_size,
            "report": _autorank_text_report(result),
        },
        "nemenyi_critical_difference": cd,
        "average_ranks": average_ranks,
        "non_significant_groups": [[variant for variant, _ in group] for group in groups],
        "block_values": {block: {variant: float(data.loc[block, variant]) for variant in display_variants} for block in common_blocks},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(split)
    if plot_results is not None:
        plot_results[split] = result
    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / f"{stem}_ranks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "average_rank"])
        for variant, rank in sorted(report["average_ranks"].items(), key=lambda item: item[1]):
            writer.writerow([variant, rank])
    _write_autorank_plot(output_dir / f"{stem}.svg", result, f"Critical Difference: {split}")
    return report


def _write_combined_plot(output_dir: Path, results: dict[str, object]) -> None:
    if len(results) < 2:
        return
    figure, axes = plt.subplots(1, len(results), figsize=(16, 5.2), squeeze=False)
    axes_flat = axes[0]
    for axis, (split, result) in zip(axes_flat, results.items()):
        if (
            isinstance(result, tuple)
            and len(result) == 4
            and result[0] == "exploratory"
        ):
            _draw_exploratory_axis(axis, result[1], result[2], result[3])
        else:
            plot_stats(result, allow_insignificant=True, ax=axis)
        axis.set_title(f"{split}", pad=18, fontsize=14, fontweight="bold")
    figure.suptitle("Critical Difference Diagrams", fontsize=17, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_dir / "combined.svg", format="svg", bbox_inches="tight")
    plt.close(figure)


def generate_critical_difference_reports(summary_path: Path, output_dir: Path, metric: str = "binary/mcc", alpha: float = 0.05) -> list[dict]:
    runs = json.loads(summary_path.read_text(encoding="utf-8"))
    splits = sorted({split for run in runs for split in run.get("eval", {})})
    plot_results = {}
    reports = [_report_for_split(runs, split, metric, alpha, output_dir, plot_results) for split in splits]
    _write_combined_plot(output_dir, plot_results)
    (output_dir / "index.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports
