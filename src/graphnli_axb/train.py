from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .data import (
    NLIGraphDataset,
    XNLI_LANGUAGES,
    collate_nli,
    load_axb,
    load_glue_ax_unlabeled,
    load_mnli,
    load_nli_manifest,
    load_xnli,
    project_root,
    split_train_validation,
)
from .graph import RELATIONS
from .logicpath import (
    LOGICPATH_NODE_TYPES,
    LOGICPATH_RELATIONS,
    LOGICPATH_V2_NODE_TYPES,
    LOGICPATH_V2_RELATIONS,
    LOGICPATH_V3_NODE_TYPES,
    LOGICPATH_V3_RELATIONS,
    TRANSFORMATIONS,
)
from .proof_graph import PROOF_NODE_TYPES, PROOF_RELATIONS
from .metrics import (
    binary_axb_metrics,
    classification_metrics,
    direct_binary_metrics,
    grouped_binary_metrics,
    grouped_direct_binary_metrics,
)
from .models import build_model
from .models_tokgraph import TOKGRAPH_VARIANTS
from .critical_difference import generate_critical_difference_reports


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _tokgraph_lexicon_paths(args) -> list[Path]:
    paths = []
    for value in (getattr(args, "pt_lkb_cache", None), getattr(args, "wiktionary_cache", None)):
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = project_root() / path
        if path.exists():
            paths.append(path)
    return paths


def make_loader(rows, tokenizer, args, shuffle: bool, include_piig: bool = False, piig_version: str = "v1", include_logicpath: bool = False, logicpath_version: str = "v1", logicpath_lexicon: str = "Wiktionary", include_tokgraph: bool = False) -> DataLoader:
    wikidata_cache_path = Path(args.wikidata_cache) if args.wikidata_cache else None
    if wikidata_cache_path is not None and not wikidata_cache_path.is_absolute():
        wikidata_cache_path = project_root() / wikidata_cache_path
    wiktionary_cache_path = Path(args.wiktionary_cache) if args.wiktionary_cache else None
    if wiktionary_cache_path is not None and not wiktionary_cache_path.is_absolute():
        wiktionary_cache_path = project_root() / wiktionary_cache_path
    dataset = NLIGraphDataset(
        rows,
        tokenizer,
        max_len=args.max_len,
        max_nodes=args.max_nodes,
        max_node_len=args.max_node_len,
        max_wikidata_entities=args.max_wikidata_entities,
        max_wikidata_len=args.max_wikidata_len,
        wikidata_cache_path=wikidata_cache_path,
        wiktionary_cache_path=wiktionary_cache_path,
        max_wiktionary_entries=args.max_wiktionary_entries,
        include_piig=include_piig,
        piig_version=piig_version,
        max_proof_nodes=args.max_proof_nodes,
        max_proof_len=args.max_proof_len,
        include_logicpath=include_logicpath,
        logicpath_version=logicpath_version,
        logicpath_lexicon=logicpath_lexicon,
        max_logicpath_nodes=args.max_logicpath_nodes,
        max_logicpath_len=args.max_logicpath_len,
        include_tokgraph=include_tokgraph,
        tokgraph_lexicon_paths=_tokgraph_lexicon_paths(args) if include_tokgraph else None,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_nli,
    )


def compute_loss(output, labels: torch.Tensor) -> torch.Tensor:
    if getattr(output, "main_loss", None) is not None:
        loss = output.main_loss
    elif (labels >= 0).any():
        loss = torch.nn.functional.cross_entropy(output.logits, labels, ignore_index=-100)
    else:
        loss = output.logits.sum() * 0.0
    if output.aux_loss is not None:
        loss = loss + output.aux_loss
    return loss


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> tuple[Dict[str, float], List[dict]]:
    model.eval()
    all_logits, all_binary_logits, all_labels3, all_binary = [], [], [], []
    all_route_gates, all_latent_relation_mass = [], []
    all_logic, all_uid, all_language = [], [], []
    all_diagnostics: Dict[str, list] = {}
    total_loss = 0.0
    steps = 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        loss = compute_loss(output, batch["labels_3"])
        total_loss += float(loss.detach().cpu())
        steps += 1
        all_logits.append(output.logits.detach().cpu().numpy())
        if output.binary_logits is not None:
            all_binary_logits.append(output.binary_logits.detach().cpu().numpy())
        if output.route_gates is not None:
            all_route_gates.append(output.route_gates.detach().cpu().numpy())
        if output.latent_relation_mass is not None:
            all_latent_relation_mass.append(output.latent_relation_mass.detach().cpu().numpy())
        if output.diagnostics:
            for key, value in output.diagnostics.items():
                all_diagnostics.setdefault(key, []).append(value.detach().float().cpu().numpy())
        all_labels3.append(batch["labels_3"].detach().cpu().numpy())
        all_binary.append(batch["labels_binary"].detach().cpu().numpy())
        all_logic.extend(batch["logic"])
        all_uid.extend(batch["uid"])
        all_language.extend(batch["language"])

    logits = np.concatenate(all_logits, axis=0)
    binary_logits = np.concatenate(all_binary_logits, axis=0) if all_binary_logits else None
    route_gates = np.concatenate(all_route_gates, axis=0) if all_route_gates else None
    latent_relation_mass = np.concatenate(all_latent_relation_mass, axis=0) if all_latent_relation_mass else None
    labels3 = np.concatenate(all_labels3, axis=0)
    labels_binary = np.concatenate(all_binary, axis=0)
    pred3 = logits.argmax(axis=1)

    metrics = {"loss": total_loss / max(steps, 1)}
    metrics.update(classification_metrics(logits, labels3, prefix="nli3/"))
    if binary_logits is not None:
        metrics.update(direct_binary_metrics(binary_logits, labels_binary, prefix="binary/"))
        metrics.update(grouped_direct_binary_metrics(binary_logits, labels_binary, all_logic))
        binary_pred = binary_logits.argmax(axis=1)
    else:
        metrics.update(binary_axb_metrics(logits, labels_binary, prefix="binary/"))
        metrics.update(grouped_binary_metrics(logits, labels_binary, all_logic))
        binary_pred = np.array([0 if int(label) == 0 else 1 for label in pred3], dtype=np.int64)

    diagnostics = {key: np.concatenate(parts, axis=0) for key, parts in all_diagnostics.items()}
    metrics.update(diagnostic_metrics(diagnostics, logits, labels3))

    predictions = []
    route_names = RELATIONS + ["semantic_alignment"]
    for idx, uid in enumerate(all_uid):
        prediction = {
            "uid": uid,
            "prediction_3": int(pred3[idx]),
            "label_3": int(labels3[idx]),
            "prediction_binary": int(binary_pred[idx]),
            "label_binary": int(labels_binary[idx]),
            "logic": all_logic[idx],
            "language": all_language[idx],
        }
        if route_gates is not None:
            prediction.update({f"route_gate/{name}": float(route_gates[idx, route_id]) for route_id, name in enumerate(route_names)})
        if latent_relation_mass is not None:
            prediction.update({f"latent_relation/{relation_id}": float(value) for relation_id, value in enumerate(latent_relation_mass[idx])})
        for key, values in diagnostics.items():
            if key.endswith("_logits"):
                prediction[f"diag/{key[:-7]}_pred"] = int(values[idx].argmax())
            elif values.ndim == 1:
                prediction[f"diag/{key}"] = float(values[idx])
        predictions.append(prediction)
    return metrics, predictions


def diagnostic_metrics(diagnostics: Dict[str, np.ndarray], logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Summaries of model-provided diagnostics (e.g. COVER-PL: structured head alone,
    [CLS] head alone, how often the structured scores change the [CLS] decision)."""
    if not diagnostics:
        return {}
    metrics: Dict[str, float] = {}
    final = logits.argmax(axis=1)
    valid = labels >= 0
    for key, values in diagnostics.items():
        if key.endswith("_logits") and values.ndim == 2:
            name = key[: -len("_logits")]
            for metric, value in classification_metrics(values, labels, prefix=f"diag/{name}/").items():
                if metric.endswith(("accuracy", "macro_f1", "mcc")):
                    metrics[metric] = value
        elif values.ndim == 1:
            metrics[f"diag/{key}_mean"] = float(np.mean(values))
    if "cls_logits" in diagnostics:
        cls_pred = diagnostics["cls_logits"].argmax(axis=1)
        changed = (cls_pred != final) & valid
        metrics["diag/changed_rate"] = float(changed.mean())
        metrics["diag/changed_fixed_rate"] = float((changed & (final == labels)).mean())
        metrics["diag/changed_broke_rate"] = float((changed & (cls_pred == labels)).mean())
    return metrics


def save_predictions(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_fusion_features(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Collect final fusion vectors and the model's predictions for a surrogate."""
    model.eval()
    features, predictions = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        if output.fusion_features is None:
            raise RuntimeError("This model does not expose feature-fusion vectors.")
        features.append(output.fusion_features.detach().cpu().numpy())
        predictions.append(output.logits.argmax(dim=-1).detach().cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(predictions, axis=0)


def save_surrogate_analysis(model, loader, device: torch.device, output_dir: Path) -> dict:
    """Fit a small interpretable surrogate and measure grouped feature importance.

    Importance is the drop in surrogate fidelity after permuting one semantic
    block of the neural feature fusion vector. This keeps the analysis at the
    level of architectural components instead of thousands of hidden units.
    """
    features, black_box_predictions = collect_fusion_features(model, loader, device)
    groups = getattr(model, "fusion_feature_groups", None)
    if not groups:
        raise RuntimeError("The selected model has no named feature-fusion groups.")
    if len(np.unique(black_box_predictions)) < 2:
        raise RuntimeError("Surrogate analysis needs at least two predicted classes.")

    indices = np.arange(len(features))
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.25,
            random_state=42,
            stratify=black_box_predictions,
        )
    except ValueError:
        train_idx, test_idx = indices, indices
    surrogate = LogisticRegression(max_iter=1000, random_state=42)
    surrogate.fit(features[train_idx], black_box_predictions[train_idx])
    test_features = features[test_idx]
    test_targets = black_box_predictions[test_idx]
    baseline = float((surrogate.predict(test_features) == test_targets).mean())
    rng = np.random.default_rng(42)
    importance = []
    for name, (start, end) in groups.items():
        scores = []
        for _ in range(10):
            permuted = test_features.copy()
            order = rng.permutation(len(permuted))
            permuted[:, start:end] = permuted[order, start:end]
            score = float((surrogate.predict(permuted) == test_targets).mean())
            scores.append(baseline - score)
        importance.append(
            {
                "feature_group": name,
                "start": int(start),
                "end": int(end),
                "dimensions": int(end - start),
                "permutation_fidelity_drop_mean": float(np.mean(scores)),
                "permutation_fidelity_drop_std": float(np.std(scores)),
            }
        )
    importance.sort(key=lambda row: row["permutation_fidelity_drop_mean"], reverse=True)
    report = {
        "method": "logistic-regression surrogate with grouped permutation importance",
        "samples": int(len(features)),
        "feature_dimensions": int(features.shape[1]),
        "surrogate_train_samples": int(len(train_idx)),
        "surrogate_test_samples": int(len(test_idx)),
        "surrogate_fidelity_accuracy": baseline,
        "black_box_predicted_classes": sorted(int(value) for value in np.unique(black_box_predictions)),
        "groups": importance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "surrogate_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "surrogate_analysis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(importance[0].keys()))
        writer.writeheader()
        writer.writerows(importance)
    return report


def _svg_label(text: str, max_chars: int = 34, max_lines: int = 3) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = f"{lines[-1][: max_chars - 3]}..."
    return lines


def _write_proof_graph_svg(path: Path, record: dict) -> None:
    """Render a dependency-free graph image from PIIG attention attribution."""
    nodes = record["nodes"]
    edges = record["edges"]
    width = 1260
    height = max(620, 250 + len(nodes) * 80)
    center_x, center_y = 430, height // 2 + 40
    radius = min(250, max(135, 40 * len(nodes)))
    positions = []
    for index in range(len(nodes)):
        angle = -np.pi / 2 + 2 * np.pi * index / max(len(nodes), 1)
        positions.append((center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#17202a}.small{font-size:12px}.title{font-size:18px;font-weight:bold}.node{font-size:12px;font-weight:bold}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="30" y="32" class="title">PIIG attention explanation | {html.escape(record["uid"])}</text>',
        f'<text x="30" y="54" class="small">Prediction: {record["prediction_binary"]} | label: {record["label_binary"]} | graph-node attribution combines cross-attention and GAT attention.</text>',
    ]
    for edge in edges:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        opacity = min(0.92, 0.15 + edge["attention"] * 2.5)
        stroke = 1 + edge["attention"] * 7
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#4d6f8e" stroke-width="{stroke:.2f}" stroke-opacity="{opacity:.2f}"/>')
    for index, node in enumerate(nodes):
        x, y = positions[index]
        score = node["importance"]
        red = int(245 - 120 * score)
        green = int(250 - 55 * score)
        blue = int(255 - 20 * score)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="31" fill="rgb({red},{green},{blue})" stroke="#1f4d70" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y - 3:.1f}" text-anchor="middle" class="node">N{index}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y + 13:.1f}" text-anchor="middle" class="small">{score:.3f}</text>')

    legend_x = 760
    parts.append(f'<text x="{legend_x}" y="96" class="title">Nodes</text>')
    for index, node in enumerate(nodes):
        y = 125 + index * 72
        parts.append(f'<circle cx="{legend_x + 10}" cy="{y - 4}" r="7" fill="#6b9bc3"/>')
        parts.append(f'<text x="{legend_x + 26}" y="{y}" class="node">N{index} | {html.escape(node["type"])} | {node["importance"]:.3f}</text>')
        for line_index, line in enumerate(_svg_label(node["text"])):
            parts.append(f'<text x="{legend_x + 26}" y="{y + 16 * (line_index + 1)}" class="small">{html.escape(line)}</text>')
    parts.append(f'<text x="30" y="{height - 24}" class="small">Edge width and opacity: mean GAT attention across PIIG graph layers. Node score: normalized mean cross-attention plus incoming/outgoing GAT attention.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _logic_filename(logic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", logic.lower()).strip("_")
    return slug or "unknown"


def _proof_explanation_record(batch: dict, output, item_index: int) -> dict:
    valid_nodes = int(batch["proof_mask"][item_index].sum().item())
    node_type_names = list(PROOF_NODE_TYPES)
    texts = batch["proof_texts"][item_index][:valid_nodes]
    importance = output.proof_node_importance[item_index, :valid_nodes].detach().cpu().tolist()
    edge_types = batch["proof_edge_types"][item_index, :valid_nodes, :valid_nodes].detach().cpu().numpy()
    attention = output.proof_graph_attention[:, item_index, :valid_nodes, :valid_nodes].mean(dim=0).detach().cpu().numpy()
    nodes = [
        {
            "id": index,
            "text": text,
            "type": node_type_names[int(batch["proof_type_ids"][item_index, index].item())],
            "importance": float(importance[index]),
        }
        for index, text in enumerate(texts)
    ]
    edges = []
    for source in range(valid_nodes):
        for target in range(source + 1, valid_nodes):
            if edge_types[source, target] == 0 and edge_types[target, source] == 0:
                continue
            relation_id = int(edge_types[source, target] or edge_types[target, source])
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": PROOF_RELATIONS[relation_id],
                    "attention": float((attention[source, target] + attention[target, source]) / 2),
                }
            )
    prediction_binary = int(output.binary_logits.argmax(dim=-1)[item_index].item())
    return {
        "uid": batch["uid"][item_index],
        "dataset": batch["dataset"][item_index],
        "logic": batch["logic"][item_index],
        "language": batch["language"][item_index],
        "prediction_binary": prediction_binary,
        "label_binary": int(batch["labels_binary"][item_index].item()),
        "attribution_method": "normalized mean cross-attention plus mean incoming/outgoing relation-aware GAT attention",
        "nodes": nodes,
        "edges": edges,
    }


@torch.no_grad()
def save_piig_explanations(model, loader, device: torch.device, output_dir: Path) -> None:
    """Save subtype graphs plus explicit Wiktionary-covered examples when present."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_logic = set()
    manifest = []
    wiktionary_dir = output_dir / "wiktionary_examples"
    wiktionary_manifest = []
    wiktionary_node_types = {"wiktionary_entry", "wiktionary_gloss", "wiktionary_relation"}
    for batch in loader:
        batch = move_batch(batch, device)
        batch["return_proof_attention"] = True
        output = model(batch)
        if output.proof_node_importance is None or output.proof_graph_attention is None:
            raise RuntimeError("PIIG model did not return proof-graph attention for explanation.")
        for item_index, logic in enumerate(batch["logic"]):
            needs_logic_example = logic not in seen_logic
            needs_wiktionary_example = len(wiktionary_manifest) < 3
            if not needs_logic_example and not needs_wiktionary_example:
                continue
            record = _proof_explanation_record(batch, output, item_index)
            filename = _logic_filename(logic)
            if needs_logic_example:
                seen_logic.add(logic)
                path = output_dir / filename
                path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                _write_proof_graph_svg(path.with_suffix(".svg"), record)
                manifest.append({"logic": logic, "uid": record["uid"], "file": f"{filename}.svg"})
            if needs_wiktionary_example and any(node["type"] in wiktionary_node_types for node in record["nodes"]):
                wiktionary_dir.mkdir(parents=True, exist_ok=True)
                external_filename = f"wiktionary_{len(wiktionary_manifest) + 1:02d}_{filename}"
                external_path = wiktionary_dir / external_filename
                external_path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                _write_proof_graph_svg(external_path.with_suffix(".svg"), record)
                wiktionary_manifest.append(
                    {"logic": logic, "uid": record["uid"], "file": f"{external_filename}.svg"}
                )
    (output_dir / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if wiktionary_manifest:
        (wiktionary_dir / "index.json").write_text(json.dumps(wiktionary_manifest, indent=2), encoding="utf-8")


def _logicpath_explanation_record(batch: dict, output, item_index: int) -> dict:
    valid_nodes = int(batch["logicpath_mask"][item_index].sum().item())
    texts = batch["logicpath_texts"][item_index][:valid_nodes]
    importance = output.logicpath_node_importance[item_index, :valid_nodes].detach().cpu().tolist()
    edge_types = batch["logicpath_edge_types"][item_index, :valid_nodes, :valid_nodes].detach().cpu().numpy()
    attention = output.logicpath_graph_attention[:, item_index, :valid_nodes, :valid_nodes].mean(dim=0).detach().cpu().numpy()
    initial_attention = (
        output.logicpath_initial_graph_attention[item_index, :valid_nodes, :valid_nodes].detach().cpu().numpy()
        if output.logicpath_initial_graph_attention is not None else attention
    )
    rewritten_attention = (
        output.logicpath_rewritten_graph_attention[item_index, :valid_nodes, :valid_nodes].detach().cpu().numpy()
        if output.logicpath_rewritten_graph_attention is not None else attention
    )
    logicpath_version = batch.get("logicpath_version", ["v1"])[item_index]
    if logicpath_version in {"v2", "v3"}:
        node_type_names = LOGICPATH_V3_NODE_TYPES if logicpath_version == "v3" else LOGICPATH_V2_NODE_TYPES
        relation_names = LOGICPATH_V3_RELATIONS if logicpath_version == "v3" else LOGICPATH_V2_RELATIONS
    elif logicpath_version == "no_transform_wiktionary_v1":
        node_type_names = LOGICPATH_V2_NODE_TYPES
        relation_names = LOGICPATH_V2_RELATIONS
    else:
        node_type_names = LOGICPATH_NODE_TYPES
        relation_names = LOGICPATH_RELATIONS
    nodes = [
        {
            "id": index,
            "text": text,
            "type": node_type_names[int(batch["logicpath_type_ids"][item_index, index].item())],
            "importance": float(importance[index]),
        }
        for index, text in enumerate(texts)
    ]
    edges = []
    for source in range(valid_nodes):
        for target in range(source + 1, valid_nodes):
            relation_id = int(edge_types[source, target] or edge_types[target, source])
            if relation_id == 0:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation_names[relation_id],
                    "attention": float((attention[source, target] + attention[target, source]) / 2),
                    "initial_attention": float((initial_attention[source, target] + initial_attention[target, source]) / 2),
                    "rewritten_attention": float((rewritten_attention[source, target] + rewritten_attention[target, source]) / 2),
                }
            )
    if output.logicpath_transform_logits is None:
        transform_scores = [0.0] * len(TRANSFORMATIONS)
    else:
        transform_scores = output.logicpath_transform_logits[item_index].sigmoid().detach().cpu().tolist()
    transform_targets = batch["logicpath_targets"][item_index].detach().cpu().tolist()
    record = {
        "uid": batch["uid"][item_index],
        "dataset": batch["dataset"][item_index],
        "logic": batch["logic"][item_index],
        "language": batch["language"][item_index],
        "logicpath_version": logicpath_version,
        "prediction_3": int(output.logits.argmax(dim=-1)[item_index].item()),
        "label_3": int(batch["labels_3"][item_index].item()),
        "prediction_binary": int(output.logits.argmax(dim=-1)[item_index].item() != 0),
        "label_binary": int(batch["labels_binary"][item_index].item()),
        "transformations": {
            name: {"score": float(transform_scores[index]), "target": float(transform_targets[index])}
            for index, name in enumerate(TRANSFORMATIONS)
        },
        "attribution_method": "normalized mean relation-aware Graph Transformer attention",
        "nodes": nodes,
        "edges": edges,
    }
    if output.logicpath_initial_node_importance is not None:
        record["initial_node_importance"] = output.logicpath_initial_node_importance[item_index, :valid_nodes].detach().cpu().tolist()
        record["rewritten_node_importance"] = output.logicpath_rewritten_node_importance[item_index, :valid_nodes].detach().cpu().tolist()
        record["rewrite"] = {
            "selected_operation": max(record["transformations"], key=lambda name: record["transformations"][name]["score"]),
            "selected_probability": max(value["score"] for value in record["transformations"].values()),
            "interpretation": "Node states are updated by the learned relation-aware rewrite layers; text and edges remain fixed.",
        }
    return record


def _write_logicpath_graph_svg(path: Path, record: dict) -> None:
    """Render a readable, layered transformation flow with adaptive node cards."""
    nodes = record["nodes"]
    edges = record["edges"]
    has_wiktionary = any(node["type"].startswith("wiktionary_") for node in nodes)
    if has_wiktionary:
        width = 2200
        columns = {
            "premise_state": (125, "Premise"),
            "premise_unit": (410, "Premise units"),
            "wiktionary_entry": (690, "Wiktionary entry"),
            "wiktionary_gloss": (900, "Wiktionary gloss"),
            "transformation": (1130, "Transformations"),
            "hypothesis_unit": (1430, "Hypothesis units"),
            "hypothesis_state": (1960, "Hypothesis"),
        }
    else:
        width = 1900
        columns = {
            "premise_state": (125, "Premise"),
            "premise_unit": (475, "Premise units"),
            "transformation": (950, "Transformations"),
            "hypothesis_unit": (1425, "Hypothesis units"),
            "hypothesis_state": (1775, "Hypothesis"),
        }
    node_width = 190
    card_padding = 32
    line_height = 15

    def node_lines(node: dict) -> list[str]:
        # Keep the complete semantic content in the SVG while wrapping it inside the card.
        return _svg_label(node["text"], max_chars=28, max_lines=12)

    node_heights = {}
    for index, node in enumerate(nodes):
        line_count = len(node_lines(node))
        node_heights[index] = max(74, card_padding + 18 + line_count * line_height)

    grouped = {node_type: [] for node_type in columns}
    for index, node in enumerate(nodes):
        grouped.setdefault(node["type"], []).append(index)
    max_column_nodes = max((len(indices) for indices in grouped.values()), default=1)
    height = max(820, 260 + max_column_nodes * 150)
    positions = {}
    for node_type, indices in grouped.items():
        x = columns.get(node_type, (720, node_type))[0]
        if len(indices) == 1:
            single_y = (height - 80) / 2
            if has_wiktionary and node_type == "wiktionary_entry":
                single_y = 275
            elif has_wiktionary and node_type == "wiktionary_gloss":
                single_y = 535
            y_positions = [single_y]
        else:
            top = 150
            bottom = height - 150
            step = (bottom - top) / (len(indices) - 1)
            y_positions = [top + step * order for order in range(len(indices))]
        for index, y in zip(indices, y_positions):
            positions[index] = (x, y)
    rank = {node_type: order for order, node_type in enumerate(columns)}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 z" fill="#4d6f8e"/></marker><marker id="red-arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 z" fill="#a33d3d"/></marker></defs>',
        '<style>text{font-family:Arial,sans-serif;fill:#17202a}.small{font-size:12px}.title{font-size:18px;font-weight:bold}.node{font-size:12px;font-weight:bold}.node-text{font-size:11px}.column{font-size:14px;font-weight:bold;fill:#52616b}.edge-label{font-size:10px;fill:#52616b}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="30" y="32" class="title">LogicPath transformation graph | {html.escape(record["uid"])}</text>',
        f'<text x="30" y="54" class="small">Binary prediction: {record["prediction_binary"]} | binary label: {record["label_binary"]} | arrows show transformation flow; width shows attention.</text>',
    ]
    for node_type, (x, label) in columns.items():
        parts.append(f'<text x="{x}" y="92" text-anchor="middle" class="column">{html.escape(label)}</text>')
    edge_lanes = {}
    for edge in edges:
        key = (min(edge["source"], edge["target"]), max(edge["source"], edge["target"]))
        edge_lanes[key] = edge_lanes.get(key, 0) + 1
    seen_lanes = {}
    for edge in edges:
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        source_index, target_index = edge["source"], edge["target"]
        if rank.get(source["type"], 2) > rank.get(target["type"], 2):
            source_index, target_index = target_index, source_index
        x1, y1 = positions[source_index]
        x2, y2 = positions[target_index]
        direction = 1 if x2 >= x1 else -1
        x1 += direction * (node_width / 2)
        x2 -= direction * (node_width / 2)
        opacity = min(0.92, 0.15 + edge["attention"] * 2.5)
        stroke = 1 + edge["attention"] * 7
        color = "#9c3f3f" if edge["relation"] == "contradicts" else "#4d6f8e"
        marker = "red-arrow" if edge["relation"] == "contradicts" else "arrow"
        key = (min(edge["source"], edge["target"]), max(edge["source"], edge["target"]))
        lane_index = seen_lanes.get(key, 0)
        seen_lanes[key] = lane_index + 1
        lane_count = edge_lanes[key]
        lane_offset = (lane_index - (lane_count - 1) / 2) * 16
        if edge["relation"] == "contradicts":
            lane_offset += 90
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + lane_offset
        path_data = f"M{x1:.1f},{y1:.1f} Q{mid_x:.1f},{mid_y:.1f} {x2:.1f},{y2:.1f}"
        parts.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="{stroke:.2f}" stroke-opacity="{opacity:.2f}" marker-end="url(#{marker})"/>')
        # Label only salient edges; low-attention edges remain visible without
        # turning dense examples into a wall of overlapping relation names.
        if edge["attention"] >= 0.08:
            label_x = mid_x
            label_y = mid_y - 5
            label = html.escape(f'{edge["relation"]} | w={edge["attention"]:.2f}')
            label_width = max(54, len(f'{edge["relation"]} | w={edge["attention"]:.2f}') * 5.4 + 8)
            parts.append(f'<rect x="{label_x - label_width / 2:.1f}" y="{label_y - 10:.1f}" width="{label_width:.1f}" height="13" rx="3" fill="#ffffff" fill-opacity="0.88"/>')
            parts.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" class="edge-label">{label}</text>')
    for index, node in enumerate(nodes):
        x, y = positions[index]
        score = node["importance"]
        fill = "#e8f3fb" if node["type"] in {"premise_state", "hypothesis_state"} else "#f7f1df" if node["type"] == "transformation" else "#eef5ee"
        card_height = node_heights[index]
        x_left = x - node_width / 2
        parts.append(f'<rect x="{x_left:.1f}" y="{y - card_height / 2:.1f}" width="{node_width}" height="{card_height:.1f}" rx="8" fill="{fill}" stroke="#1f4d70" stroke-width="2"/>')
        lines = node_lines(node)
        title_y = y - card_height / 2 + 21
        parts.append(f'<text x="{x:.1f}" y="{title_y:.1f}" text-anchor="middle" class="node">N{index} | {score:.3f}</text>')
        for line_index, line in enumerate(lines):
            text_y = title_y + 17 + line_height * line_index
            parts.append(f'<text x="{x:.1f}" y="{text_y:.1f}" text-anchor="middle" class="node-text">{html.escape(line)}</text>')
    parts.append(f'<text x="30" y="{height - 24}" class="small">Node score: normalized relation-aware attention. Edge width shows attention; labels are shown for salient edges.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_graph_rewrite_svg(path: Path, record: dict) -> None:
    """Render a publication-style before/operation/after explanation."""
    nodes = record["nodes"]
    edges = record["edges"]
    initial = record.get("initial_node_importance", [node["importance"] for node in nodes])
    rewritten = record.get("rewritten_node_importance", [node["importance"] for node in nodes])
    selected = record.get("rewrite", {}).get("selected_operation", "unknown")
    selected_probability = record.get("rewrite", {}).get("selected_probability", 0.0)
    width, height = 2100, 1040
    panel_y, panel_h = 130, 780
    panel_w = 650
    palette = {
        "premise_state": ("#e7f0f7", "#2f6f9f"),
        "hypothesis_state": ("#e7f0f7", "#2f6f9f"),
        "premise_unit": ("#eef5ef", "#4e8b62"),
        "hypothesis_unit": ("#eef5ef", "#4e8b62"),
        "transformation": ("#fff1d6", "#c27a1a"),
        "wiktionary_entry": ("#f1ecfa", "#8064a2"),
        "wiktionary_gloss": ("#f1ecfa", "#8064a2"),
    }

    def pretty_name(value: str) -> str:
        return value.replace("_", " ").title()

    def label_lines(text: str, limit: int = 19, max_lines: int = 4) -> list[str]:
        words, lines, current = text.split(), [], ""
        for word in words:
            if len(current) + len(word) + 1 > limit and current:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
        return lines[:max_lines]

    def node_lines(node: dict) -> list[str]:
        text = node["text"]
        if ":" not in text:
            return label_lines(text)
        prefix, value = text.split(":", 1)
        prefix = prefix.strip()
        value = value.strip()
        if prefix == "transformation":
            operation, _, direction = value.partition(":")
            return ["Transformation", pretty_name(operation.strip()), direction.strip() or "Premise → Hypothesis"]
        return [pretty_name(prefix), *label_lines(value, 19, max_lines=3)][:4]

    grouped = {}
    for index, node in enumerate(nodes):
        grouped.setdefault(node["type"], []).append(index)

    def draw_graph(parts: list[str], offset: int, title: str, scores: list[float], final: bool) -> None:
        parts.append(f'<rect x="{offset}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="18" fill="#ffffff" stroke="#d5dee7" stroke-width="2"/>')
        parts.append(f'<text x="{offset + 28}" y="{panel_y + 42}" class="panel-title">{html.escape(title)}</text>')
        parts.append(f'<text x="{offset + 28}" y="{panel_y + 67}" class="panel-subtitle">{("attention-weighted rewritten states" if final else "input evidence and initial states")}</text>')
        positions = {}
        # Pyramid layout: sentence states at the top, lexical units in the
        # middle, and the learned transformation at the central base.
        fixed_positions = {
            "premise_state": (145, 235),
            "hypothesis_state": (505, 235),
            "premise_unit": (175, 430),
            "hypothesis_unit": (475, 430),
            "transformation": (325, 625),
            "wiktionary_entry": (325, 390),
            "wiktionary_gloss": (325, 510),
        }
        for node_type, indices in grouped.items():
            base_x, base_y = fixed_positions.get(node_type, (325, 430))
            count = len(indices)
            spacing = 145 if count <= 3 else 105
            for order, index in enumerate(indices):
                positions[index] = (offset + base_x + (order - (count - 1) / 2) * spacing, panel_y + base_y)
        for index in range(len(nodes)):
            positions.setdefault(index, (offset + panel_w / 2, panel_y + 210 + index * 80))

        for edge in edges:
            x1, y1 = positions[edge["source"]]
            x2, y2 = positions[edge["target"]]
            weight = float(edge.get("rewritten_attention", edge.get("attention", 0.0))) if final else float(edge.get("initial_attention", edge.get("attention", 0.0)))
            strength = weight
            opacity = min(0.9, 0.18 + strength * 2.2)
            stroke = 1.2 + strength * 5.0
            color = "#b24a4a" if edge["relation"] == "contradicts" else "#7c93a8"
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{stroke:.2f}" stroke-opacity="{opacity:.2f}" marker-end="url(#arrow)"/>')
            label_x, label_y = (x1 + x2) / 2, (y1 + y2) / 2
            parts.append(f'<rect x="{label_x - 18:.1f}" y="{label_y - 9:.1f}" width="36" height="16" rx="4" fill="white" fill-opacity="0.88"/>')
            parts.append(f'<text x="{label_x:.1f}" y="{label_y + 3:.1f}" text-anchor="middle" class="edge-weight">w={weight:.2f}</text>')

        for index, node in enumerate(nodes):
            x, y = positions[index]
            fill, stroke_color = palette.get(node["type"], ("#f2f4f5", "#667781"))
            score = float(scores[index])
            delta = float(rewritten[index]) - float(initial[index])
            if final and abs(delta) >= 0.025:
                stroke_color = "#bd5a3a" if delta > 0 else "#477aa1"
                fill = "#fff0e8" if delta > 0 else "#eaf3fb"
            parts.append(f'<rect x="{x - 82:.1f}" y="{y - 55:.1f}" width="164" height="110" rx="12" fill="{fill}" stroke="{stroke_color}" stroke-width="2.5"/>')
            node_label = f"N{index}'" if final else f"N{index}"
            parts.append(f'<text x="{x:.1f}" y="{y - 35:.1f}" text-anchor="middle" class="node-title">{node_label}  |  {score:.3f}</text>')
            for line_index, line in enumerate(node_lines(node)):
                parts.append(f'<text x="{x:.1f}" y="{y - 12 + 14 * line_index:.1f}" text-anchor="middle" class="node-text">{html.escape(line)}</text>')
            if final:
                sign = "+" if delta >= 0 else ""
                parts.append(f'<text x="{x:.1f}" y="{y + 43:.1f}" text-anchor="middle" class="delta">Δ {sign}{delta:.3f}</text>')
        if final:
            parts.append(f'<text x="{offset + 28}" y="{panel_y + panel_h - 28}" class="legend">Orange/blue outline = state change relative to initial graph</text>')
        else:
            parts.append(f'<text x="{offset + 28}" y="{panel_y + panel_h - 28}" class="legend">Node value = normalized initial state magnitude</text>')

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="7" refX="7" refY="3.5" orient="auto"><path d="M0,0 L8,3.5 L0,7 z" fill="#7c93a8"/></marker></defs>',
        '<style>text{font-family:Arial,sans-serif;fill:#1d2b36}.title{font-size:29px;font-weight:bold}.subtitle{font-size:15px;fill:#5f7180}.panel-title{font-size:20px;font-weight:bold;fill:#294d68}.panel-subtitle{font-size:13px;fill:#718391}.node-title{font-size:12px;font-weight:bold}.node-text{font-size:11px;fill:#33434f}.delta{font-size:11px;font-weight:bold;fill:#a65334}.edge-weight{font-size:9px;font-weight:bold;fill:#5f7180}.legend{font-size:12px;fill:#718391}.operation-title{font-size:20px;font-weight:bold;fill:#8b591a}.operation{font-size:13px;font-weight:bold}.prob{font-size:12px;fill:#5d6d78}.selected{font-size:16px;font-weight:bold;fill:#a65d18}</style>',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        f'<text x="42" y="48" class="title">Graph Rewrite v1: before, operation, after</text>',
        f'<text x="42" y="78" class="subtitle">{html.escape(record["uid"])}  |  prediction {record["prediction_binary"]}  |  label {record["label_binary"]}</text>',
    ]
    draw_graph(parts, 42, "A  Initial graph", initial, False)
    draw_graph(parts, 1408, "C  Rewritten graph", rewritten, True)

    op_x, op_y, op_w = 715, panel_y, 650
    parts.append(f'<rect x="{op_x}" y="{op_y}" width="{op_w}" height="{panel_h}" rx="18" fill="#fffaf1" stroke="#e6c98e" stroke-width="2"/>')
    parts.append(f'<text x="{op_x + 30}" y="{op_y + 42}" class="operation-title">B  Rewrite decision</text>')
    parts.append(f'<text x="{op_x + 30}" y="{op_y + 68}" class="panel-subtitle">sigmoid operation scores predicted by the network</text>')
    y = op_y + 125
    for name, values in sorted(record["transformations"].items(), key=lambda item: item[1]["score"], reverse=True):
        score = float(values["score"])
        bar_x, bar_w = op_x + 280, 255
        color = "#c2761f" if name == selected else "#b9c7d1"
        parts.append(f'<text x="{op_x + 30}" y="{y}" class="operation">{html.escape(pretty_name(name))}</text>')
        parts.append(f'<rect x="{bar_x}" y="{y - 13}" width="{bar_w}" height="16" rx="8" fill="#e7edf1"/><rect x="{bar_x}" y="{y - 13}" width="{bar_w * score:.1f}" height="16" rx="8" fill="{color}"/>')
        parts.append(f'<text x="{bar_x + bar_w + 18}" y="{y}" class="prob">{score:.3f}</text>')
        y += 35
    parts.append(f'<line x1="{op_x + 30}" y1="{y + 12}" x2="{op_x + op_w - 30}" y2="{y + 12}" stroke="#ead9b8"/>')
    parts.append(f'<text x="{op_x + 30}" y="{y + 52}" class="selected">Selected operation: {html.escape(pretty_name(selected))}</text>')
    parts.append(f'<text x="{op_x + 30}" y="{y + 79}" class="prob">confidence score: {selected_probability:.3f}</text>')
    parts.append(f'<text x="{op_x + 30}" y="{op_y + panel_h - 48}" class="legend">The graph topology is preserved; node states are updated</text>')
    parts.append(f'<text x="{op_x + 30}" y="{op_y + panel_h - 28}" class="legend">by relation-aware message passing and the selected operation.</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


@torch.no_grad()
def save_logicpath_explanations(model, loader, device: torch.device, output_dir: Path) -> None:
    """Save one graph per AX-b subtype plus transformation-rich examples."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_logic = set()
    transformation_dir = output_dir / "transformation_examples"
    transformation_manifest = []
    manifest = []
    for batch in loader:
        batch = move_batch(batch, device)
        batch["return_logicpath_attention"] = True
        output = model(batch)
        if output.logicpath_node_importance is None or output.logicpath_graph_attention is None:
            raise RuntimeError("LogicPath model did not return graph attention for explanation.")
        for item_index, logic in enumerate(batch["logic"]):
            record = _logicpath_explanation_record(batch, output, item_index)
            filename = _logic_filename(logic)
            if logic not in seen_logic:
                seen_logic.add(logic)
                path = output_dir / filename
                path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                writer = _write_graph_rewrite_svg if record.get("rewrite") else _write_logicpath_graph_svg
                writer(path.with_suffix(".svg"), record)
                manifest.append({"logic": logic, "uid": record["uid"], "file": f"{filename}.svg"})
            active_transformations = [name for name, values in record["transformations"].items() if values["target"]]
            if active_transformations and len(transformation_manifest) < 5:
                transformation_dir.mkdir(parents=True, exist_ok=True)
                name = f"transformation_{len(transformation_manifest) + 1:02d}_{filename}"
                path = transformation_dir / name
                path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                writer = _write_graph_rewrite_svg if record.get("rewrite") else _write_logicpath_graph_svg
                writer(path.with_suffix(".svg"), record)
                transformation_manifest.append({"logic": logic, "transformations": active_transformations, "uid": record["uid"], "file": f"{name}.svg"})
    (output_dir / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if transformation_manifest:
        (transformation_dir / "index.json").write_text(json.dumps(transformation_manifest, indent=2), encoding="utf-8")


def train_one(model_name: str, variant: str, seed: int, train_rows, val_rows, eval_sets: dict, args) -> dict:
    set_seed(seed)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    include_piig = variant in {"piig_v1", "piig_v2", "piig_v3", "piig_v4", "piig_v5", "piig_v6", "piig_pt_lkb_v1"}
    include_logicpath = variant in {
        "logicpath_v1", "logicpath_v2", "logicpath_pt_lkb_v1", "graph_rewrite_v1",
        "logicpath_ablation_states", "logicpath_ablation_transformations", "logicpath_no_transform_v1",
    }
    if variant == "graph_rewrite_v1":
        logicpath_version = "v3"
    elif variant == "logicpath_pt_lkb_v1":
        logicpath_version = "v2"
    elif variant in {"logicpath_ablation_states", "logicpath_ablation_transformations"}:
        logicpath_version = variant.removeprefix("logicpath_")
    elif variant == "logicpath_no_transform_v1":
        logicpath_version = "no_transform_wiktionary_v1"
    else:
        logicpath_version = variant.removeprefix("logicpath_") if include_logicpath else "v1"
    logicpath_lexicon = "PT-LKB" if variant == "logicpath_pt_lkb_v1" else "Wiktionary"
    include_tokgraph = variant in TOKGRAPH_VARIANTS
    piig_version = "v4" if variant == "piig_pt_lkb_v1" else (variant.removeprefix("piig_") if include_piig else "v1")
    original_wiktionary_cache = args.wiktionary_cache
    if variant in {"piig_pt_lkb_v1", "logicpath_pt_lkb_v1"} and getattr(args, "pt_lkb_cache", None):
        args.wiktionary_cache = args.pt_lkb_cache
    train_loader = make_loader(train_rows, tokenizer, args, shuffle=True, include_piig=include_piig, piig_version=piig_version, include_logicpath=include_logicpath, logicpath_version=logicpath_version, logicpath_lexicon=logicpath_lexicon, include_tokgraph=include_tokgraph)
    val_loader = make_loader(val_rows, tokenizer, args, shuffle=False, include_piig=include_piig, piig_version=piig_version, include_logicpath=include_logicpath, logicpath_version=logicpath_version, logicpath_lexicon=logicpath_lexicon, include_tokgraph=include_tokgraph)
    eval_loaders = {name: make_loader(rows, tokenizer, args, shuffle=False, include_piig=include_piig, piig_version=piig_version, include_logicpath=include_logicpath, logicpath_version=logicpath_version, logicpath_lexicon=logicpath_lexicon, include_tokgraph=include_tokgraph) for name, rows in eval_sets.items()}
    args.wiktionary_cache = original_wiktionary_cache

    model = build_model(
        variant=variant,
        model_name=model_name,
        dropout=args.dropout,
        graph_layers=args.graph_layers,
        aux_weight=args.aux_weight,
    ).to(device)

    if hasattr(model, "set_partial_label_datasets"):
        model.set_partial_label_datasets(getattr(args, "partial_label_datasets", ""))

    out_dir = Path(args.output_dir) / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "optimizer_groups"):
        parameters = model.optimizer_groups(args.lr, args.weight_decay, float(getattr(args, "graph_lr_multiplier", 1.0)))
    else:
        parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best = {"metric": -1.0, "state": None, "epoch": 0}
    history = []
    epochs_without_improvement = 0
    stopped_epoch = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"{variant}/seed{seed}/epoch{epoch}", leave=False):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = compute_loss(output, batch["labels_3"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach().cpu())

        val_metrics, _ = evaluate(model, val_loader, device)
        score = val_metrics.get(args.selection_metric)
        if score is None:
            score = val_metrics.get("nli3/macro_f1", val_metrics.get("nli3/accuracy", 0.0))
        history.append({"epoch": epoch, "train_loss": running / max(len(train_loader), 1), "val": val_metrics})
        if score > best["metric"]:
            best = {
                "metric": score,
                "epoch": epoch,
                "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                stopped_epoch = epoch
        if stopped_epoch:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    eval_results = {}
    surrogate_report = None
    for name, loader in eval_loaders.items():
        metrics, predictions = evaluate(model, loader, device)
        eval_results[name] = metrics
        save_predictions(out_dir / f"predictions_{name}.csv", predictions)
        if include_piig and args.save_piig_explanations and name == args.piig_explain_split:
            save_piig_explanations(model, loader, device, out_dir / "explanations" / name)
        if include_logicpath and args.save_piig_explanations and name == args.piig_explain_split:
            save_logicpath_explanations(model, loader, device, out_dir / "explanations" / name)
        if (
            getattr(args, "surrogate_analysis", False)
            and variant == "logicpath_v2"
            and name == getattr(args, "surrogate_split", "inferbr_test")
        ):
            surrogate_report = save_surrogate_analysis(model, loader, device, out_dir / "surrogate")

    result = {
        "variant": variant,
        "seed": seed,
        "model_name": model_name,
        "best_epoch": best["epoch"],
        "selection_metric": args.selection_metric,
        "best_validation_metric": best["metric"],
        "early_stopped": stopped_epoch > 0,
        "stopped_epoch": stopped_epoch,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "eval": eval_results,
        "surrogate_analysis": surrogate_report,
        "args": vars(args),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def maybe_limit(rows, limit: int | None, seed: int):
    rows = list(rows)
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows

    rng = random.Random(seed)
    by_label: Dict[int, List] = {}
    for row in rows:
        by_label.setdefault(row.label_3, []).append(row)
    for label_rows in by_label.values():
        rng.shuffle(label_rows)

    # Preserve the NLI label mix when a faster training subset is requested.
    selected = []
    labels = sorted(by_label)
    base, remainder = divmod(limit, len(labels))
    for index, label in enumerate(labels):
        quota = base + (1 if index < remainder else 0)
        selected.extend(by_label[label][:quota])
    rng.shuffle(selected)
    return selected


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_eval_languages(value: str) -> List[str]:
    languages = parse_csv(value)
    if languages == ["all"]:
        return list(XNLI_LANGUAGES)
    invalid = sorted(set(languages) - set(XNLI_LANGUAGES))
    if invalid:
        raise ValueError(f"Unsupported XNLI evaluation languages: {', '.join(invalid)}")
    return languages


def create_incremental_experiment_dir(base_dir: Path, prefix: str = "experiment") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 100000):
        candidate = base_dir / f"{prefix}_{index:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a new experiment directory under {base_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Project data directory.")
    parser.add_argument("--output-dir", default="runs", help="Base directory where experiment_XXX folders are created.")
    parser.add_argument("--experiment-prefix", default="experiment", help="Prefix for incremental experiment folders.")
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Existing experiment directory to continue. Completed model/seed results are skipped.",
    )
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--models", default="pair_cls,bi_encoder,kbert_injection,kganet_rel_gat,kgnli_dual,dige_v1,dige_v2,dige_v3,dige_v4,dige_v5,dige_v6,dige_v7,dige_v8,dige_v9,piig_v1,piig_v2,piig_v3,piig_v4,piig_v6,piig_pt_lkb_v1,logicpath_v1,logicpath_v2,logicpath_pt_lkb_v1,graph_rewrite_v1")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--train-languages", default="en", help="Used only by the legacy XNLI-dev training protocol.")
    parser.add_argument(
        "--eval-languages",
        default="all",
        help="Comma-separated XNLI test languages, or all for the 15 official XNLI languages.",
    )
    parser.add_argument(
        "--training-protocol",
        choices=["mnli", "xnli_dev_legacy"],
        default="mnli",
        help="mnli: official-style MultiNLI train/dev_matched protocol; xnli_dev_legacy reproduces earlier exploratory runs.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--max-node-len", type=int, default=8)
    parser.add_argument("--max-wikidata-entities", type=int, default=8)
    parser.add_argument("--max-wikidata-len", type=int, default=32)
    parser.add_argument("--max-proof-nodes", type=int, default=12)
    parser.add_argument("--max-proof-len", type=int, default=32)
    parser.add_argument("--max-logicpath-nodes", type=int, default=32)
    parser.add_argument("--max-logicpath-len", type=int, default=24)
    parser.add_argument("--wikidata-cache", default="data/cache/wikidata_entities.jsonl")
    parser.add_argument("--wiktionary-cache", default="data/cache/wiktionary_entries.jsonl")
    parser.add_argument("--pt-lkb-cache", default="data/cache/pt_lkb_relations.jsonl")
    parser.add_argument("--max-wiktionary-entries", type=int, default=2)
    parser.add_argument(
        "--save-piig-explanations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one attention-attribution graph (SVG and JSON) for each PIIG run.",
    )
    parser.add_argument(
        "--piig-explain-split",
        default="axb",
        help="Evaluation split used for the PIIG explanation artifact; axb by default.",
    )
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument(
        "--graph-lr-multiplier",
        type=float,
        default=10.0,
        help="Learning-rate multiplier for newly initialised graph parameters (token-graph variants only).",
    )
    parser.add_argument(
        "--partial-label-datasets",
        default="assin,assin2",
        help="Comma-separated datasets whose 'neutral' label means not-entailment (neutral or contradiction). Used by coverage_pl_v1.",
    )
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--selection-metric", default="binary/mcc", help="Validation metric used for checkpoint selection.")
    parser.add_argument("--patience", type=int, default=0, help="Early stop after this many non-improving validation epochs. 0 disables early stopping.")
    parser.add_argument("--cd-metric", default="binary/mcc", help="Metric used for critical-difference ranking reports.")
    parser.add_argument("--cd-alpha", type=float, default=0.05, help="Alpha level for Friedman/Nemenyi reports.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction for xnli_dev_legacy only.")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=60000,
        help="Stratified training cap; 0 uses all MultiNLI examples.",
    )
    parser.add_argument(
        "--train-manifest",
        default=None,
        help="Fixed JSONL training subset. When set, train-limit and seed-based sampling are bypassed.",
    )
    parser.add_argument("--eval-limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--surrogate-analysis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit a grouped surrogate analysis on the final LogicPath feature fusion.",
    )
    parser.add_argument("--surrogate-split", default="inferbr_test")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = project_root()
    output_base_dir = (root / args.output_dir).resolve()
    if args.resume_dir:
        resume_path = Path(args.resume_dir)
        experiment_dir = (resume_path if resume_path.is_absolute() else root / resume_path).resolve()
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Resume experiment directory not found: {experiment_dir}")
        saved_config_path = experiment_dir / "resume_config.json"
        if not saved_config_path.exists():
            saved_config_path = experiment_dir / "run_config.json"
        if saved_config_path.exists():
            saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
            provided_flags = set(sys.argv[1:])
            for key, value in saved_config.items():
                flag = f"--{key.replace('_', '-')}"
                if key in {"output_dir", "output_base_dir", "resume_dir"} or flag in provided_flags:
                    continue
                if hasattr(args, key):
                    setattr(args, key, value)
            print(f"loaded resume configuration: {saved_config_path}")
        print(f"resuming experiment_dir: {experiment_dir}")
    else:
        experiment_dir = create_incremental_experiment_dir(output_base_dir, args.experiment_prefix)
    data_root = root / args.data_dir
    output_base_dir = (root / args.output_dir).resolve()
    args.output_base_dir = str(output_base_dir)
    args.output_dir = str(experiment_dir)
    config_name = "resume_config.json" if args.resume_dir else "run_config.json"
    (experiment_dir / config_name).write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"experiment_dir: {experiment_dir}")

    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    variants = parse_csv(args.models)
    train_languages = parse_csv(args.train_languages)
    eval_languages = resolve_eval_languages(args.eval_languages)

    all_results = []
    for seed in seeds:
        if args.training_protocol == "mnli":
            if args.train_manifest:
                manifest_path = Path(args.train_manifest)
                if not manifest_path.is_absolute():
                    manifest_path = root / manifest_path
                train_rows = load_nli_manifest(manifest_path)
            else:
                train_rows = maybe_limit(load_mnli(data_root, "train"), args.train_limit, seed)
            val_rows = load_mnli(data_root, "dev_matched")
        else:
            train_source = load_xnli(data_root, "dev", train_languages)
            train_source = maybe_limit(train_source, args.train_limit, seed)
            train_rows, val_rows = split_train_validation(train_source, args.val_fraction, seed)

        eval_sets = {}
        for language in eval_languages:
            rows = load_xnli(data_root, "test", [language])
            eval_sets[f"xnli_test_{language}"] = maybe_limit(rows, args.eval_limit, seed)
        eval_sets["axb"] = maybe_limit(load_axb(data_root), args.eval_limit, seed)

        for variant in variants:
            result_path = experiment_dir / variant / f"seed_{seed}" / "result.json"
            if result_path.exists():
                print(f"Skipping completed {variant}/seed{seed}: {result_path}")
                all_results.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            result = train_one(args.model_name, variant, seed, train_rows, val_rows, eval_sets, args)
            all_results.append(result)

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    cd_dir = Path(args.output_dir) / "critical_difference"
    reports = generate_critical_difference_reports(summary_path, cd_dir, args.cd_metric, args.cd_alpha)
    print(f"wrote {cd_dir} ({len(reports)} split reports)")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
