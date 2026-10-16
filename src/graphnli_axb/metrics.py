from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_recall_fscore_support

from .labels import LABELS_3, nli3_to_binary_id


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(0.1, 1.0, n_bins)):
        mask = (conf > low) & (conf <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_multiclass(probs: np.ndarray, labels: np.ndarray, num_labels: int) -> float:
    target = np.zeros((labels.shape[0], num_labels), dtype=np.float32)
    target[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def classification_metrics(logits: np.ndarray, labels: np.ndarray, prefix: str = "") -> Dict[str, float]:
    valid = labels >= 0
    if not valid.any():
        return {}
    logits = logits[valid]
    labels = labels[valid]
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, pred, average="macro", zero_division=0
    )
    return {
        f"{prefix}accuracy": float(accuracy_score(labels, pred)),
        f"{prefix}macro_f1": float(macro_f1),
        f"{prefix}weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        f"{prefix}precision_macro": float(precision),
        f"{prefix}recall_macro": float(recall),
        f"{prefix}mcc": float(matthews_corrcoef(labels, pred)) if len(set(labels.tolist())) > 1 else 0.0,
        f"{prefix}ece": expected_calibration_error(probs, labels),
        f"{prefix}brier": brier_multiclass(probs, labels, probs.shape[1]),
    }


def binary_axb_metrics(logits: np.ndarray, labels_binary: np.ndarray, prefix: str = "binary_") -> Dict[str, float]:
    valid = labels_binary >= 0
    if not valid.any():
        return {}
    logits = logits[valid]
    labels = labels_binary[valid]
    probs3 = softmax(logits)
    pred3 = probs3.argmax(axis=1)
    pred = np.array([nli3_to_binary_id(x) for x in pred3], dtype=np.int64)
    probs = np.stack([probs3[:, 0], probs3[:, 1:].sum(axis=1)], axis=1)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, pred, average="macro", zero_division=0
    )
    return {
        f"{prefix}accuracy": float(accuracy_score(labels, pred)),
        f"{prefix}macro_f1": float(macro_f1),
        f"{prefix}weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        f"{prefix}precision_macro": float(precision),
        f"{prefix}recall_macro": float(recall),
        f"{prefix}mcc": float(matthews_corrcoef(labels, pred)) if len(set(labels.tolist())) > 1 else 0.0,
        f"{prefix}ece": expected_calibration_error(probs, labels),
        f"{prefix}brier": brier_multiclass(probs, labels, 2),
    }


def direct_binary_metrics(logits: np.ndarray, labels_binary: np.ndarray, prefix: str = "binary_") -> Dict[str, float]:
    valid = labels_binary >= 0
    if not valid.any():
        return {}
    logits = logits[valid]
    labels = labels_binary[valid]
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, pred, average="macro", zero_division=0
    )
    return {
        f"{prefix}accuracy": float(accuracy_score(labels, pred)),
        f"{prefix}macro_f1": float(macro_f1),
        f"{prefix}weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        f"{prefix}precision_macro": float(precision),
        f"{prefix}recall_macro": float(recall),
        f"{prefix}mcc": float(matthews_corrcoef(labels, pred)) if len(set(labels.tolist())) > 1 else 0.0,
        f"{prefix}ece": expected_calibration_error(probs, labels),
        f"{prefix}brier": brier_multiclass(probs, labels, 2),
    }


def grouped_binary_metrics(logits: np.ndarray, labels_binary: np.ndarray, groups: List[str]) -> Dict[str, float]:
    out = {}
    by_group: Dict[str, List[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        by_group[group].append(idx)
    for group, indices in by_group.items():
        if len(indices) < 2:
            continue
        metrics = binary_axb_metrics(logits[indices], labels_binary[indices], prefix=f"logic/{group}/")
        out.update(metrics)
    return out


def grouped_direct_binary_metrics(logits: np.ndarray, labels_binary: np.ndarray, groups: List[str]) -> Dict[str, float]:
    out = {}
    by_group: Dict[str, List[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        by_group[group].append(idx)
    for group, indices in by_group.items():
        if len(indices) < 2:
            continue
        metrics = direct_binary_metrics(logits[indices], labels_binary[indices], prefix=f"logic/{group}/")
        out.update(metrics)
    return out
