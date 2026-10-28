"""Read-only accessors for recorded evaluation metrics and result artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRIC_FILES = {
    "GCN": Path("results/GCN/metrics.json"),
    "GraphSAGE": Path("results/GraphSAGE/metrics.json"),
    "GAT": Path("results/GAT/metrics.json"),
    "Ensemble": Path("results/Ensemble/ensemble_metrics.json"),
}

ARTIFACTS = {
    "GCN": {
        "Confusion matrix": "results/GCN/confusion_matrix.png",
        "Precision–recall curve": "results/GCN/precision_recall_curve.png",
        "ROC curve": "results/GCN/roc_curve.png",
        "Training loss": "results/GCN/training_loss.png",
        "Validation fraud F1": "results/GCN/validation_f1.png",
    },
    "GraphSAGE": {
        "Confusion matrix": "results/GraphSAGE/confusion_matrix.png",
        "Precision–recall curve": "results/GraphSAGE/precision_recall_curve.png",
        "ROC curve": "results/GraphSAGE/roc_curve.png",
        "Training loss": "results/GraphSAGE/training_loss.png",
        "Validation fraud F1": "results/GraphSAGE/validation_f1.png",
        "Validation PR-AUC": "results/GraphSAGE/validation_pr_auc.png",
    },
    "GAT": {
        "Confusion matrix": "results/GAT/confusion_matrix.png",
        "Precision–recall curve": "results/GAT/precision_recall_curve.png",
        "ROC curve": "results/GAT/roc_curve.png",
        "Training loss": "results/GAT/training_loss.png",
        "Validation fraud F1": "results/GAT/validation_f1.png",
        "Validation PR-AUC": "results/GAT/validation_pr_auc.png",
    },
    "Ensemble": {
        "Confusion matrix": "results/Ensemble/confusion_matrix.png",
        "Precision–recall curve": "results/Ensemble/precision_recall_curve.png",
        "ROC curve": "results/Ensemble/roc_curve.png",
        "Validation F1 heatmap": "results/Ensemble/validation_f1_heatmap.png",
    },
}


def load_recorded_metrics(project_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load only metrics that exist in tracked/local artifacts; never synthesize scores."""
    project_root = Path(project_root)
    metrics: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for key, relative_path in METRIC_FILES.items():
        path = project_root / relative_path
        if not path.is_file():
            errors[key] = f"Metrics file not found: {relative_path.as_posix()}"
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics[key] = _normalize_metrics(key, data)
        except Exception as exc:
            errors[key] = f"Could not read {relative_path.as_posix()}: {type(exc).__name__}: {exc}"
    return metrics, errors


def _normalize_metrics(key: str, data: dict[str, Any]) -> dict[str, Any]:
    raw = data.get("test_metrics", data)
    aliases = {
        "accuracy": ("accuracy",),
        "fraud_precision": ("fraud_precision", "precision"),
        "fraud_recall": ("fraud_recall", "recall"),
        "fraud_f1": ("fraud_f1", "f1", "f1_score"),
        "pr_auc_average_precision": ("pr_auc_average_precision", "pr_auc", "average_precision"),
        "roc_auc": ("roc_auc",),
    }
    normalized: dict[str, Any] = {
        "model": data.get("model", key),
        "source": METRIC_FILES[key].as_posix(),
        "selection_split": data.get("selection_split", "test metrics recorded in experiment artifact"),
        "test_support": data.get("test_support"),
        "confusion_matrix": data.get("confusion_matrix"),
        "confusion_matrix_labels": data.get("confusion_matrix_labels", ["Legitimate", "Fraud"]),
        "classification_report": data.get("classification_report"),
        "threshold": data.get("threshold"),
        "alpha_graphsage": data.get("alpha_graphsage"),
        "alpha_gat": data.get("alpha_gat"),
    }
    for target, candidates in aliases.items():
        value = next((raw[name] for name in candidates if name in raw), None)
        normalized[target] = float(value) if value is not None else None
    return normalized


def available_artifacts(project_root: Path, model_key: str) -> list[tuple[str, Path]]:
    """Return existing result images for one model, preserving declared order."""
    root = Path(project_root)
    return [
        (label, root / relative)
        for label, relative in ARTIFACTS.get(model_key, {}).items()
        if (root / relative).is_file()
    ]
