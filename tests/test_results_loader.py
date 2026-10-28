import json

from app.results_loader import load_recorded_metrics


def test_load_recorded_metrics_normalizes_flat_and_nested_artifacts(tmp_path):
    (tmp_path / "results" / "GCN").mkdir(parents=True)
    (tmp_path / "results" / "GraphSAGE").mkdir(parents=True)
    (tmp_path / "results" / "GCN" / "metrics.json").write_text(
        json.dumps({
            "model": "GCN",
            "accuracy": 0.95,
            "fraud_precision": 0.6,
            "fraud_recall": 0.4,
            "fraud_f1": 0.48,
            "confusion_matrix": [[10, 2], [3, 4]],
        }),
        encoding="utf-8",
    )
    (tmp_path / "results" / "GraphSAGE" / "metrics.json").write_text(
        json.dumps({
            "model": "GraphSAGE",
            "test_metrics": {
                "accuracy": 0.94,
                "fraud_precision": 0.5,
                "fraud_recall": 0.6,
                "fraud_f1": 0.55,
                "pr_auc_average_precision": 0.7,
                "roc_auc": 0.8,
            },
            "split_sizes": {"test": 100},
        }),
        encoding="utf-8",
    )

    metrics, errors = load_recorded_metrics(tmp_path)

    assert metrics["GCN"]["fraud_f1"] == 0.48
    assert metrics["GCN"]["confusion_matrix"] == [[10, 2], [3, 4]]
    assert metrics["GraphSAGE"]["fraud_f1"] == 0.55
    assert metrics["GraphSAGE"]["pr_auc_average_precision"] == 0.7
    assert "GAT" in errors
    assert "Ensemble" in errors


def test_metrics_loader_does_not_fabricate_missing_values(tmp_path):
    (tmp_path / "results" / "GCN").mkdir(parents=True)
    (tmp_path / "results" / "GCN" / "metrics.json").write_text(
        json.dumps({"model": "GCN", "accuracy": 0.91}),
        encoding="utf-8",
    )

    metrics, _ = load_recorded_metrics(tmp_path)

    assert metrics["GCN"]["accuracy"] == 0.91
    assert metrics["GCN"]["fraud_precision"] is None
    assert metrics["GCN"]["fraud_recall"] is None
    assert metrics["GCN"]["fraud_f1"] is None
