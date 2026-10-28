"""Model loading and inference helpers for the GraphShield Streamlit frontend.

Checkpoint files are intentionally local/ignored. Missing checkpoints are reported
per model so the rest of the app (including recorded metrics) can still load.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

try:
    from .explainer_utils import ProbabilityEnsembleAdapter
except ImportError:  # Streamlit launched from app/
    from explainer_utils import ProbabilityEnsembleAdapter


class GCN(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int = 64, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GAT(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 16,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.feature_dropout = dropout
        self.conv1 = GATConv(
            num_features, hidden_channels, heads=num_heads, concat=True, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels * num_heads, num_classes, heads=1, concat=False, dropout=dropout
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.feature_dropout, training=self.training)
        return self.conv2(x, edge_index)


@dataclass
class ModelBundle:
    key: str
    label: str
    model: nn.Module
    checkpoint_path: Path
    threshold: float = 0.5
    ensemble_components: tuple[nn.Module, ...] = ()
    ensemble_weights: tuple[float, ...] = ()


MODEL_SPECS = {
    "GCN": {"label": "Graph Convolutional Network", "checkpoint": "best_gcn_model.pt"},
    "GraphSAGE": {"label": "GraphSAGE", "checkpoint": "best_graphsage_model.pt"},
    "GAT": {"label": "Graph Attention Network", "checkpoint": "best_gat_model.pt"},
}


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {path}. Expected a state_dict.")
    return state


def _build_model(key: str, state: dict[str, torch.Tensor], num_features: int) -> nn.Module:
    if key == "GCN":
        first = state.get("conv1.lin.weight")
        second = state.get("conv2.lin.weight")
        if first is None or second is None:
            raise ValueError("GCN checkpoint is missing conv1/conv2 linear weights.")
        if int(first.shape[1]) != num_features:
            raise ValueError(f"GCN checkpoint expects {first.shape[1]} features; graph has {num_features}.")
        model = GCN(num_features, int(first.shape[0]), int(second.shape[0]))
    elif key == "GraphSAGE":
        first = state.get("conv1.lin_l.weight")
        second = state.get("conv2.lin_l.weight")
        if first is None or second is None:
            raise ValueError("GraphSAGE checkpoint is missing conv1/conv2 lin_l weights.")
        if int(first.shape[1]) != num_features:
            raise ValueError(f"GraphSAGE checkpoint expects {first.shape[1]} features; graph has {num_features}.")
        model = GraphSAGE(num_features, int(first.shape[0]), int(second.shape[0]), dropout=0.5)
    elif key == "GAT":
        first = state.get("conv1.lin.weight")
        second = state.get("conv2.lin.weight")
        att_src = state.get("conv1.att_src")
        if first is None or second is None or att_src is None:
            raise ValueError("GAT checkpoint is missing expected convolution/attention weights.")
        if int(first.shape[1]) != num_features:
            raise ValueError(f"GAT checkpoint expects {first.shape[1]} features; graph has {num_features}.")
        heads = int(att_src.shape[1]) if att_src.dim() >= 3 else 4
        hidden = int(att_src.shape[2]) if att_src.dim() >= 3 else int(first.shape[0] // heads)
        model = GAT(num_features, hidden, heads, int(second.shape[0]), dropout=0.5)
    else:
        raise KeyError(f"Unknown model key: {key}")

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_model_registry(
    graph_data: Any,
    project_root: Path,
    ensemble_config_path: Path | None = None,
) -> tuple[dict[str, ModelBundle], dict[str, str]]:
    """Load each available checkpoint independently and return per-model errors."""
    model_dir = Path(project_root) / "model"
    feature_count = int(graph_data.x.shape[1])
    bundles: dict[str, ModelBundle] = {}
    errors: dict[str, str] = {}

    for key, spec in MODEL_SPECS.items():
        checkpoint = model_dir / spec["checkpoint"]
        if not checkpoint.is_file():
            errors[key] = f"Checkpoint not found: {checkpoint}"
            continue
        try:
            state = _load_state(checkpoint)
            model = _build_model(key, state, feature_count)
            bundles[key] = ModelBundle(key, spec["label"], model, checkpoint)
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"

    # Ensemble settings come from the recorded validation-selected experiment.
    # Do not invent a threshold or blend if the artifact is unavailable.
    config_path = ensemble_config_path or (Path(project_root) / "results" / "Ensemble" / "ensemble_metrics.json")
    if "GraphSAGE" in bundles and "GAT" in bundles and config_path.is_file():
        try:
            import json
            config = json.loads(config_path.read_text(encoding="utf-8"))
            alpha_sage = float(config["alpha_graphsage"])
            alpha_gat = float(config["alpha_gat"])
            threshold = float(config["threshold"])
            if not (0.0 <= alpha_sage <= 1.0 and 0.0 <= alpha_gat <= 1.0):
                raise ValueError("Recorded ensemble weights must be between 0 and 1.")
            if abs(alpha_sage + alpha_gat - 1.0) > 1e-5:
                raise ValueError("Recorded ensemble weights do not sum to 1.")
            if not 0.0 < threshold < 1.0:
                raise ValueError("Recorded ensemble threshold must be between 0 and 1.")
            sage = bundles["GraphSAGE"].model
            gat = bundles["GAT"].model
            adapter = ProbabilityEnsembleAdapter(
                [sage, gat], [alpha_sage, alpha_gat], threshold=threshold, positive_class=1
            )
            adapter.eval()
            bundles["Ensemble"] = ModelBundle(
                "Ensemble",
                "GraphSAGE–GAT probability ensemble",
                adapter,
                config_path,
                threshold=threshold,
                ensemble_components=(sage, gat),
                ensemble_weights=(alpha_sage, alpha_gat),
            )
        except Exception as exc:
            errors["Ensemble"] = f"Could not configure ensemble: {type(exc).__name__}: {exc}"
    else:
        missing = []
        if "GraphSAGE" not in bundles:
            missing.append("GraphSAGE checkpoint")
        if "GAT" not in bundles:
            missing.append("GAT checkpoint")
        if not config_path.is_file():
            missing.append(f"ensemble settings ({config_path})")
        errors["Ensemble"] = "Missing required " + ", ".join(missing) + "."

    return bundles, errors


@torch.no_grad()
def predict_graph(bundle: ModelBundle, graph_data: Any) -> dict[str, Any]:
    """Compute graph-wide probabilities once; callers should cache this result."""
    x, edge_index = graph_data.x, graph_data.edge_index
    bundle.model.eval()

    if bundle.key == "Ensemble":
        component_probs = []
        for model in bundle.ensemble_components:
            model.eval()
            component_probs.append(torch.softmax(model(x, edge_index), dim=1)[:, 1])
        weights = torch.as_tensor(bundle.ensemble_weights, dtype=component_probs[0].dtype)
        fraud_probability = sum(weights[i] * p for i, p in enumerate(component_probs))
        predictions = (fraud_probability >= bundle.threshold).long()
        probabilities = torch.stack((1.0 - fraud_probability, fraud_probability), dim=1)
    else:
        logits = bundle.model(x, edge_index)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        fraud_probability = probabilities[:, 1]

    return {
        "probabilities": probabilities.detach().cpu(),
        "predictions": predictions.detach().cpu(),
        "fraud_probability": fraud_probability.detach().cpu(),
    }
