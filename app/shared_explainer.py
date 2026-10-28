"""Shared GNNExplainer helpers for node-classification models.

Models must implement forward(x, edge_index) and return raw class logits.
GCN, GraphSAGE, and GAT fit that contract. Probability ensembles can be wrapped
with ProbabilityEnsembleAdapter. Checkpoint loading remains the app's job.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch_geometric.explain import Explainer, GNNExplainer


def create_explainer(
    model: nn.Module,
    *,
    epochs: int = 100,
    node_mask_type: str = "attributes",
    edge_mask_type: str = "object",
) -> Explainer:
    """Create a PyG explainer for a node-classification model returning logits."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if node_mask_type not in {"attributes", "common_attributes", "object", None}:
        raise ValueError("Unsupported node_mask_type.")
    if edge_mask_type not in {"object", "attributes", None}:
        raise ValueError("Unsupported edge_mask_type.")
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )


def explain_node(explainer: Explainer, graph_data: Any, node_index: int):
    """Explain one node after validating its global node index."""
    node_index = int(node_index)
    num_nodes = int(graph_data.x.shape[0])
    if not 0 <= node_index < num_nodes:
        raise ValueError(f"node_index must be between 0 and {num_nodes - 1}.")
    explainer.model.eval()
    return explainer(graph_data.x, graph_data.edge_index, index=node_index)


def _feature_mask_for_node(
    node_mask: torch.Tensor,
    node_index: int,
    *,
    num_nodes: int | None = None,
) -> torch.Tensor:
    """Resolve full-graph, shared-attribute, and single-row masks safely."""
    if node_mask is None:
        raise ValueError("The explanation does not contain a node feature mask.")
    mask = node_mask.detach()
    if mask.dim() == 1:
        return mask
    if mask.dim() != 2:
        raise ValueError(f"Unexpected node mask shape: {tuple(mask.shape)}")
    rows = int(mask.shape[0])
    if rows == 1:
        return mask[0]
    if num_nodes is not None and rows != int(num_nodes):
        raise ValueError(
            f"Cannot safely map a {rows}-row node mask to a graph with "
            f"{num_nodes} nodes. Expected one shared row or one row per node."
        )
    if not 0 <= int(node_index) < rows:
        raise ValueError(f"node_index {node_index} is outside the {rows}-row mask.")
    return mask[int(node_index)]


def get_top_features(
    explanation: Any,
    node_index: int,
    top_k: int = 10,
    *,
    num_nodes: int | None = None,
) -> list[dict[str, float | int]]:
    """Return the highest-scoring feature-mask entries for one explained node."""
    if top_k < 1:
        return []
    node_mask = getattr(explanation, "node_mask", None)
    if node_mask is None:
        return []
    scores = _feature_mask_for_node(
        node_mask, int(node_index), num_nodes=num_nodes
    ).reshape(-1).cpu()
    k = min(int(top_k), scores.numel())
    values, indices = torch.topk(scores, k=k)
    return [
        {"feature": int(index), "importance": float(value)}
        for index, value in zip(indices.tolist(), values.tolist())
    ]


def get_top_edges(
    explanation: Any, edge_index: torch.Tensor, top_k: int = 10
) -> list[dict[str, float | int]]:
    """Return the highest-scoring edges from the explanation edge mask."""
    edge_mask = getattr(explanation, "edge_mask", None)
    if edge_mask is None or top_k < 1:
        return []
    scores = edge_mask.detach().reshape(-1).cpu()
    edges = edge_index.detach().cpu()
    if scores.numel() != edges.shape[1]:
        raise ValueError(
            f"Edge-mask length ({scores.numel()}) does not match the number "
            f"of input edges ({edges.shape[1]})."
        )
    k = min(int(top_k), scores.numel())
    values, indices = torch.topk(scores, k=k)
    return [
        {
            "source": int(edges[0, edge_idx]),
            "target": int(edges[1, edge_idx]),
            "importance": float(value),
        }
        for value, edge_idx in zip(values.tolist(), indices.tolist())
    ]


def get_important_neighbors(
    node_index: int,
    top_edges: Sequence[dict[str, Any]],
    graph_data: Any,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Aggregate important edges into unique direct neighbors."""
    if top_k < 1:
        return []
    neighbors: dict[int, dict[str, Any]] = {}
    for edge in top_edges:
        source, target = int(edge["source"]), int(edge["target"])
        if source == node_index:
            neighbor = target
        elif target == node_index:
            neighbor = source
        else:
            continue
        importance = float(edge["importance"])
        if neighbor in neighbors:
            neighbors[neighbor]["importance"] += importance
            continue
        label_name = None
        if hasattr(graph_data, "y"):
            raw_label = int(graph_data.y[neighbor].item())
            label_name = {1: "Illicit", 0: "Licit"}.get(raw_label, "Unknown")
        neighbors[neighbor] = {
            "node": neighbor, "importance": importance, "label": label_name
        }
    return sorted(
        neighbors.values(), key=lambda item: item["importance"], reverse=True
    )[:top_k]


def get_visualization_edges(
    explanation: Any,
    edge_index: torch.Tensor,
    selected_node: int,
    direct_k: int = 5,
    additional_k: int = 10,
) -> list[dict[str, float | int]]:
    """Prefer important direct edges, then edges touching their neighbors."""
    if direct_k < 0 or additional_k < 0:
        raise ValueError("direct_k and additional_k must be non-negative.")
    edge_mask = getattr(explanation, "edge_mask", None)
    if edge_mask is None:
        return []
    scores, edges = edge_mask.detach().reshape(-1).cpu(), edge_index.detach().cpu()
    if scores.numel() != edges.shape[1]:
        raise ValueError("Edge-mask length does not match edge_index.")
    direct, other = [], []
    for idx, score in enumerate(scores.tolist()):
        source, target = int(edges[0, idx]), int(edges[1, idx])
        edge = {"source": source, "target": target, "importance": float(score)}
        (direct if selected_node in (source, target) else other).append(edge)
    direct.sort(key=lambda item: item["importance"], reverse=True)
    other.sort(key=lambda item: item["importance"], reverse=True)
    selected_direct = direct[:direct_k]
    neighbors = {
        edge["target"] if edge["source"] == selected_node else edge["source"]
        for edge in selected_direct
    }
    second_hop = [
        edge for edge in other
        if edge["source"] in neighbors or edge["target"] in neighbors
    ]
    unrelated = [
        edge for edge in other
        if edge["source"] not in neighbors and edge["target"] not in neighbors
    ]
    return selected_direct + (second_hop + unrelated)[:additional_k]


def summarize_local_explanations(
    explanations: Sequence[Any],
    node_indices: Sequence[int],
    graph_data: Any,
    top_k: int = 10,
) -> dict[str, Any]:
    """Aggregate local masks for a sample-based global summary.

    This is not a single global explanation. Use a documented, reproducible node
    sample; explaining every node in a graph this large can be prohibitively costly.
    """
    if len(explanations) != len(node_indices):
        raise ValueError("explanations and node_indices must have equal lengths.")
    if not explanations:
        raise ValueError("At least one local explanation is required.")
    feature_rows, edge_rows = [], []
    for explanation, node_index in zip(explanations, node_indices):
        node_mask = getattr(explanation, "node_mask", None)
        if node_mask is not None:
            feature_rows.append(_feature_mask_for_node(
                node_mask, int(node_index), num_nodes=int(graph_data.x.shape[0])
            ).detach().float().cpu())
        edge_mask = getattr(explanation, "edge_mask", None)
        if edge_mask is not None:
            row = edge_mask.detach().reshape(-1).float().cpu()
            if row.numel() != graph_data.edge_index.shape[1]:
                raise ValueError("An explanation edge mask does not match edge_index.")
            edge_rows.append(row)

    result: dict[str, Any] = {"num_explanations": len(explanations)}
    if feature_rows:
        mean_features = torch.stack(feature_rows).mean(dim=0)
        k = min(max(int(top_k), 0), mean_features.numel())
        values, indices = torch.topk(mean_features, k=k)
        result["mean_feature_importance"] = mean_features
        result["top_features"] = [
            {"feature": int(i), "importance": float(v)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]
    else:
        result.update(mean_feature_importance=None, top_features=[])
    if edge_rows:
        mean_edges = torch.stack(edge_rows).mean(dim=0)
        k = min(max(int(top_k), 0), mean_edges.numel())
        values, indices = torch.topk(mean_edges, k=k)
        edges = graph_data.edge_index.detach().cpu()
        result["mean_edge_importance"] = mean_edges
        result["top_edges"] = [
            {"source": int(edges[0, i]), "target": int(edges[1, i]), "importance": float(v)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]
    else:
        result.update(mean_edge_importance=None, top_edges=[])
    return result


class ProbabilityEnsembleAdapter(nn.Module):
    """Adapt weighted binary probability ensembles to a differentiable logit model.

    Component models must return raw logits. Weights must be non-negative and sum
    to one. The returned logits' argmax follows the ensemble's fraud-probability
    threshold, including thresholds other than 0.5.
    """
    def __init__(
        self,
        models: Sequence[nn.Module],
        weights: Sequence[float],
        threshold: float = 0.5,
        positive_class: int = 1,
    ) -> None:
        super().__init__()
        if not models or len(models) != len(weights):
            raise ValueError("Provide the same non-zero number of models and weights.")
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if not torch.isfinite(weight_tensor).all() or (weight_tensor < 0).any():
            raise ValueError("Ensemble weights must be finite and non-negative.")
        if not torch.isclose(weight_tensor.sum(), torch.tensor(1.0), atol=1e-5):
            raise ValueError("Ensemble weights must sum to 1.")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1.")
        if positive_class not in (0, 1):
            raise ValueError("positive_class must be 0 or 1.")
        self.models = nn.ModuleList(models)
        self.register_buffer("weights", weight_tensor)
        self.threshold = float(threshold)
        self.positive_class = int(positive_class)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        probs = []
        for model in self.models:
            model.eval()
            logits = model(x, edge_index)
            if logits.dim() != 2 or logits.shape[1] != 2:
                raise ValueError("Each ensemble model must return [num_nodes, 2] logits.")
            probs.append(torch.softmax(logits, dim=-1)[:, self.positive_class])
        stacked = torch.stack(probs, dim=0)
        weights = self.weights.to(device=stacked.device, dtype=stacked.dtype)
        positive_probability = (weights[:, None] * stacked).sum(dim=0)
        eps = torch.finfo(positive_probability.dtype).eps
        positive_probability = positive_probability.clamp(eps, 1.0 - eps)
        threshold = positive_probability.new_tensor(self.threshold)
        threshold_log_odds = torch.log(threshold) - torch.log1p(-threshold)
        probability_log_odds = torch.log(positive_probability) - torch.log1p(-positive_probability)
        positive_logit = probability_log_odds - threshold_log_odds
        if self.positive_class == 1:
            return torch.stack((torch.zeros_like(positive_logit), positive_logit), dim=-1)
        return torch.stack((positive_logit, torch.zeros_like(positive_logit)), dim=-1)
