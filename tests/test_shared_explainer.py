"""Focused unit tests. Run: python -m pytest tests/test_shared_explainer.py"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from app.shared_explainer import (
    ProbabilityEnsembleAdapter,
    _feature_mask_for_node,
    create_explainer,
    get_top_edges,
    get_top_features,
    summarize_local_explanations,
)


def test_full_graph_feature_mask_selects_requested_node():
    mask = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    assert torch.equal(_feature_mask_for_node(mask, 2, num_nodes=4), torch.tensor([6., 7., 8.]))


def test_common_attribute_mask_uses_single_shared_row():
    mask = torch.tensor([[0.1, 0.9, 0.3]])
    assert torch.equal(_feature_mask_for_node(mask, 3, num_nodes=4), mask[0])


def test_unrecognized_local_mask_shape_fails_loudly():
    with pytest.raises(ValueError, match="Cannot safely map"):
        _feature_mask_for_node(torch.ones(2, 3), 1, num_nodes=4)


def test_get_top_features_supports_common_attributes():
    explanation = SimpleNamespace(node_mask=torch.tensor([[0.2, 0.8, 0.5]]))
    result = get_top_features(explanation, node_index=99, top_k=2, num_nodes=100)
    assert [item["feature"] for item in result] == [1, 2]


def test_get_top_edges_checks_mask_length():
    explanation = SimpleNamespace(edge_mask=torch.tensor([0.1, 0.2]))
    with pytest.raises(ValueError, match="does not match"):
        get_top_edges(explanation, torch.tensor([[0], [1]]))


def test_global_summary_averages_explained_target_features():
    graph = SimpleNamespace(x=torch.zeros((3, 2)), edge_index=torch.tensor([[0, 1], [1, 2]]))
    explanations = [
        SimpleNamespace(
            node_mask=torch.tensor([[.9, .1], [.2, .8], [.1, .2]]),
            edge_mask=torch.tensor([.8, .2]),
        ),
        SimpleNamespace(
            node_mask=torch.tensor([[.9, .1], [.6, .4], [.1, .2]]),
            edge_mask=torch.tensor([.4, .6]),
        ),
    ]
    result = summarize_local_explanations(explanations, [1, 1], graph, top_k=2)
    assert torch.allclose(result["mean_feature_importance"], torch.tensor([.4, .6]))
    assert torch.allclose(result["mean_edge_importance"], torch.tensor([.6, .4]))


class FixedLogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x, edge_index):
        return self.logits.expand(x.shape[0], -1) + x[:, :1] * 0.0


def test_probability_ensemble_adapter_respects_threshold():
    x = torch.zeros((2, 1))
    edge_index = torch.tensor([[0, 1], [1, 0]])
    models = [
        FixedLogitModel([[0.0, 1.3862944]]),  # p ~= .8
        FixedLogitModel([[0.0, 0.4054651]]),  # p ~= .6
    ]
    adapter = ProbabilityEnsembleAdapter(models, weights=[.5, .5], threshold=.75)
    assert adapter(x, edge_index).argmax(dim=1).tolist() == [0, 0]


def test_create_explainer_contract():
    model = FixedLogitModel([[0.0, 1.0]])
    assert create_explainer(model, epochs=2).model is model
