import torch

from app.model_registry import GAT, GCN, GraphSAGE, _build_model


def _toy_graph():
    x = torch.randn(6, 5)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 0, 2], [1, 2, 3, 4, 5, 0, 2, 4]],
        dtype=torch.long,
    )
    return x, edge_index


def test_gcn_checkpoint_architecture_round_trip():
    x, edge_index = _toy_graph()
    original = GCN(num_features=5, hidden_channels=8, num_classes=2)
    loaded = _build_model("GCN", original.state_dict(), num_features=5)
    loaded.eval()
    with torch.no_grad():
        output = loaded(x, edge_index)
    assert output.shape == (6, 2)


def test_graphsage_checkpoint_architecture_round_trip():
    x, edge_index = _toy_graph()
    original = GraphSAGE(num_features=5, hidden_channels=8, num_classes=2)
    loaded = _build_model("GraphSAGE", original.state_dict(), num_features=5)
    loaded.eval()
    with torch.no_grad():
        output = loaded(x, edge_index)
    assert output.shape == (6, 2)


def test_gat_checkpoint_architecture_round_trip():
    x, edge_index = _toy_graph()
    original = GAT(num_features=5, hidden_channels=3, num_heads=2, num_classes=2)
    loaded = _build_model("GAT", original.state_dict(), num_features=5)
    loaded.eval()
    with torch.no_grad():
        output = loaded(x, edge_index)
    assert output.shape == (6, 2)


def test_model_loader_rejects_feature_mismatch():
    original = GCN(num_features=5, hidden_channels=8, num_classes=2)
    try:
        _build_model("GCN", original.state_dict(), num_features=4)
    except ValueError as exc:
        assert "expects 5 features" in str(exc)
    else:
        raise AssertionError("Feature mismatch should be rejected.")
