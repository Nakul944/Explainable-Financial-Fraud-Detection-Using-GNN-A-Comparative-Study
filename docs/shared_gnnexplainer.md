# Shared GNNExplainer

This change centralizes explanation logic for any node-classification model whose
forward(x, edge_index) returns raw class logits. The project's GCN, GraphSAGE,
and GAT models use that contract.

## Existing app integration

The existing Streamlit app imports app/explainer_utils.py as explainer_utils when
started from the app directory. That file re-exports the shared implementation
from app/shared_explainer.py and preserves the function names used by the current
GCN dashboard.

## Explain one transaction

    from app.shared_explainer import create_explainer, explain_node
    from app.shared_explainer import get_top_features, get_top_edges

    model.eval()
    explainer = create_explainer(model, epochs=100)
    explanation = explain_node(explainer, graph_data, node_index=166967)
    features = get_top_features(
        explanation, 166967, top_k=10, num_nodes=graph_data.num_nodes
    )
    edges = get_top_edges(explanation, graph_data.edge_index, top_k=10)

Use the correct architecture and trained checkpoint when creating model. This
module intentionally does not guess checkpoint paths or reconstruct models from
a model name.

## Probability ensemble

The GraphSAGE+GAT ensemble averages fraud probabilities and applies a
validation-selected threshold. Averaging raw logits would change that decision
rule. Wrap the loaded component models with the exact selected weights and
threshold:

    from app.shared_explainer import ProbabilityEnsembleAdapter, create_explainer

    ensemble_model = ProbabilityEnsembleAdapter(
        [graphsage_model, gat_model],
        weights=[alpha_graphsage, alpha_gat],
        threshold=validation_selected_threshold,
    )
    explainer = create_explainer(ensemble_model, epochs=100)

The adapter returns differentiable two-class logits whose argmax follows the
configured fraud-probability threshold. Use the actual saved/selected ensemble
weights and threshold; never estimate them from test data.

## Sample-based global summary

GNNExplainer creates local explanations. A global view can be built by
aggregating local masks for a reproducible sample of transactions. This is a
sample-based summary, not a separate global explainer. Do not run it over all
203,769 transactions by default.

    from app.shared_explainer import explain_node, summarize_local_explanations

    sample_nodes = [166967, 166970, 167299]  # document the sampling strategy
    local_results = [explain_node(explainer, graph_data, i) for i in sample_nodes]
    summary = summarize_local_explanations(
        local_results, sample_nodes, graph_data, top_k=15
    )
    print(summary["top_features"])
    print(summary["top_edges"])

## Validation

Run from the repository root:

    python -m pytest tests/test_shared_explainer.py

Tests cover full-graph and shared-attribute masks, edge-mask alignment, local-mask
aggregation, ensemble threshold behavior, and explainer configuration.

## Scope and limitations

- The GCN notebook's experimental cells are preserved; this change does not
  delete or rewrite them.
- The current dashboard still loads the GCN checkpoint. Selecting/loading
  GraphSAGE, GAT, and the ensemble in the UI is a separate integration step,
  because their architectures, checkpoint files, ensemble weights, and threshold
  must be loaded consistently.
- Feature scores should be read alongside prediction-fidelity and sensitivity
  checks. Mask values alone are not causal proof.
