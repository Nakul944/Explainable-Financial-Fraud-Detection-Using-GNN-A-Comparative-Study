"""Backward-compatible imports for the shared explanation implementation."""
from shared_explainer import (
    ProbabilityEnsembleAdapter,
    create_explainer,
    explain_node,
    get_important_neighbors,
    get_top_edges,
    get_top_features,
    get_visualization_edges,
    summarize_local_explanations,
)

__all__ = [
    "ProbabilityEnsembleAdapter",
    "create_explainer",
    "explain_node",
    "get_important_neighbors",
    "get_top_edges",
    "get_top_features",
    "get_visualization_edges",
    "summarize_local_explanations",
]
