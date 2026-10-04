import torch

from torch_geometric.explain import (
    Explainer,
    GNNExplainer
)


# =========================================================
# CREATE GNN EXPLAINER
# =========================================================

def create_explainer(model):

    explainer = Explainer(

        model=model,

        algorithm=GNNExplainer(
            epochs=100
        ),

        explanation_type="model",

        node_mask_type="attributes",

        edge_mask_type="object",

        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw"
        )
    )

    return explainer


# =========================================================
# EXPLAIN A NODE
# =========================================================

def explain_node(
    explainer,
    graph_data,
    node_index
):

    explanation = explainer(
        graph_data.x,
        graph_data.edge_index,
        index=node_index
    )

    return explanation


# =========================================================
# GET TOP IMPORTANT EDGES
# =========================================================

def get_top_edges(
    explanation,
    edge_index,
    top_k=10
):

    edge_mask = explanation.edge_mask

    if edge_mask is None:
        return []

    edge_mask = (
        edge_mask
        .detach()
        .cpu()
    )

    edge_index_cpu = (
        edge_index
        .detach()
        .cpu()
    )

    top_k = min(
        top_k,
        edge_mask.numel()
    )

    values, indices = torch.topk(
        edge_mask,
        k=top_k
    )

    results = []

    for importance, edge_idx in zip(
        values,
        indices
    ):

        source = int(
            edge_index_cpu[
                0,
                edge_idx
            ].item()
        )

        target = int(
            edge_index_cpu[
                1,
                edge_idx
            ].item()
        )

        results.append(
            {
                "source": source,
                "target": target,
                "importance": float(
                    importance.item()
                )
            }
        )

    return results


# =========================================================
# GET TOP FEATURES
# =========================================================

def get_top_features(
    explanation,
    node_index,
    top_k=10
):

    node_mask = explanation.node_mask

    if node_mask is None:
        return []

    node_mask = (
        node_mask
        .detach()
        .cpu()
    )

    # -----------------------------------------------------
    # GNNExplainer may return masks in slightly different
    # shapes depending on PyG version/configuration.
    # -----------------------------------------------------

    if node_mask.dim() == 2:

        # [num_nodes, num_features]
        if node_mask.shape[0] > 1:
            feature_mask = node_mask[
                node_index
            ]

        else:
            # [1, num_features]
            feature_mask = node_mask[0]

    elif node_mask.dim() == 1:

        feature_mask = node_mask

    else:

        raise ValueError(
            f"Unexpected node mask shape: "
            f"{node_mask.shape}"
        )


    top_k = min(
        top_k,
        feature_mask.numel()
    )

    values, indices = torch.topk(
        feature_mask,
        k=top_k
    )

    results = []

    for importance, feature_idx in zip(
        values,
        indices
    ):

        results.append(
            {
                "feature": int(
                    feature_idx.item()
                ),
                "importance": float(
                    importance.item()
                )
            }
        )

    return results


# =========================================================
# GET IMPORTANT NEIGHBORS
# =========================================================

def get_important_neighbors(
    node_index,
    top_edges,
    graph_data,
    top_k=5
):

    neighbors = {}

    for edge in top_edges:

        source = edge["source"]
        target = edge["target"]
        importance = edge["importance"]

        neighbor = None

        if source == node_index:
            neighbor = target

        elif target == node_index:
            neighbor = source

        if neighbor is None:
            continue

        # Keep strongest importance if same neighbor appears
        if (
            neighbor not in neighbors
            or importance
            > neighbors[neighbor]["importance"]
        ):

            label = None

            if hasattr(
                graph_data,
                "y"
            ):

                raw_label = int(
                    graph_data.y[
                        neighbor
                    ].item()
                )

                if raw_label == 1:
                    label = "Illicit"

                elif raw_label == 0:
                    label = "Licit"

                else:
                    label = "Unknown"

            neighbors[neighbor] = {
                "node": neighbor,
                "importance": importance,
                "label": label
            }


    neighbors = list(
        neighbors.values()
    )

    neighbors.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return neighbors[:top_k]

def get_visualization_edges(
    explanation,
    edge_index,
    selected_node,
    direct_k=5,
    additional_k=10
):

    edge_mask = explanation.edge_mask

    if edge_mask is None:
        return []

    edge_mask = (
        edge_mask
        .detach()
        .cpu()
    )

    edge_index_cpu = (
        edge_index
        .detach()
        .cpu()
    )

    direct_edges = []
    other_edges = []

    # -----------------------------------------------------
    # Examine every explained edge
    # -----------------------------------------------------

    for edge_idx in range(
        edge_mask.numel()
    ):

        source = int(
            edge_index_cpu[
                0,
                edge_idx
            ].item()
        )

        target = int(
            edge_index_cpu[
                1,
                edge_idx
            ].item()
        )

        importance = float(
            edge_mask[
                edge_idx
            ].item()
        )

        edge = {
            "source": source,
            "target": target,
            "importance": importance
        }

        # Directly connected to selected node
        if (
            source == selected_node
            or target == selected_node
        ):

            direct_edges.append(
                edge
            )

        else:

            other_edges.append(
                edge
            )

    # -----------------------------------------------------
    # Sort by explanation importance
    # -----------------------------------------------------

    direct_edges.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    other_edges.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    # -----------------------------------------------------
    # Always keep strongest direct connections first
    # -----------------------------------------------------

    selected_direct = direct_edges[
        :direct_k
    ]

    # -----------------------------------------------------
    # Prefer additional edges connected to direct neighbors
    # -----------------------------------------------------

    direct_neighbors = set()

    for edge in selected_direct:

        if edge["source"] == selected_node:

            direct_neighbors.add(
                edge["target"]
            )

        else:

            direct_neighbors.add(
                edge["source"]
            )


    second_hop_edges = []

    unrelated_edges = []

    for edge in other_edges:

        if (
            edge["source"] in direct_neighbors
            or edge["target"] in direct_neighbors
        ):

            second_hop_edges.append(
                edge
            )

        else:

            unrelated_edges.append(
                edge
            )

    # Prefer second-hop edges
    additional_edges = second_hop_edges[
        :additional_k
    ]

    # Fill remaining slots if needed
    remaining = (
        additional_k
        - len(additional_edges)
    )

    if remaining > 0:

        additional_edges.extend(
            unrelated_edges[
                :remaining
            ]
        )

    return (
        selected_direct
        + additional_edges
    )