import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


# =========================================================
# GCN MODEL
# =========================================================

class GCN(torch.nn.Module):

    def __init__(
        self,
        num_features,
        hidden_channels,
        num_classes
    ):

        super().__init__()

        self.conv1 = GCNConv(
            num_features,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            num_classes
        )

    def forward(
        self,
        x,
        edge_index
    ):

        # First graph convolution
        x = self.conv1(
            x,
            edge_index
        )

        # Activation
        x = F.relu(x)

        # Dropout is automatically disabled during model.eval()
        x = F.dropout(
            x,
            p=0.5,
            training=self.training
        )

        # Final graph convolution
        x = self.conv2(
            x,
            edge_index
        )

        return x


# =========================================================
# LOAD GRAPH
# =========================================================

def load_graph(graph_path):

    graph_data = torch.load(
        graph_path,
        map_location="cpu",
        weights_only=False
    )

    return graph_data


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

def load_model(
    model_path,
    graph_data
):

    # Load weights
    state_dict = torch.load(
        model_path,
        map_location="cpu"
    )

    # -----------------------------------------------------
    # Infer architecture directly from trained weights
    # -----------------------------------------------------

    # conv1.lin.weight shape:
    # [hidden_channels, num_features]

    conv1_weight = state_dict[
        "conv1.lin.weight"
    ]

    hidden_channels = conv1_weight.shape[0]
    num_features = conv1_weight.shape[1]

    # conv2.lin.weight shape:
    # [num_classes, hidden_channels]

    conv2_weight = state_dict[
        "conv2.lin.weight"
    ]

    num_classes = conv2_weight.shape[0]

    # Safety check
    if graph_data.x.shape[1] != num_features:

        raise ValueError(
            f"Feature mismatch: graph contains "
            f"{graph_data.x.shape[1]} features, "
            f"but the trained model expects "
            f"{num_features}."
        )

    # Recreate model architecture
    model = GCN(
        num_features=num_features,
        hidden_channels=hidden_channels,
        num_classes=num_classes
    )

    # Load trained weights
    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# =========================================================
# RUN MODEL ON COMPLETE GRAPH
# =========================================================

@torch.no_grad()
def get_predictions(
    model,
    graph_data
):

    model.eval()

    logits = model(
        graph_data.x,
        graph_data.edge_index
    )

    # Convert raw model outputs into probabilities
    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = probabilities.argmax(
        dim=1
    )

    return (
        logits,
        probabilities,
        predictions
    )


# =========================================================
# ANALYZE INDIVIDUAL NODE
# =========================================================

def analyze_node(
    node_index,
    graph_data,
    probabilities,
    predictions
):

    num_nodes = graph_data.num_nodes

    if node_index < 0 or node_index >= num_nodes:

        raise ValueError(
            f"Node index must be between "
            f"0 and {num_nodes - 1}."
        )

    predicted_class = int(
        predictions[node_index].item()
    )

    class_probabilities = (
        probabilities[node_index]
        .detach()
        .cpu()
        .tolist()
    )

    confidence = float(
        probabilities[
            node_index,
            predicted_class
        ].item()
    )

    # Get true label when available
    true_label = None

    if hasattr(graph_data, "y"):

        label = int(
            graph_data.y[node_index].item()
        )

        # Many preprocessing pipelines use -1
        # for unknown/unlabelled transactions.
        if label >= 0:
            true_label = label

    return {
        "node_index": node_index,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "true_label": true_label
    }