import math
from collections import defaultdict

from pyvis.network import Network


def create_local_network(
    selected_node,
    top_edges,
    graph_data,
    max_edges=15
):
    """
    Create a cleaner local explanation graph.

    Layout strategy:
    - selected node at center
    - direct neighbors on inner ring
    - second-hop neighbors on outer ring
    - subtle arrows and thinner edges
    """

    # -----------------------------------------------------
    # Limit edges
    # -----------------------------------------------------

    important_edges = top_edges[:max_edges]

    # -----------------------------------------------------
    # Build node groups
    # -----------------------------------------------------

    direct_neighbors = set()
    second_hop_nodes = set()

    for edge in important_edges:
        source = edge["source"]
        target = edge["target"]

        if source == selected_node:
            direct_neighbors.add(target)

        elif target == selected_node:
            direct_neighbors.add(source)

    for edge in important_edges:
        source = edge["source"]
        target = edge["target"]

        if source != selected_node and target != selected_node:
            if source in direct_neighbors:
                second_hop_nodes.add(target)
            elif target in direct_neighbors:
                second_hop_nodes.add(source)

    all_nodes = {selected_node} | direct_neighbors | second_hop_nodes

    # -----------------------------------------------------
    # Label helper
    # -----------------------------------------------------

    def get_node_label(node):
        if not hasattr(graph_data, "y"):
            return "Unknown"

        label = int(graph_data.y[node].item())

        if label == 1:
            return "Illicit"
        elif label == 0:
            return "Licit"
        else:
            return "Unknown"

    # -----------------------------------------------------
    # Create network
    # -----------------------------------------------------

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#070B12",
        font_color="#F8FAFC",
        directed=True
    )

    # -----------------------------------------------------
    # Cleaner visual options
    # -----------------------------------------------------

    net.set_options(
        """
        {
          "nodes": {
            "shape": "dot",
            "borderWidth": 2,
            "borderWidthSelected": 3,
            "font": {
              "color": "#F8FAFC",
              "size": 16,
              "face": "arial"
            },
            "shadow": {
              "enabled": true,
              "color": "rgba(0,0,0,0.35)",
              "size": 10,
              "x": 0,
              "y": 2
            }
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "dynamic",
              "roundness": 0.18
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.35
              }
            },
            "shadow": false
          },
          "physics": {
            "enabled": false
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 120,
            "navigationButtons": true,
            "keyboard": false,
            "dragNodes": true,
            "zoomView": true,
            "dragView": true
          }
        }
        """
    )

    # -----------------------------------------------------
    # Manual layout positions
    # -----------------------------------------------------

    positions = {}

    # Selected node at center
    positions[selected_node] = (0, 0)

    # Direct neighbors on inner ring
    direct_neighbors = list(direct_neighbors)
    second_hop_nodes = list(second_hop_nodes)

    inner_radius = 220
    outer_radius = 430

    if len(direct_neighbors) > 0:
        for i, node in enumerate(direct_neighbors):
            angle = (2 * math.pi * i) / len(direct_neighbors)
            x = inner_radius * math.cos(angle)
            y = inner_radius * math.sin(angle)
            positions[node] = (x, y)

    # -----------------------------------------------------
    # Map second-hop nodes near their associated direct node
    # -----------------------------------------------------

    second_hop_by_parent = defaultdict(list)

    for edge in important_edges:
        source = edge["source"]
        target = edge["target"]

        if source in direct_neighbors and target in second_hop_nodes:
            second_hop_by_parent[source].append(target)

        elif target in direct_neighbors and source in second_hop_nodes:
            second_hop_by_parent[target].append(source)

    # Place second-hop nodes outside their parent
    for parent, children in second_hop_by_parent.items():

        if parent not in positions:
            continue

        parent_x, parent_y = positions[parent]
        parent_angle = math.atan2(parent_y, parent_x)

        spread = 0.55
        count = len(children)

        for idx, child in enumerate(children):
            if count == 1:
                child_angle = parent_angle
            else:
                start_angle = parent_angle - spread / 2
                step = spread / (count - 1)
                child_angle = start_angle + idx * step

            x = outer_radius * math.cos(child_angle)
            y = outer_radius * math.sin(child_angle)
            positions[child] = (x, y)

    # Any remaining nodes without position
    placed_nodes = set(positions.keys())
    leftover_nodes = [n for n in all_nodes if n not in placed_nodes]

    if len(leftover_nodes) > 0:
        for i, node in enumerate(leftover_nodes):
            angle = (2 * math.pi * i) / len(leftover_nodes)
            x = outer_radius * math.cos(angle)
            y = outer_radius * math.sin(angle)
            positions[node] = (x, y)

    # -----------------------------------------------------
    # Add nodes
    # -----------------------------------------------------

    for node in all_nodes:

        label = get_node_label(node)
        x, y = positions[node]

        if node == selected_node:
            color = {
                "background": "#F59E0B",
                "border": "#FBBF24",
                "highlight": {
                    "background": "#FBBF24",
                    "border": "#FDE68A"
                }
            }
            size = 34
            title = (
                f"<b>Selected Transaction</b><br>"
                f"Node: {node}<br>"
                f"Label: {label}"
            )

        elif label == "Illicit":
            color = {
                "background": "#EF4444",
                "border": "#F87171",
                "highlight": {
                    "background": "#F87171",
                    "border": "#FCA5A5"
                }
            }
            size = 24
            title = (
                f"<b>Node:</b> {node}<br>"
                f"<b>Label:</b> Illicit"
            )

        elif label == "Licit":
            color = {
                "background": "#22C55E",
                "border": "#4ADE80",
                "highlight": {
                    "background": "#4ADE80",
                    "border": "#86EFAC"
                }
            }
            size = 22
            title = (
                f"<b>Node:</b> {node}<br>"
                f"<b>Label:</b> Licit"
            )

        else:
            color = {
                "background": "#64748B",
                "border": "#94A3B8",
                "highlight": {
                    "background": "#94A3B8",
                    "border": "#CBD5E1"
                }
            }
            size = 22
            title = (
                f"<b>Node:</b> {node}<br>"
                f"<b>Label:</b> Unknown"
            )

        net.add_node(
            node,
            label=str(node),
            title=title,
            color=color,
            size=size,
            x=x,
            y=y,
            fixed=True
        )

    # -----------------------------------------------------
    # Normalize edge importance
    # -----------------------------------------------------

    importance_values = [edge["importance"] for edge in important_edges]

    if importance_values:
        min_imp = min(importance_values)
        max_imp = max(importance_values)
    else:
        min_imp = 0
        max_imp = 1

    # -----------------------------------------------------
    # Add edges
    # -----------------------------------------------------

    for edge in important_edges:
        source = edge["source"]
        target = edge["target"]
        importance = edge["importance"]

        if max_imp > min_imp:
            normalized = (importance - min_imp) / (max_imp - min_imp)
        else:
            normalized = 1.0

        # Much cleaner width range
        width = 1.5 + normalized * 3.0

        # Highlight direct edges to selected node
        if source == selected_node or target == selected_node:
            edge_color = "rgba(96, 165, 250, 0.95)"
        else:
            edge_color = "rgba(56, 189, 248, 0.70)"

        net.add_edge(
            source,
            target,
            width=width,
            value=width,
            color=edge_color,
            title=f"Importance: {importance:.4f}",
            arrows="to"
        )

    return net