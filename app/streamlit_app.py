from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
import streamlit.components.v1 as components

try:
    from .model_registry import load_model_registry, predict_graph
    from .results_loader import available_artifacts, load_recorded_metrics
    from .explainer_utils import (
        create_explainer,
        explain_node,
        get_important_neighbors,
        get_top_edges,
        get_top_features,
        get_visualization_edges,
    )
    from .visualization import create_local_network
except ImportError:  # supports: streamlit run app/streamlit_app.py
    from model_registry import load_model_registry, predict_graph
    from results_loader import available_artifacts, load_recorded_metrics
    from explainer_utils import (
        create_explainer,
        explain_node,
        get_important_neighbors,
        get_top_edges,
        get_top_features,
        get_visualization_edges,
    )
    from visualization import create_local_network


st.set_page_config(
    page_title="GraphShield | Explainable Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic_graph.pt"
MODEL_ORDER = ["GCN", "GraphSAGE", "GAT", "Ensemble"]
MODEL_LABELS = {
    "GCN": "GCN",
    "GraphSAGE": "GraphSAGE",
    "GAT": "GAT",
    "Ensemble": "GraphSAGE–GAT Ensemble",
}
CLASS_LABELS = {0: "Legitimate", 1: "Fraud / illicit"}

st.markdown(
    """
<style>
.stApp { background: radial-gradient(circle at top right, rgba(34,197,94,.07), transparent 28%), #070B12; color:#E2E8F0; }
.block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0B111B,#080D15); border-right:1px solid #1E293B; }
h1,h2,h3 { letter-spacing:-.025em; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#111827,#0A101B); border:1px solid #1E293B; padding:15px; border-radius:14px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#233047; border-radius:16px; }
.stButton button { border-radius:10px; min-height:42px; font-weight:600; }
.gs-kicker { color:#4ADE80; font-weight:700; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; }
.gs-subtitle { color:#94A3B8; margin-top:-.5rem; margin-bottom:1.25rem; }
.gs-card { background:linear-gradient(145deg,#111827,#0A101B); border:1px solid #1E293B; border-radius:16px; padding:20px; height:100%; }
.gs-muted { color:#94A3B8; font-size:.9rem; }
.gs-good { color:#4ADE80; }
.gs-warning { color:#FBBF24; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading graph, model checkpoints, and recorded metrics…")
def initialize_system(graph_path_text: str, project_root_text: str):
    graph_path = Path(graph_path_text)
    project_root = Path(project_root_text)
    graph_data = None
    model_registry: dict[str, Any] = {}
    model_errors: dict[str, str] = {}
    system_error = None
    metrics, metrics_errors = load_recorded_metrics(project_root)

    if not graph_path.is_file():
        system_error = (
            f"Processed graph not found at {graph_path}. "
            "The app expects data/processed/elliptic_graph.pt. "
            "If you currently have elliptic_graph.zip, use the project's existing "
            "preprocessing/export workflow to create the .pt file; this app will not "
            "silently unpack or replace research data."
        )
        return graph_data, model_registry, model_errors, metrics, metrics_errors, system_error

    try:
        try:
            graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)
        except TypeError:
            graph_data = torch.load(graph_path, map_location="cpu")
        if not hasattr(graph_data, "x") or not hasattr(graph_data, "edge_index"):
            raise ValueError("Processed graph must contain x and edge_index tensors.")
        if graph_data.x.dim() != 2 or graph_data.edge_index.dim() != 2:
            raise ValueError("Unexpected graph tensor dimensions.")
        model_registry, model_errors = load_model_registry(graph_data, project_root)
    except Exception as exc:
        system_error = f"{type(exc).__name__}: {exc}"
    return graph_data, model_registry, model_errors, metrics, metrics_errors, system_error


(
    graph_data,
    models,
    model_errors,
    recorded_metrics,
    metrics_errors,
    system_error,
) = initialize_system(str(GRAPH_PATH), str(PROJECT_ROOT))

if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}
if "analysis_node" not in st.session_state:
    st.session_state.analysis_node = None
if "explanation_cache" not in st.session_state:
    st.session_state.explanation_cache = {}
if "explainer_cache" not in st.session_state:
    st.session_state.explainer_cache = {}

with st.sidebar:
    st.markdown("## 🛡️ GraphShield")
    st.caption("Explainable financial fraud intelligence")
    st.divider()
    page = st.radio(
        "Workspace",
        ["Overview", "Transaction Analyzer", "Model Performance", "Explainability"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("#### Runtime status")
    if graph_data is not None:
        st.success(f"Graph loaded · {int(graph_data.num_nodes):,} nodes")
    elif system_error:
        st.error("Graph unavailable")
    for key in MODEL_ORDER:
        if key in models:
            st.caption(f"🟢 {MODEL_LABELS[key]} ready")
        else:
            st.caption(f"⚪ {MODEL_LABELS[key]} unavailable")
    st.divider()
    st.caption("Dataset: Elliptic Bitcoin")
    st.caption("Explanations: GNNExplainer")
    st.caption("Recorded results are separate from live predictions.")

st.markdown('<div class="gs-kicker">Graph intelligence · explainable AI</div>', unsafe_allow_html=True)
st.title("GraphShield")
st.markdown(
    '<div class="gs-subtitle">Explore transaction predictions, compare graph neural networks, and inspect the local evidence behind model decisions.</div>',
    unsafe_allow_html=True,
)

if system_error:
    st.error("The transaction graph could not be initialized.")
    st.code(system_error)
    st.info("Recorded experiment metrics may still be available below, but live predictions and explanations require the processed graph.")

if model_errors:
    with st.expander("Model loading details", expanded=False):
        for key, error in model_errors.items():
            st.warning(f"{MODEL_LABELS.get(key, key)}: {error}")


def metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return "Unavailable" if value is None else f"{value:.4f}"


def recorded_frame(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for key in MODEL_ORDER:
        if key not in metrics_by_model:
            continue
        m = metrics_by_model[key]
        rows.append({
            "Model": MODEL_LABELS[key],
            "Accuracy": m.get("accuracy"),
            "Fraud precision": m.get("fraud_precision"),
            "Fraud recall": m.get("fraud_recall"),
            "Fraud F1": m.get("fraud_f1"),
            "PR-AUC (AP)": m.get("pr_auc_average_precision"),
            "ROC-AUC": m.get("roc_auc"),
        })
    return pd.DataFrame(rows)


def get_prediction_cache(key: str) -> dict[str, Any] | None:
    return st.session_state.prediction_cache.get(key)


def run_model_predictions(node_index: int) -> dict[str, Any]:
    if graph_data is None:
        raise RuntimeError("The processed graph is not loaded.")
    if not 0 <= int(node_index) < int(graph_data.num_nodes):
        raise ValueError(f"Node index must be between 0 and {int(graph_data.num_nodes) - 1}.")
    cache = st.session_state.prediction_cache
    # Cache key includes graph file identity and model checkpoint identity.
    graph_stamp = f"{GRAPH_PATH.stat().st_mtime_ns}:{GRAPH_PATH.stat().st_size}"
    for key, bundle in models.items():
        checkpoint = bundle.checkpoint_path
        stamp = f"{checkpoint}:{checkpoint.stat().st_mtime_ns if checkpoint.is_file() else 'missing'}"
        cache_key = f"{graph_stamp}|{key}|{stamp}"
        if cache_key not in cache:
            cache[cache_key] = predict_graph(bundle, graph_data)
        cache[key] = cache[cache_key]
    return {
        key: {
            "node_index": int(node_index),
            "predicted_class": int(cache[key]["predictions"][node_index].item()),
            "fraud_probability": float(cache[key]["fraud_probability"][node_index].item()),
            "legitimate_probability": float(cache[key]["probabilities"][node_index, 0].item()),
            "probabilities": cache[key]["probabilities"][node_index].tolist(),
            "threshold": float(bundle.threshold),
        }
        for key, bundle in models.items()
    }


def get_explanation(node_index: int, model_key: str, epochs: int) -> dict[str, Any]:
    if graph_data is None:
        raise RuntimeError("The processed graph is not loaded.")
    if model_key not in models:
        raise RuntimeError(f"{MODEL_LABELS.get(model_key, model_key)} is not available.")
    cache_key = f"{GRAPH_PATH.stat().st_mtime_ns}:{node_index}:{model_key}:{epochs}"
    if cache_key in st.session_state.explanation_cache:
        return st.session_state.explanation_cache[cache_key]

    bundle = models[model_key]
    explainer_key = f"{model_key}:{epochs}"
    if explainer_key not in st.session_state.explainer_cache:
        st.session_state.explainer_cache[explainer_key] = create_explainer(bundle.model, epochs=epochs)
    explainer = st.session_state.explainer_cache[explainer_key]
    explanation = explain_node(explainer, graph_data, int(node_index))
    top_edges = get_top_edges(explanation, graph_data.edge_index, top_k=30)
    visualization_edges = get_visualization_edges(
        explanation, graph_data.edge_index, selected_node=int(node_index), direct_k=8, additional_k=12
    )
    top_features = get_top_features(
        explanation, int(node_index), top_k=20, num_nodes=int(graph_data.num_nodes)
    )
    neighbors = get_important_neighbors(int(node_index), top_edges, graph_data, top_k=8)
    result = {
        "top_edges": top_edges,
        "visualization_edges": visualization_edges,
        "top_features": top_features,
        "neighbors": neighbors,
    }
    st.session_state.explanation_cache[cache_key] = result
    return result


if page == "Overview":
    st.markdown("### Project overview")
    st.write(
        "GraphShield is a research dashboard for transaction-level illicit activity detection on the "
        "Elliptic Bitcoin graph. It combines graph neural network predictions with local GNNExplainer "
        "evidence so users can compare model behavior rather than relying on accuracy alone."
    )
    if graph_data is not None:
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Transaction nodes", f"{int(graph_data.num_nodes):,}")
        n2.metric("Directed graph edges", f"{int(graph_data.edge_index.shape[1]):,}")
        n3.metric("Input features", f"{int(graph_data.x.shape[1]):,}")
        known = int(((graph_data.y == 0) | (graph_data.y == 1)).sum().item()) if hasattr(graph_data, "y") else None
        n4.metric("Known labels", f"{known:,}" if known is not None else "Unavailable")
    st.markdown("### The four model approaches")
    descriptions = {
        "GCN": "Learns node representations by aggregating information from graph neighborhoods.",
        "GraphSAGE": "Uses a learned neighborhood aggregation function to combine node and neighbor information.",
        "GAT": "Learns attention weights over neighboring nodes during message passing.",
        "Ensemble": "Uses the recorded validation-selected GraphSAGE/GAT probability weighting and threshold.",
    }
    cols = st.columns(4)
    for col, key in zip(cols, MODEL_ORDER):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {MODEL_LABELS[key]}")
                st.caption("Runtime checkpoint available" if key in models else "Runtime checkpoint unavailable")
                st.write(descriptions[key])
                m = recorded_metrics.get(key)
                if m:
                    st.metric("Recorded fraud F1", metric_value(m, "fraud_f1"))
                else:
                    st.caption("Recorded metrics unavailable")
    st.markdown("### Recorded test-set comparison")
    frame = recorded_frame(recorded_metrics)
    if not frame.empty:
        st.dataframe(frame, use_container_width=True, hide_index=True)
        chart_data = frame.set_index("Model")[["Fraud precision", "Fraud recall", "Fraud F1"]]
        st.bar_chart(chart_data, use_container_width=True)
        st.caption("These are recorded evaluation metrics from experiment artifacts, not live predictions for a selected transaction.")
    else:
        st.info("No readable metric artifacts were found.")
    if metrics_errors:
        with st.expander("Metric artifact availability"):
            for key, error in metrics_errors.items():
                st.caption(f"{MODEL_LABELS[key]}: {error}")


elif page == "Transaction Analyzer":
    st.markdown("### Transaction analyzer")
    st.write("Enter a graph node index to compare the predictions of every model whose local checkpoint is available.")
    if graph_data is None:
        st.warning("Live analysis requires the processed graph file.")
    else:
        node_value = st.number_input(
            "Graph node index",
            min_value=0,
            max_value=int(graph_data.num_nodes) - 1,
            value=int(st.session_state.analysis_node or 0),
            step=1,
            help="This is the internal node index in elliptic_graph.pt, not an external transaction hash.",
        )
        if st.button("Analyze with available models", type="primary", use_container_width=True):
            with st.spinner("Running or reusing cached graph-wide predictions…"):
                try:
                    run_model_predictions(int(node_value))
                    st.session_state.analysis_node = int(node_value)
                except Exception as exc:
                    st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
        node = st.session_state.analysis_node
        if node is not None and any(key in st.session_state.prediction_cache for key in models):
            # Refresh node-specific view from cached graph predictions.
            current = {}
            cache = st.session_state.prediction_cache
            for key, bundle in models.items():
                found = next((v for ck, v in cache.items() if ck.startswith(f"{GRAPH_PATH.stat().st_mtime_ns}:{GRAPH_PATH.stat().st_size}|{key}|")), None)
                if found is not None:
                    current[key] = {
                        "predicted_class": int(found["predictions"][node].item()),
                        "fraud_probability": float(found["fraud_probability"][node].item()),
                        "legitimate_probability": float(found["probabilities"][node, 0].item()),
                    }
            if current:
                st.markdown(f"#### Transaction node {node:,}")
                cols = st.columns(min(4, len(current)))
                for col, (key, result) in zip(cols, current.items()):
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{MODEL_LABELS[key]}**")
                            predicted = result["predicted_class"]
                            st.metric("Prediction", CLASS_LABELS.get(predicted, "Unknown"))
                            st.metric("Fraud probability", f"{result['fraud_probability'] * 100:.2f}%")
                            if key == "Ensemble":
                                st.caption(f"Recorded decision threshold: {models[key].threshold:.2f}")
                st.markdown("#### Cross-model comparison")
                comparison = pd.DataFrame([
                    {
                        "Model": MODEL_LABELS[key],
                        "Prediction": CLASS_LABELS.get(value["predicted_class"], "Unknown"),
                        "Fraud probability": value["fraud_probability"],
                        "Legitimate probability": value["legitimate_probability"],
                    }
                    for key, value in current.items()
                ])
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                if len({v["predicted_class"] for v in current.values()}) > 1:
                    st.warning("Models disagree on this node. Treat this as a signal for further review, not proof of fraud or a calibrated uncertainty estimate.")
                if hasattr(graph_data, "y"):
                    actual = int(graph_data.y[node].item())
                    if actual in (0, 1):
                        st.caption(f"Recorded label: {CLASS_LABELS[actual]}. This label is distinct from each model's prediction.")
                    else:
                        st.caption("Recorded label: Unknown / unlabeled. The model prediction is not ground truth.")
                st.markdown("#### Explain the selected model")
                available = list(current.keys())
                default = "GraphSAGE" if "GraphSAGE" in available else available[0]
                selected_model = st.selectbox("Model to explain", available, index=available.index(default))
                epochs = st.slider("GNNExplainer optimization epochs", min_value=20, max_value=150, value=50, step=10)
                if st.button("Generate local explanation", use_container_width=True):
                    with st.spinner("Optimizing a local GNN explanation…"):
                        try:
                            explanation = get_explanation(int(node), selected_model, int(epochs))
                            st.session_state["active_explanation"] = {
                                "node": int(node), "model": selected_model, "epochs": int(epochs),
                                "result": explanation,
                            }
                        except Exception as exc:
                            st.error(f"Explanation failed: {type(exc).__name__}: {exc}")
                active = st.session_state.get("active_explanation")
                if active and active["node"] == node and active["model"] == selected_model:
                    ex = active["result"]
                    if ex["visualization_edges"]:
                        try:
                            network = create_local_network(
                                selected_node=int(node), top_edges=ex["visualization_edges"],
                                graph_data=graph_data, max_edges=20,
                            )
                            components.html(network.generate_html(), height=560, scrolling=False)
                        except Exception as exc:
                            st.warning(f"Graph rendering failed: {type(exc).__name__}: {exc}")
                    if ex["top_features"]:
                        st.markdown("**Top-ranked feature indices**")
                        features = pd.DataFrame(ex["top_features"]).rename(columns={"feature": "Feature index", "importance": "Mask importance"})
                        features["Feature"] = features["Feature index"].map(lambda x: f"Feature {x}")
                        st.bar_chart(features.set_index("Feature")[["Mask importance"]], horizontal=True)
                        st.caption("Feature columns are anonymous indices; semantic meanings are not inferred.")
                    if ex["neighbors"]:
                        st.markdown("**Influential neighboring transactions**")
                        st.dataframe(pd.DataFrame(ex["neighbors"]), use_container_width=True, hide_index=True)


elif page == "Model Performance":
    st.markdown("### Model performance")
    st.write("Compare saved test-set evaluation artifacts. Metrics are not recalculated from live dashboard predictions.")
    frame = recorded_frame(recorded_metrics)
    if frame.empty:
        st.info("No recorded metrics are available.")
    else:
        metric_choice = st.selectbox(
            "Metric to compare",
            ["Fraud F1", "Fraud precision", "Fraud recall", "Accuracy", "PR-AUC (AP)", "ROC-AUC"],
        )
        metric_column = {
            "Fraud F1": "Fraud F1", "Fraud precision": "Fraud precision",
            "Fraud recall": "Fraud recall", "Accuracy": "Accuracy",
            "PR-AUC (AP)": "PR-AUC (AP)", "ROC-AUC": "ROC-AUC",
        }[metric_choice]
        chart = frame[["Model", metric_column]].dropna().set_index("Model")
        if not chart.empty:
            st.bar_chart(chart, use_container_width=True)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.caption("For this imbalanced dataset, inspect fraud precision, recall, F1, and PR-AUC alongside accuracy.")
    selected = st.selectbox("Inspect evaluation artifacts", MODEL_ORDER, index=1)
    metric_data = recorded_metrics.get(selected)
    if metric_data and metric_data.get("confusion_matrix"):
        st.markdown("#### Recorded confusion matrix")
        cm = pd.DataFrame(
            metric_data["confusion_matrix"],
            index=["Actual legitimate", "Actual fraud"],
            columns=["Predicted legitimate", "Predicted fraud"],
        )
        st.dataframe(cm, use_container_width=True)
    images = available_artifacts(PROJECT_ROOT, selected)
    if images:
        st.markdown("#### Available experiment plots")
        for label, path in images:
            with st.expander(label, expanded=(label == "Confusion matrix")):
                st.image(str(path), caption=f"Recorded artifact · {path.relative_to(PROJECT_ROOT).as_posix()}", use_container_width=True)
    else:
        st.info("No generated plot files for this model were found locally.")
    if selected == "Ensemble" and metric_data:
        st.warning(
            f"Recorded ensemble selection: GraphSAGE weight α={metric_data.get('alpha_graphsage')}, "
            f"GAT weight α={metric_data.get('alpha_gat')}, threshold={metric_data.get('threshold')}. "
            "The saved configuration assigns all probability weight to GraphSAGE, so this run is a degenerate blend rather than evidence that combining both models improved performance."
        )


elif page == "Explainability":
    st.markdown("### Explainability lab")
    st.write("Explore a node's local explanation and compare feature/edge masks across available models. Explanation masks describe model behavior; they are not causal proof.")
    if graph_data is None or not models:
        st.warning("The graph and at least one valid model checkpoint are required for live explanations.")
    else:
        default_node = int(st.session_state.analysis_node or 0)
        node = st.number_input(
            "Transaction node index",
            min_value=0, max_value=int(graph_data.num_nodes) - 1, value=default_node, step=1,
            key="xai_node",
        )
        selected_model = st.selectbox("Model", list(models.keys()), index=0, key="xai_model")
        epochs = st.slider("Explanation optimization epochs", min_value=20, max_value=150, value=50, step=10, key="xai_epochs")
        explain_col, compare_col = st.columns(2)
        with explain_col:
            generate_one = st.button("Explore this explanation", type="primary", use_container_width=True)
        with compare_col:
            generate_compare = st.button("Compare available models", use_container_width=True)
        if generate_one:
            with st.spinner(f"Generating {MODEL_LABELS[selected_model]} explanation…"):
                try:
                    result = get_explanation(int(node), selected_model, int(epochs))
                    st.session_state["xai_active"] = {
                        "node": int(node), "model": selected_model, "epochs": int(epochs), "result": result
                    }
                except Exception as exc:
                    st.error(f"Explanation failed: {type(exc).__name__}: {exc}")
        if generate_compare:
            results = {}
            progress = st.progress(0, text="Preparing cross-model explanations…")
            keys = list(models.keys())
            for index, key in enumerate(keys):
                try:
                    results[key] = get_explanation(int(node), key, int(epochs))
                except Exception as exc:
                    results[key] = {"error": f"{type(exc).__name__}: {exc}"}
                progress.progress((index + 1) / len(keys), text=f"Explained {index + 1} of {len(keys)} models")
            progress.empty()
            st.session_state["xai_comparison"] = {"node": int(node), "epochs": int(epochs), "results": results}
        active = st.session_state.get("xai_active")
        if active and active["node"] == int(node):
            ex = active["result"]
            st.markdown(f"#### {MODEL_LABELS[active['model']]} · node {node}")
            left, right = st.columns([1.45, 1])
            with left:
                st.markdown("**Local graph evidence**")
                if ex["visualization_edges"]:
                    try:
                        network = create_local_network(
                            selected_node=int(node), top_edges=ex["visualization_edges"],
                            graph_data=graph_data, max_edges=25,
                        )
                        components.html(network.generate_html(), height=650, scrolling=False)
                    except Exception as exc:
                        st.warning(f"Graph rendering failed: {type(exc).__name__}: {exc}")
                else:
                    st.info("No important edges were returned for this explanation.")
            with right:
                st.markdown("**Feature explorer**")
                if ex["top_features"]:
                    features = pd.DataFrame(ex["top_features"]).rename(columns={"feature": "Feature index", "importance": "Mask importance"})
                    features["Feature"] = features["Feature index"].map(lambda x: f"Feature {x}")
                    st.bar_chart(features.set_index("Feature")[["Mask importance"]], horizontal=True)
                    st.dataframe(features[["Feature index", "Mask importance"]], hide_index=True, use_container_width=True)
                    st.caption("Feature indices are shown because the dataset's feature-name mapping is not assumed.")
                else:
                    st.info("No feature mask was returned.")
            with st.expander("Ranked edge evidence"):
                st.dataframe(pd.DataFrame(ex["top_edges"]), use_container_width=True, hide_index=True)
            with st.expander("Influential neighbors"):
                st.dataframe(pd.DataFrame(ex["neighbors"]), use_container_width=True, hide_index=True)
            st.info(
                "Interpretation: the listed feature and edge masks identify components the explainer found useful for this model's local prediction. "
                "They do not establish causation, and GAT attention coefficients are not interchangeable with GNNExplainer masks."
            )
        comparison = st.session_state.get("xai_comparison")
        if comparison and comparison["node"] == int(node):
            st.markdown("#### Cross-model explanation comparison")
            rows = []
            for key, result in comparison["results"].items():
                if "error" in result:
                    rows.append({"Model": MODEL_LABELS[key], "Status": result["error"], "Top feature indices": "", "Top edge count": 0})
                    continue
                rows.append({
                    "Model": MODEL_LABELS[key],
                    "Status": "Ready",
                    "Top feature indices": ", ".join(str(x["feature"]) for x in result["top_features"][:5]),
                    "Top edge count": len(result["top_edges"]),
                    "Top neighbor indices": ", ".join(str(x["node"]) for x in result["neighbors"][:5]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Feature/edge rankings can differ across models; compare them as model-specific evidence, not as a single ground-truth explanation.")
