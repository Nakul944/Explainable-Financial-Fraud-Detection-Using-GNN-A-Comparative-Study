from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import torch

from model_utils import (
    load_graph,
    load_model,
    get_predictions,
    analyze_node
)

from explainer_utils import (
    create_explainer,
    explain_node,
    get_top_edges,
    get_top_features,
    get_important_neighbors,
    get_visualization_edges
)

import streamlit.components.v1 as components

from visualization import (
    create_local_network
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="GraphShield | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT PATHS
# =========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "best_gcn_model.pt"
)

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "elliptic_graph.pt"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scaler.pkl"
)

CLASSES_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Elliptic_Bitcoin"
    / "elliptic_txs_classes.csv"
)

EDGES_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Elliptic_Bitcoin"
    / "elliptic_txs_edgelist.csv"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Elliptic_Bitcoin"
    / "elliptic_txs_features.csv"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at top right,
            rgba(34, 197, 94, 0.06),
            transparent 25%
        ),
        #070B12;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background-color: transparent;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 4px;
    letter-spacing: -1px;
}

.main-subtitle {
    color: #94A3B8;
    font-size: 1rem;
    margin-bottom: 22px;
}

.green-text {
    color: #22C55E;
}

.tech-badge {
    display: inline-block;
    background-color: #0F1F17;
    color: #4ADE80;
    border: 1px solid #1B5E35;
    padding: 6px 12px;
    margin-right: 7px;
    margin-bottom: 5px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.section-title {
    color: #F1F5F9;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 15px;
}

.dashboard-card {
    background:
        linear-gradient(
            145deg,
            #111827,
            #0A101B
        );

    border: 1px solid #1E293B;
    border-radius: 16px;

    padding: 22px;

    height: 200px;
    box-sizing: border-box;

    display: flex;
    flex-direction: column;

    overflow: hidden;

    box-shadow:
        0 8px 30px
        rgba(0, 0, 0, 0.22);

    transition:
        transform 0.2s ease,
        border-color 0.2s ease;
}

.dashboard-card:hover {
    transform: translateY(-3px);
    border-color: #256B43;
}

.card-icon {
    font-size: 1.6rem;
    color: #22C55E;
    margin-bottom: 8px;
    line-height: 1;
}

.card-label {
    color: #94A3B8;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    margin-bottom: 7px;
}

.card-number {
    color: #F8FAFC;
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 10px;
}

.card-detail {
    color: #64748B;
    font-size: 0.78rem;
    line-height: 1.45;
    margin-top: auto;
    white-space: normal;
    overflow-wrap: break-word;
    word-break: normal;
}

.analyzer-panel {
    background:
        linear-gradient(
            120deg,
            #0F172A,
            #0B1220
        );

    border: 1px solid #1E293B;

    border-radius: 16px;

    padding: 22px;

    margin-top: 10px;
    margin-bottom: 20px;

    color: #CBD5E1;
    line-height: 1.6;
}

.status-pill {
    display: inline-block;
    background-color: #102719;
    color: #4ADE80;
    border: 1px solid #1B5E35;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
}

div[data-testid="stMetric"] {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    padding: 16px;
    border-radius: 12px;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0B111B,
            #080D15
        );

    border-right: 1px solid #172033;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 44px;
}

</style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL / GRAPH
# =========================================================

@st.cache_resource
def initialize_system():

    graph_data = load_graph(
        GRAPH_PATH
    )

    model = load_model(
        MODEL_PATH,
        graph_data
    )

    logits, probabilities, predictions = (
        get_predictions(
            model,
            graph_data
        )
    )

    return (
        graph_data,
        model,
        logits,
        probabilities,
        predictions
    )


@st.cache_resource
def initialize_explainer(_model):

    return create_explainer(
        _model
    )


try:

    (
        graph_data,
        model,
        logits,
        probabilities,
        predictions
    ) = initialize_system()

    explainer = initialize_explainer(
        model
    )

    model_loaded = True
    model_error = None

except Exception as e:

    graph_data = None
    model = None
    logits = None
    probabilities = None
    predictions = None
    explainer = None

    model_loaded = False
    model_error = str(e)


# =========================================================
# GRAPH STATS
# =========================================================

if model_loaded:

    NUM_NODES = graph_data.num_nodes
    NUM_EDGES = graph_data.edge_index.shape[1]

else:

    NUM_NODES = 0
    NUM_EDGES = 0


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analyzed_node" not in st.session_state:
    st.session_state.analyzed_node = None

if "explanation_result" not in st.session_state:
    st.session_state.explanation_result = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
<h2 style="color:#F8FAFC; margin-bottom:0;">
🛡️ GraphShield
</h2>

<p style="
    color:#64748B;
    font-size:0.80rem;
    margin-top:3px;
">
Financial Fraud Intelligence
</p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Transaction Analyzer",
            "Model Performance",
            "Explainability"
        ]
    )

    st.divider()

    st.markdown("##### System Status")

    if model_loaded:

        st.success(
            "● GCN Model Ready"
        )

    else:

        st.error(
            "● Model Loading Failed"
        )

        with st.expander(
            "View Error"
        ):
            st.code(
                model_error
            )

    st.caption(
        "Dataset: Elliptic Bitcoin Transaction Dataset"
    )

    st.caption(
        "Model: Graph Convolutional Network"
    )

    st.caption(
        "Explainability: GNNExplainer"
    )

    st.divider()

    st.markdown(
        """
<span class="status-pill">
● System Online
</span>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="main-title">
    Explainable Financial
    <span class="green-text">
        Fraud Detection
    </span>
</div>

<div class="main-subtitle">
    Graph-based blockchain transaction intelligence powered
    by Graph Convolutional Networks and GNNExplainer.
</div>

<span class="tech-badge">
GCN
</span>

<span class="tech-badge">
PyTorch Geometric
</span>

<span class="tech-badge">
GNNExplainer
</span>

<span class="tech-badge">
Elliptic Dataset
</span>
    """,
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
<div class="dashboard-card">

<div class="card-icon">
◉
</div>

<div class="card-label">
TRANSACTIONS
</div>

<div class="card-number">
{NUM_NODES:,}
</div>

<div class="card-detail">
Nodes in the Elliptic transaction graph
</div>

</div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
<div class="dashboard-card">

<div class="card-icon">
⌁
</div>

<div class="card-label">
GRAPH EDGES
</div>

<div class="card-number">
{NUM_EDGES:,}
</div>

<div class="card-detail">
Transaction-to-transaction relationships
</div>

</div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
<div class="dashboard-card">

<div class="card-icon">
⬡
</div>

<div class="card-label">
MODEL
</div>

<div class="card-number">
GCN
</div>

<div class="card-detail">
Graph Convolutional Network
</div>

</div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
<div class="dashboard-card">

<div class="card-icon">
✦
</div>

<div class="card-label">
EXPLAINABILITY
</div>

<div class="card-number">
XAI
</div>

<div class="card-detail">
Local explanations using GNNExplainer
</div>

</div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns(
        [1.5, 1],
        gap="large"
    )

    with left:

        st.markdown(
            """
<div class="section-title">
Fraud Investigation Pipeline
</div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            """
Select a blockchain transaction and allow the GCN to analyze
both its transaction features and its surrounding graph
structure.

The prediction can then be interpreted using GNNExplainer
to identify important neighboring transactions, graph edges,
and transaction features.
            """
        )

    with right:

        st.markdown(
            """
<div class="section-title">
Model Status
</div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "✓ GCN model trained"
        )

        st.success(
            "✓ Transaction graph prepared"
        )

        st.success(
            "✓ GNNExplainer connected"
        )


# =========================================================
# TRANSACTION ANALYZER
# =========================================================

elif page == "Transaction Analyzer":

    st.markdown(
        """
<div class="section-title">
Transaction Investigation
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="analyzer-panel">

<strong>Investigate a blockchain transaction</strong>

<br><br>

The Graph Convolutional Network analyzes the selected
transaction using both its transaction features and its
connections with neighboring transactions.

</div>
        """,
        unsafe_allow_html=True
    )


    if not model_loaded:

        st.error(
            "The trained GCN could not be loaded."
        )

        st.code(
            model_error
        )

        st.stop()


    # -----------------------------------------------------
    # INPUT
    # -----------------------------------------------------

    input_col1, input_col2 = st.columns(
        [3, 1]
    )

    with input_col1:

        node_index = st.number_input(
            "Node Index",
            min_value=0,
            max_value=graph_data.num_nodes - 1,
            step=1,
            value=166967,
            help=(
                "Internal node index used by the "
                "processed PyTorch Geometric graph."
            )
        )

    with input_col2:

        st.write("")
        st.write("")

        analyze_button = st.button(
            "Analyze Transaction",
            use_container_width=True,
            type="primary"
        )


    # -----------------------------------------------------
    # ANALYZE
    # -----------------------------------------------------

    if analyze_button:

        result = analyze_node(
            int(node_index),
            graph_data,
            probabilities,
            predictions
        )

        st.session_state.analysis_result = result
        st.session_state.analyzed_node = int(
            node_index
        )

        # Remove previous explanation if a new node is chosen
        st.session_state.explanation_result = None


    # -----------------------------------------------------
    # SHOW ANALYSIS
    # -----------------------------------------------------

    if st.session_state.analysis_result is not None:

        result = st.session_state.analysis_result

        analyzed_node = st.session_state.analyzed_node

        predicted_class = result[
            "predicted_class"
        ]

        confidence = result[
            "confidence"
        ]

        true_label = result[
            "true_label"
        ]

        class_probabilities = result[
            "class_probabilities"
        ]


        # -------------------------------------------------
        # LABEL MAPPING
        # -------------------------------------------------

        label_names = {
            0: "LICIT",
            1: "ILLICIT"
        }

        predicted_label = label_names.get(
            predicted_class,
            "UNKNOWN"
        )

        if true_label is not None:

            actual_label = label_names.get(
                true_label,
                "UNKNOWN"
            )

        else:

            actual_label = "UNKNOWN"


        licit_probability = (
            class_probabilities[0]
        )

        fraud_probability = (
            class_probabilities[1]
        )


        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        if fraud_probability >= 0.80:

            risk_level = "HIGH RISK"

        elif fraud_probability >= 0.50:

            risk_level = "MEDIUM RISK"

        else:

            risk_level = "LOW RISK"


        # -------------------------------------------------
        # RISK ASSESSMENT
        # -------------------------------------------------

        st.markdown(
            """
<div class="section-title">
Risk Assessment
</div>
            """,
            unsafe_allow_html=True
        )


        if predicted_class == 1:

            st.markdown(
                f"""
<div style="
    background: linear-gradient(135deg, #2A0D12, #17090C);
    border: 1px solid #7F1D1D;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 20px;
">

<div style="
    color:#FCA5A5;
    font-size:0.85rem;
    font-weight:700;
    letter-spacing:1px;
">
{risk_level}
</div>

<div style="
    color:#F87171;
    font-size:2.2rem;
    font-weight:800;
    margin-top:5px;
">
⚠ ILLICIT TRANSACTION
</div>

<div style="
    color:#CBD5E1;
    font-size:1rem;
    margin-top:10px;
">
Fraud probability:
<strong style="color:#F8FAFC;">
{fraud_probability * 100:.2f}%
</strong>
</div>

</div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
<div style="
    background: linear-gradient(135deg, #0B2317, #07150E);
    border: 1px solid #166534;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 20px;
">

<div style="
    color:#86EFAC;
    font-size:0.85rem;
    font-weight:700;
    letter-spacing:1px;
">
{risk_level}
</div>

<div style="
    color:#4ADE80;
    font-size:2.2rem;
    font-weight:800;
    margin-top:5px;
">
✓ LICIT TRANSACTION
</div>

<div style="
    color:#CBD5E1;
    font-size:1rem;
    margin-top:10px;
">
Legitimate probability:
<strong style="color:#F8FAFC;">
{licit_probability * 100:.2f}%
</strong>
</div>

</div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )

        with result_col1:

            st.metric(
                "Model Prediction",
                predicted_label
            )

        with result_col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        with result_col3:

            st.metric(
                "Actual Label",
                actual_label
            )


        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        st.markdown(
            """
<div class="section-title">
Prediction Probabilities
</div>
            """,
            unsafe_allow_html=True
        )

        prob_col1, prob_col2 = (
            st.columns(2)
        )

        with prob_col1:

            st.metric(
                "Licit Probability",
                f"{licit_probability * 100:.2f}%"
            )

        with prob_col2:

            st.metric(
                "Illicit Probability",
                f"{fraud_probability * 100:.2f}%"
            )


        # -------------------------------------------------
        # TRANSACTION INFO
        # -------------------------------------------------

        st.markdown(
            """
<div class="section-title">
Transaction Information
</div>
            """,
            unsafe_allow_html=True
        )

        info_col1, info_col2, info_col3 = (
            st.columns(3)
        )

        with info_col1:

            st.metric(
                "Node Index",
                f"{analyzed_node:,}"
            )

        with info_col2:

            st.metric(
                "Input Features",
                graph_data.x.shape[1]
            )

        with info_col3:

            st.metric(
                "Graph Edges",
                graph_data.edge_index.shape[1]
            )


        # -------------------------------------------------
        # EXPLANATION SECTION
        # -------------------------------------------------

        st.markdown(
            """
<div class="section-title">
Explain This Prediction
</div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
<div class="analyzer-panel">

GNNExplainer analyzes which transaction relationships and
input features contributed most strongly to the GCN's
decision for this transaction.

</div>
            """,
            unsafe_allow_html=True
        )

        explain_button = st.button(
            "Generate GNN Explanation",
            use_container_width=True
        )


        # -------------------------------------------------
        # RUN EXPLAINER
        # -------------------------------------------------

        if explain_button:

            with st.spinner(
                "GNNExplainer is analyzing the transaction graph..."
            ):

                explanation = explain_node(
                    explainer,
                    graph_data,
                    int(analyzed_node)
                )

                top_edges = get_top_edges(
                    explanation,
                    graph_data.edge_index,
                    top_k=20
                )
                
                visualization_edges = (
                    get_visualization_edges(
                        explanation,
                        graph_data.edge_index,
                        selected_node=int(
                            analyzed_node
                        ),
                        direct_k=5,
                        additional_k=10
                    )
                )

                top_features = get_top_features(
                    explanation,
                    int(analyzed_node),
                    top_k=10,
                    num_nodes=graph_data.num_nodes
                )

                important_neighbors = (
                    get_important_neighbors(
                        int(analyzed_node),
                        top_edges,
                        graph_data,
                        top_k=5
                    )
                )

                st.session_state.explanation_result = {
                    "top_edges": top_edges,
                    "top_features": top_features,
                    "important_neighbors": important_neighbors,
                    "visualization_edges": visualization_edges
                }


        # -------------------------------------------------
        # DISPLAY EXPLANATION
        # -------------------------------------------------

        if (
            st.session_state.explanation_result
            is not None
        ):

            explanation_result = (
                st.session_state.explanation_result
            )

            top_edges = explanation_result[
                "top_edges"
            ]

            top_features = explanation_result[
                "top_features"
            ]

            important_neighbors = (
                explanation_result[
                    "important_neighbors"
                ]
            )
            
            visualization_edges = (
                explanation_result[
                    "visualization_edges"
                ]
            )
        
            # =============================================
            # LOCAL TRANSACTION NETWORK
            # =============================================

            st.markdown(
                """
<div class="section-title">
Local Transaction Network
</div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
<div class="analyzer-panel">

This graph visualizes the most influential transaction
relationships identified by GNNExplainer.

<strong>Orange</strong> = selected transaction &nbsp;&nbsp;
<strong>Red</strong> = illicit &nbsp;&nbsp;
<strong>Green</strong> = licit &nbsp;&nbsp;
<strong>Gray</strong> = unknown

<br><br>

Thicker edges indicate stronger importance in the
GNNExplainer explanation.

</div>
                """,
                unsafe_allow_html=True
            )


            network = create_local_network(
                selected_node=int(
                    analyzed_node
                ),
                top_edges=visualization_edges,
                graph_data=graph_data,
                max_edges=15
            )


            network_html = network.generate_html()

            components.html(
                network_html,
                height=620,
                scrolling=False
            )
            
            legend_col1, legend_col2, legend_col3, legend_col4 = (
                st.columns(4)
            )

            with legend_col1:
                st.caption("🟠 Selected transaction")

            with legend_col2:
                st.caption("🔴 Illicit")

            with legend_col3:
                st.caption("🟢 Licit")

            with legend_col4:
                st.caption("⚪ Unknown")


            # =============================================
            # IMPORTANT NEIGHBORS
            # =============================================

            st.markdown(
                """
<div class="section-title">
Influential Neighboring Transactions
</div>
                """,
                unsafe_allow_html=True
            )


            if important_neighbors:

                neighbor_columns = st.columns(
                    len(
                        important_neighbors
                    )
                )

                for column, neighbor in zip(
                    neighbor_columns,
                    important_neighbors
                ):

                    with column:

                        label = (
                            neighbor["label"]
                            if neighbor["label"]
                            is not None
                            else "Unknown"
                        )

                        st.metric(
                            f"Node {neighbor['node']}",
                            label,
                            (
                                "Importance "
                                f"{neighbor['importance']:.4f}"
                            )
                        )

            else:

                st.info(
                    "No directly connected influential "
                    "neighbors were found among the "
                    "highest-ranked edges."
                )


            # =============================================
            # IMPORTANT EDGES
            # =============================================

            st.markdown(
                """
<div class="section-title">
Most Influential Graph Connections
</div>
                """,
                unsafe_allow_html=True
            )


            edge_data = []

            for rank, edge in enumerate(
                top_edges[:10],
                start=1
            ):

                edge_data.append(
                    {
                        "Rank": rank,
                        "Source Node": edge[
                            "source"
                        ],
                        "Target Node": edge[
                            "target"
                        ],
                        "Importance": round(
                            edge[
                                "importance"
                            ],
                            4
                        )
                    }
                )


            edge_df = pd.DataFrame(
                edge_data
            )

            st.dataframe(
                edge_df,
                use_container_width=True,
                hide_index=True
            )


            # =============================================
            # FEATURES
            # =============================================

            st.markdown(
                """
<div class="section-title">
Top Influential Features
</div>
                """,
                unsafe_allow_html=True
            )


            feature_data = []

            for feature in top_features:

                feature_data.append(
                    {
                        "Feature": (
                            f"Feature "
                            f"{feature['feature']}"
                        ),
                        "Importance": (
                            feature[
                                "importance"
                            ]
                        )
                    }
                )


            feature_df = pd.DataFrame(
                feature_data
            )


            feature_df = (
                feature_df
                .sort_values(
                    "Importance",
                    ascending=True
                )
            )


            st.bar_chart(
                feature_df,
                x="Feature",
                y="Importance",
                horizontal=True,
                use_container_width=True
            )


            # =============================================
            # EXPLANATION SUMMARY
            # =============================================

            if (
                important_neighbors
                and top_features
            ):

                strongest_neighbor = (
                    important_neighbors[0]
                )

                strongest_feature = max(
                    top_features,
                    key=lambda x:
                    x["importance"]
                )

                st.markdown(
                    """
<div class="section-title">
Explanation Summary
</div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    f"""
For **Node {analyzed_node}**, GNNExplainer identified
**Node {strongest_neighbor['node']}** as one of the most
influential directly connected transactions.

The strongest identified transaction attribute was
**Feature {strongest_feature['feature']}**.

These graph relationships and transaction attributes
contributed most strongly to the GCN's prediction for
this node.
                    """
                )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        """
<div class="section-title">
GCN Model Performance
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="analyzer-panel">

This section will show the evaluation results of your
trained GCN model.

The focus will be on fraud-detection metrics rather than
accuracy alone.

</div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Accuracy",
            "Pending"
        )

    with col2:

        st.metric(
            "Precision",
            "Pending"
        )

    with col3:

        st.metric(
            "Recall",
            "Pending"
        )

    with col4:

        st.metric(
            "F1 Score",
            "Pending"
        )

    st.info(
        """
Next we can replace these placeholders with the exact
metrics produced by your trained GCN and add the confusion
matrix.
        """
    )


# =========================================================
# EXPLAINABILITY PAGE
# =========================================================

elif page == "Explainability":

    st.markdown(
        """
<div class="section-title">
GNNExplainer
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="analyzer-panel">

<strong>Explainable AI Layer</strong>

<br><br>

GNNExplainer identifies which graph relationships and
node attributes influenced the GCN's prediction.

This gives the fraud-detection model a local explanation
layer instead of presenting predictions as a black box.

</div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
<div class="dashboard-card">

<div class="card-icon">
⌁
</div>

<div class="card-label">
GRAPH EXPLANATION
</div>

<div class="card-number">
Edges
</div>

<div class="card-detail">
Important neighboring transactions and graph connections
are identified using edge importance masks.
</div>

</div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
<div class="dashboard-card">

<div class="card-icon">
▥
</div>

<div class="card-label">
FEATURE EXPLANATION
</div>

<div class="card-number">
Features
</div>

<div class="card-detail">
Important transaction attributes are identified using
feature importance masks.
</div>

</div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.info(
        """
Use the **Transaction Analyzer** page to select a node,
run the GCN prediction, and generate its local GNNExplainer
explanation.
        """
    )