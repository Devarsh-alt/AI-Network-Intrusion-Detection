import inspect
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR / "src"))

import assistant
from alerts import (
    score_flows,
    explain_flow,
    incident_report,
    SEVERITY_ORDER,
)
from config import MODEL_DIR, REPORTS_DIR, CATEGORIES, NORMAL

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Network Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "Normal": "#2ecc71",
    "Attack": "#e74c3c",
    "DoS": "#e74c3c",
    "PortScan": "#f39c12",
    "Brute Force": "#9b59b6",
    "Botnet": "#34495e",
    "Web Attack": "#3498db",
    "Other": "#7f8c8d",
}

SEVERITY_COLORS = {
    "Low": "#95a5a6",
    "Medium": "#f1c40f",
    "High": "#e67e22",
    "Critical": "#c0392b",
}

# Knowledge base section that describes each attack category
ATTACK_SECTIONS = {
    "DoS": "Denial of Service (DoS and DDoS)",
    "PortScan": "Port Scan",
    "Brute Force": "Brute Force",
    "Botnet": "Botnet",
    "Web Attack": "Web Attack",
    "Other": "Other (Infiltration and Heartbleed)",
}

HISTORY_LIMIT = 50000

ALERT_COLUMNS = [
    "Time", "Predicted", "Severity", "Confidence", "Destination Port", "Actual Label",
]

# Keys from a local .env file (never committed) are made available to the app
env_path = BASE_DIR / ".env"

if env_path.exists():
    for line in env_path.read_text().splitlines():
        name, separator, value = line.partition("=")
        if separator and not name.strip().startswith("#"):
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))

# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "cicids_multiclass.pkl")

@st.cache_data
def load_demo_stream():
    return joblib.load(MODEL_DIR / "demo_stream.pkl")

@st.cache_data
def load_baseline():
    return json.loads((MODEL_DIR / "normal_baseline.json").read_text())

@st.cache_data
def load_metrics(prefix):
    path = REPORTS_DIR / f"{prefix}_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None

@st.cache_data
def load_table(name):
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None

@st.cache_resource
def load_knowledge():
    return assistant.KnowledgeBase()

if not (MODEL_DIR / "cicids_multiclass.pkl").exists():
    st.error(
        "No trained model found. Run `python src/train_model.py` first "
        "(see the README)."
    )
    st.stop()

model = load_model()

# The model's input columns (all CICFlowMeter flow features)
FEATURES = list(model.feature_names_in_)

demo_stream = load_demo_stream()
baseline = load_baseline()
knowledge = load_knowledge()

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()
    st.session_state.position = 0
    st.session_state.running = False
    st.session_state.chat = []

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title("🛡️ Network IDS")
st.sidebar.caption("CICIDS2017 · NSL-KDD · Random Forest")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Live Monitor", "Investigate", "Model Performance", "Assistant"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

threshold = st.sidebar.slider(
    "Alert threshold (attack probability)",
    min_value=0.30, max_value=0.99, value=0.50, step=0.01,
    help="A flow raises an alert when it is classified as an attack and its "
         "attack probability is at or above this value.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Deployed model**")
st.sidebar.write("Random Forest, multi-class")
st.sidebar.write(f"Features: `{len(FEATURES)}`")
st.sidebar.write(f"Classes: `{len(model.classes_)}`")

# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

# Older Streamlit versions size Plotly charts with use_container_width,
# newer ones with width
PLOTLY_WIDTH = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.plotly_chart).parameters
    else {"use_container_width": True}
)

def show_chart(fig, key=None):
    st.plotly_chart(fig, key=key, **PLOTLY_WIDTH)

def show_explanation(flow, key):
    """Prediction, class probabilities and feature contributions of one flow."""
    row = pd.DataFrame([flow[FEATURES]], columns=FEATURES).astype(float)
    probabilities = model.predict_proba(row)[0]
    predicted = model.classes_[probabilities.argmax()]

    result_col, proba_col = st.columns([1, 2])

    with result_col:
        if predicted == NORMAL:
            st.success("✅ NORMAL TRAFFIC")
        else:
            st.error(f"🚨 ATTACK DETECTED: {predicted}")
        st.write(f"Predicted class: **{predicted}**")
        st.write(f"Confidence: **{probabilities.max():.1%}**")
        if "Label" in flow.index:
            st.write(f"True label in dataset: **{flow['Label']}**")

    with proba_col:
        proba_df = pd.DataFrame({
            "Class": model.classes_, "Probability": probabilities,
        }).sort_values("Probability")
        fig = px.bar(
            proba_df, x="Probability", y="Class", orientation="h",
            color="Class", color_discrete_map=COLORS, range_x=[0, 1],
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(t=10, b=10))
        show_chart(fig, key=f"{key}_proba")

    st.markdown("**Why this prediction?**")
    st.caption(
        "Each feature is replaced by its typical value in normal traffic. "
        "The bar shows how much the predicted class probability drops, so "
        "longer bars mean the feature mattered more. This is an approximation."
    )

    explanation = explain_flow(model, flow, baseline)

    chart_col, table_col = st.columns([1, 1])

    with chart_col:
        fig = px.bar(
            explanation.sort_values("Contribution"),
            x="Contribution", y="Feature", orientation="h",
        )
        fig.update_traces(marker_color="#e74c3c")
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        show_chart(fig, key=f"{key}_explain")

    with table_col:
        st.dataframe(
            explanation.style.format({
                "Value": "{:,.2f}",
                "Typical Normal Value": "{:,.2f}",
                "Contribution": "{:+.3f}",
            }),
            hide_index=True, width="stretch",
        )

    if predicted != NORMAL:
        title = ATTACK_SECTIONS[predicted]
        section = next(c for c in knowledge.chunks if c["title"] == title)
        with st.expander(f"About this attack: {title}"):
            st.markdown(section["text"])


def confusion_chart(metrics, normalise, key):
    matrix = np.array(metrics["confusion_matrix"], dtype=float)
    if normalise:
        matrix = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig = px.imshow(
        matrix,
        x=metrics["labels"], y=metrics["labels"],
        text_auto=".1%" if normalise else ",.0f",
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Share" if normalise else "Flows"),
        aspect="auto",
    )
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    show_chart(fig, key=key)


def per_class_table(metrics):
    table = pd.DataFrame(metrics["per_class"]).T.loc[metrics["labels"]]
    table = table.rename(columns={
        "precision": "Precision", "recall": "Recall",
        "f1-score": "F1", "support": "Test Flows",
    })
    st.dataframe(
        table.style.format({
            "Precision": "{:.2%}", "Recall": "{:.2%}",
            "F1": "{:.2%}", "Test Flows": "{:,.0f}",
        }),
        width="stretch",
    )


def comparison_section(table, name_column, key):
    best = table.loc[table["Macro F1"].idxmax(), name_column]
    st.caption(f"Best macro F1: **{best}**")
    long = table.melt(
        id_vars=name_column, value_vars=["Accuracy", "Macro F1"],
        var_name="Metric", value_name="Score",
    )
    fig = px.bar(
        long, x="Score", y=name_column, color="Metric",
        barmode="group", orientation="h", range_x=[0, 1],
    )
    fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title=None)
    show_chart(fig, key=key)
    st.dataframe(
        table.style.format({
            "Accuracy": "{:.2%}", "Macro F1": "{:.2%}", "Weighted F1": "{:.2%}",
            "Train Seconds": "{:.1f}", "Predict Seconds": "{:.2f}",
            **{c: "{:.2%}" for c in table.columns if c.startswith("F1 ")},
        }),
        hide_index=True, width="stretch",
    )

# ============================================================================
# PAGE: Overview
# ============================================================================

if page == "Overview":
    st.title("AI-Based Network Intrusion Detection")
    st.caption("Machine-learning detection and classification of network attacks")

    cicids_binary = load_metrics("cicids_binary")
    cicids_multi = load_metrics("cicids_multiclass")
    nsl_binary = load_metrics("nsl_kdd_binary")
    nsl_multi = load_metrics("nsl_kdd_multiclass")

    st.subheader("CICIDS2017")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attack detection accuracy", f"{cicids_binary['accuracy']:.2%}")
    c2.metric("Attack detection F1", f"{cicids_binary['f1']:.2%}")
    c3.metric("Attack type accuracy", f"{cicids_multi['accuracy']:.2%}")
    c4.metric("Attack type macro F1", f"{cicids_multi['f1']:.2%}")

    if nsl_binary and nsl_multi:
        st.subheader("NSL-KDD")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Attack detection accuracy", f"{nsl_binary['accuracy']:.2%}")
        c2.metric("Attack detection F1", f"{nsl_binary['f1']:.2%}")
        c3.metric("Attack type accuracy", f"{nsl_multi['accuracy']:.2%}")
        c4.metric("Attack type macro F1", f"{nsl_multi['f1']:.2%}")
        st.caption(
            "NSL-KDD is tested on KDDTest+, which contains attack types that "
            "never appear in training, so its scores are much lower."
        )

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("CICIDS2017 Flows per Category")
        distribution = load_table("cicids_class_distribution.csv")
        counts = distribution.groupby("Category", as_index=False)["Count"].sum()
        fig = px.bar(
            counts, x="Category", y="Count", color="Category", text="Count",
            color_discrete_map=COLORS, log_y=True,
            category_orders={"Category": CATEGORIES},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_title="Flows (log scale)")
        show_chart(fig)

    with col2:
        st.subheader("What the System Does")
        st.markdown(
            "- **Detects** malicious network flows with a Random Forest model\n"
            "- **Classifies** them as DoS, Port Scan, Brute Force, Botnet, "
            "Web Attack or Other\n"
            "- **Monitors** a simulated live traffic stream and raises alerts "
            "with a severity level\n"
            "- **Explains** which features drove each prediction\n"
            "- **Answers questions** in plain English through the assistant"
        )
        st.info(
            "The Live Monitor replays held-out CICIDS2017 test flows. "
            "It does not capture packets from a real network."
        )

# ============================================================================
# PAGE: Live Monitor
# ============================================================================

elif page == "Live Monitor":
    st.title("Live Monitor")
    st.caption("Simulated traffic stream replayed from held-out CICIDS2017 test flows")

    b1, b2, b3, speed_col = st.columns([1, 1, 1, 3])

    if b1.button("▶ Start", width="stretch", disabled=st.session_state.running):
        st.session_state.running = True
        st.rerun()

    if b2.button("⏸ Pause", width="stretch", disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()

    if b3.button("↺ Reset", width="stretch"):
        st.session_state.history = pd.DataFrame()
        st.session_state.position = 0
        st.session_state.running = False
        st.rerun()

    flows_per_second = speed_col.slider("Flows per second", 10, 300, 60, step=10)

    def process_next_flows():
        start = st.session_state.position
        rows = [(start + i) % len(demo_stream) for i in range(flows_per_second)]
        batch = score_flows(model, demo_stream.iloc[rows], threshold)
        batch.insert(0, "Time", pd.Timestamp.now().floor("s"))
        st.session_state.history = pd.concat(
            [st.session_state.history, batch], ignore_index=True,
        ).tail(HISTORY_LIMIT)
        st.session_state.position = (start + flows_per_second) % len(demo_stream)
        return batch

    @st.fragment(run_every=1.0 if st.session_state.running else None)
    def monitor():
        batch = process_next_flows() if st.session_state.running else None

        history = st.session_state.history

        if history.empty:
            st.info("Press **Start** to begin replaying traffic.")
            return

        alerts = history[history["Alert"]]

        if batch is not None:
            batch_alerts = batch[batch["Alert"]]
            if batch_alerts.empty:
                st.success("No attacks detected in the last second.")
            else:
                top = batch_alerts["Predicted"].value_counts()
                st.error(
                    f"🚨 {len(batch_alerts)} alerts in the last second — "
                    + ", ".join(f"{name}: {count}" for name, count in top.items())
                )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flows processed", f"{len(history):,}")
        c2.metric("Alerts raised", f"{len(alerts):,}")
        c3.metric("Alert rate", f"{len(alerts) / len(history):.1%}")
        c4.metric(
            "Live accuracy",
            f"{(history['Predicted'] == history['Category']).mean():.2%}",
            help="Share of flows whose predicted category matches the dataset label.",
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Alerts per Second")
            recent = alerts[
                alerts["Time"] >= history["Time"].max() - pd.Timedelta(seconds=90)
            ]
            if recent.empty:
                st.caption("No alerts in the last 90 seconds.")
            else:
                timeline = (
                    recent.groupby(["Time", "Predicted"]).size().reset_index(name="Alerts")
                )
                fig = px.bar(
                    timeline, x="Time", y="Alerts", color="Predicted",
                    color_discrete_map=COLORS,
                )
                fig.update_layout(height=320, margin=dict(t=10, b=10), legend_title=None)
                show_chart(fig, key="timeline")

        with col2:
            st.subheader("Traffic Mix")
            mix = history["Predicted"].value_counts().reset_index()
            mix.columns = ["Class", "Flows"]
            fig = px.pie(
                mix, names="Class", values="Flows", color="Class",
                color_discrete_map=COLORS, hole=0.5,
            )
            fig.update_layout(height=320, margin=dict(t=10, b=10))
            show_chart(fig, key="mix")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Latest Alerts")
            st.dataframe(
                alerts.rename(columns={"Label": "Actual Label"})[ALERT_COLUMNS]
                .iloc[::-1].head(100).style.format({
                    "Time": "{:%H:%M:%S}",
                    "Confidence": "{:.1%}",
                    "Destination Port": "{:.0f}",
                }),
                hide_index=True, width="stretch", height=320,
            )

        with col2:
            st.subheader("Alerts by Severity")
            severity = (
                alerts["Severity"].value_counts()
                .reindex(SEVERITY_ORDER, fill_value=0).reset_index()
            )
            severity.columns = ["Severity", "Alerts"]
            fig = px.bar(
                severity, x="Severity", y="Alerts", color="Severity",
                color_discrete_map=SEVERITY_COLORS, text="Alerts",
            )
            fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
            show_chart(fig, key="severity")

        # Exports are offered while paused, so they are not rebuilt every second
        if not st.session_state.running:
            st.markdown("---")
            d1, d2 = st.columns(2)
            d1.download_button(
                "⬇ Download alert log (CSV)",
                alerts.drop(columns=["Alert"]).to_csv(index=False),
                file_name="alert_log.csv", mime="text/csv", width="stretch",
            )
            d2.download_button(
                "⬇ Download incident report (Markdown)",
                incident_report(history),
                file_name="incident_report.md", mime="text/markdown", width="stretch",
            )

    monitor()

# ============================================================================
# PAGE: Investigate
# ============================================================================

elif page == "Investigate":
    st.title("Investigate")
    st.caption("Explain an alert, test a single flow, or classify a CSV file")

    alert_tab, flow_tab, upload_tab = st.tabs(
        ["Alert from the monitor", "Single flow", "Upload CSV"]
    )

    with alert_tab:
        history = st.session_state.history
        alerts = history[history["Alert"]].iloc[::-1].head(200) if len(history) else history

        if alerts.empty:
            st.info("No alerts yet. Start the Live Monitor to generate some.")
        else:
            choice = st.selectbox(
                "Alert (most recent first)",
                alerts.index,
                format_func=lambda i: (
                    f"{alerts.loc[i, 'Time']:%H:%M:%S} · {alerts.loc[i, 'Predicted']} · "
                    f"{alerts.loc[i, 'Severity']} · port "
                    f"{int(alerts.loc[i, 'Destination Port'])} · "
                    f"{alerts.loc[i, 'Confidence']:.0%}"
                ),
            )
            show_explanation(alerts.loc[choice], key="alert")

    with flow_tab:
        if "flow" not in st.session_state:
            st.session_state.flow = demo_stream.iloc[0]
            st.session_state.flow_version = 0

        pick_col, info_col = st.columns([1, 3])

        category = pick_col.selectbox("Load a test flow of type", ["Any"] + CATEGORIES)

        if pick_col.button("🎲 Load random flow", width="stretch"):
            pool = demo_stream if category == "Any" else demo_stream[
                demo_stream["Category"] == category
            ]
            st.session_state.flow = pool.sample(1).iloc[0]
            # A new key resets the editor to the newly loaded values
            st.session_state.flow_version += 1

        info_col.caption(
            "Loaded flow's true label: "
            f"**{st.session_state.flow['Label']}** (not shown to the model). "
            "Edit any value in the table to see how the prediction changes."
        )

        edited = st.data_editor(
            pd.DataFrame({
                "Feature": FEATURES,
                "Value": st.session_state.flow[FEATURES].astype(float).values,
            }),
            disabled=["Feature"], hide_index=True, width="stretch", height=300,
            key=f"flow_editor_{st.session_state.flow_version}",
        )

        st.markdown("---")
        show_explanation(
            pd.Series(edited["Value"].values, index=FEATURES), key="flow",
        )

    with upload_tab:
        st.write(
            f"Upload a CSV of network flows. It must contain the {len(FEATURES)} "
            "flow feature columns produced by CICFlowMeter (the CICIDS2017 format)."
        )

        st.download_button(
            "⬇ Download a sample CSV (500 test flows)",
            demo_stream[FEATURES].sample(500, random_state=1).to_csv(index=False),
            file_name="sample_flows.csv", mime="text/csv",
        )

        uploaded = st.file_uploader("CSV file", type="csv")

        if uploaded is not None:
            flows = pd.read_csv(uploaded, encoding="latin-1", low_memory=False)
            flows.columns = flows.columns.str.strip()

            missing = [f for f in FEATURES if f not in flows.columns]

            if missing:
                st.error(
                    f"{len(missing)} required columns are missing: "
                    + ", ".join(missing[:10])
                    + (" ..." if len(missing) > 10 else "")
                )
            else:
                valid = (
                    flows.replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=FEATURES)
                )
                if len(valid) < len(flows):
                    st.warning(
                        f"{len(flows) - len(valid):,} rows with missing or "
                        "infinite values were skipped."
                    )

                scored = score_flows(model, valid, threshold)

                c1, c2, c3 = st.columns(3)
                c1.metric("Flows", f"{len(scored):,}")
                c2.metric("Alerts", f"{int(scored['Alert'].sum()):,}")
                c3.metric("Alert rate", f"{scored['Alert'].mean():.1%}")

                counts = scored["Predicted"].value_counts().reset_index()
                counts.columns = ["Class", "Flows"]
                fig = px.bar(
                    counts, x="Class", y="Flows", color="Class", text="Flows",
                    color_discrete_map=COLORS,
                )
                fig.update_layout(showlegend=False, height=320)
                show_chart(fig)

                result_columns = [
                    "Predicted", "Confidence", "Attack Probability", "Severity", "Alert",
                ]
                st.dataframe(
                    scored[result_columns + FEATURES].head(1000),
                    hide_index=True, width="stretch",
                )
                st.download_button(
                    "⬇ Download predictions (CSV)",
                    scored.to_csv(index=False),
                    file_name="predictions.csv", mime="text/csv",
                )

# ============================================================================
# PAGE: Model Performance
# ============================================================================

elif page == "Model Performance":
    st.title("Model Performance")

    c1, c2, c3 = st.columns([1, 1, 2])
    dataset = c1.selectbox("Dataset", ["CICIDS2017", "NSL-KDD"])
    task = c2.selectbox("Task", ["Attack type (multi-class)", "Attack detection (binary)"])

    prefix = (
        ("cicids" if dataset == "CICIDS2017" else "nsl_kdd")
        + ("_multiclass" if task.startswith("Attack type") else "_binary")
    )

    metrics = load_metrics(prefix)

    if metrics is None:
        st.warning("No results found for this model. Run the training scripts first.")
        st.stop()

    st.caption(
        f"Random Forest · {metrics['test_samples']:,} test samples · "
        f"precision, recall and F1 are "
        + ("for the Attack class" if metrics["average"] == "binary" else "macro averages")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall", f"{metrics['recall']:.2%}")
    c4.metric("F1 Score", f"{metrics['f1']:.2%}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Confusion Matrix")
        normalise = st.toggle("Show as share of each actual class", value=True)
        confusion_chart(metrics, normalise, key="confusion")

    with col2:
        st.subheader("Per-class Results")
        per_class_table(metrics)

    prefix_dataset = "cicids" if dataset == "CICIDS2017" else "nsl_kdd"

    comparison = load_table(f"{prefix_dataset}_model_comparison.csv")

    if comparison is not None:
        st.markdown("---")
        st.subheader("Model Comparison (multi-class)")
        if dataset == "CICIDS2017":
            st.caption(
                "Trained on a sample capped at 60,000 flows per category so that "
                "slow models finish, which means scores differ from the "
                "full-data model above."
            )
        comparison_section(comparison, "Model", key="models")

    if dataset == "CICIDS2017":
        selection = load_table("cicids_feature_selection.csv")
        if selection is not None:
            st.markdown("---")
            st.subheader("Feature Selection Comparison (Random Forest)")
            st.caption(
                "Full dataset, same train/test split as the deployed model. "
                "The per-class F1 columns show which attacks each feature set misses."
            )
            comparison_section(selection, "Feature Set", key="selection")

        importance = load_table("cicids_feature_importance.csv")
        if importance is not None:
            st.markdown("---")
            st.subheader("Top 20 Feature Importances of the Deployed Model")
            fig = px.bar(
                importance.head(20).sort_values("Importance"),
                x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Reds",
            )
            fig.update_layout(coloraxis_showscale=False, height=560)
            show_chart(fig)
    else:
        distribution = load_table("nsl_kdd_class_distribution.csv")
        if distribution is not None:
            st.markdown("---")
            st.subheader("Class Distribution")
            st.dataframe(distribution, hide_index=True)

# ============================================================================
# PAGE: Assistant
# ============================================================================

elif page == "Assistant":
    st.title("Assistant")
    st.caption(
        "Ask about the alerts in this session, attack types, features, "
        "the models or their results"
    )

    c1, c2, c3 = st.columns([1, 2, 2])

    provider = c1.selectbox("Provider", list(assistant.PROVIDERS))
    settings = assistant.PROVIDERS[provider]

    llm_model = c2.text_input("Model", value=settings["default_model"])

    api_key = os.environ.get(settings["key_variable"], "")

    if api_key:
        c3.text_input("API key", value=f"Loaded from {settings['key_variable']}", disabled=True)
    else:
        api_key = c3.text_input(
            "API key", type="password",
            help=f"Or set {settings['key_variable']} in a .env file in the project folder.",
        )

    if not api_key:
        st.info(
            "No API key set. Without one, the assistant shows the most relevant "
            "knowledge base passages instead of a written answer."
        )

    examples = [
        "How many alerts were raised and which attack type was most common?",
        "Which ports were targeted most by DoS alerts?",
        "What is a port scan and how should I respond?",
        "Why is NSL-KDD accuracy lower than CICIDS2017?",
    ]

    question = None

    if not st.session_state.chat:
        for column, example in zip(st.columns(len(examples)), examples):
            if column.button(example, width="stretch"):
                question = example

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("details"):
                with st.expander("Sources"):
                    st.markdown(message["details"])

    question = st.chat_input("Ask a question") or question

    if question:
        st.session_state.chat.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            tools_used = []

            if api_key:
                conversation = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat
                ]
                try:
                    with st.spinner("Thinking..."):
                        reply, sources, tools_used = assistant.answer(
                            assistant.get_client(provider, api_key),
                            llm_model,
                            conversation,
                            knowledge,
                            st.session_state.history,
                        )
                except Exception as error:
                    reply, sources = f"The request to {provider} failed: {error}", []
            else:
                sources = knowledge.search(question, k=2)
                reply = "\n\n".join(
                    f"**{chunk['title']}**\n\n{chunk['text']}" for chunk in sources
                ) or "Nothing relevant was found in the knowledge base."

            details = "\n".join(
                [f"- Tool: `{name}`" for name in dict.fromkeys(tools_used)]
                + [
                    f"- Knowledge: {title}"
                    for title in dict.fromkeys(
                        f"{chunk['source']} · {chunk['title']}" for chunk in sources
                    )
                ]
            )

            st.markdown(reply)
            if details:
                with st.expander("Sources"):
                    st.markdown(details)

        st.session_state.chat.append(
            {"role": "assistant", "content": reply, "details": details}
        )

    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()
