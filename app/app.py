import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------------------------------------------------------
# Config & Paths
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Network Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "results" / "reports"
FIGURES_DIR = BASE_DIR / "results" / "figures"

FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "SYN Flag Count", "ACK Flag Count", "Average Packet Size",
]

# ----------------------------------------------------------------------------
# Cached loaders — model & scaler load once; test data loads once
# ----------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "random_forest.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.pkl")

@st.cache_data
def load_test_data():
    X_test = joblib.load(MODEL_DIR / "X_test.pkl")
    y_test = joblib.load(MODEL_DIR / "y_test.pkl")
    return X_test, y_test

@st.cache_data
def load_metrics():
    return pd.read_csv(REPORTS_DIR / "metrics.csv")

@st.cache_data
def load_classification_report():
    with open(REPORTS_DIR / "classification_report.txt") as f:
        return f.read()

model = load_model()
scaler = load_scaler()
X_test, y_test = load_test_data()
metrics_df = load_metrics()
report_text = load_classification_report()

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------

st.sidebar.title("🛡️ Network IDS")
st.sidebar.caption("CICIDS2017 · Random Forest Classifier")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Dataset", "Model Performance", "Live Prediction"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model**")
st.sidebar.write(f"Type: `RandomForestClassifier`")
st.sidebar.write(f"Trees: `{model.n_estimators}`")
st.sidebar.write(f"Max depth: `{model.max_depth}`")
st.sidebar.markdown("---")
st.sidebar.caption("Artifacts loaded from local `models/` and `results/` folders — nothing recomputed from raw CSVs.")

# ----------------------------------------------------------------------------
# Shared derived values
# ----------------------------------------------------------------------------

n_total = len(y_test)
n_normal = int((y_test == 0).sum())
n_anomaly = int((y_test == 1).sum())

acc = metrics_df.loc[metrics_df["Metric"] == "Accuracy", "Score"].values[0]
prec = metrics_df.loc[metrics_df["Metric"] == "Precision", "Score"].values[0]
rec = metrics_df.loc[metrics_df["Metric"] == "Recall", "Score"].values[0]
f1 = metrics_df.loc[metrics_df["Metric"] == "F1 Score", "Score"].values[0]

# ============================================================================
# PAGE: Overview
# ============================================================================

if page == "Overview":
    st.title("Network Intrusion Detection — Dashboard")
    st.caption("CICIDS2017 · Binary flow classification · Normal vs Anomaly")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Precision", f"{prec:.2%}")
    c3.metric("Recall", f"{rec:.2%}")
    c4.metric("F1 Score", f"{f1:.2%}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Test Set Class Balance")
        pie_df = pd.DataFrame({
            "Class": ["Normal", "Anomaly"],
            "Count": [n_normal, n_anomaly],
        })
        fig = px.pie(
            pie_df, names="Class", values="Count",
            color="Class",
            color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"},
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("At a Glance")
        st.write(f"**Total test flows:** {n_total:,}")
        st.write(f"**Normal flows:** {n_normal:,}")
        st.write(f"**Anomalous flows:** {n_anomaly:,}")
        st.write(f"**Features used:** {len(FEATURES)}")
        st.write(f"**Model:** Random Forest (balanced class weights)")
        st.info(
            "This model classifies network flow records as either **Normal** "
            "traffic or an **Anomaly** (attack / intrusion), based on flow-level "
            "statistics such as packet size, duration, and flag counts."
        )

# ============================================================================
# PAGE: Dataset
# ============================================================================

elif page == "Dataset":
    st.title("Dataset Overview")
    st.caption("Held-out test split — loaded directly from saved artifacts, no raw CSV reprocessing")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{n_total:,}")
    c2.metric("Normal", f"{n_normal:,}")
    c3.metric("Anomaly", f"{n_anomaly:,}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Normal vs Anomaly")
        bar_df = pd.DataFrame({
            "Class": ["Normal", "Anomaly"],
            "Count": [n_normal, n_anomaly],
        })
        fig = px.bar(
            bar_df, x="Class", y="Count", color="Class", text="Count",
            color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Features Used by the Model")
        st.dataframe(
            pd.DataFrame({"#": range(1, len(FEATURES) + 1), "Feature": FEATURES}),
            hide_index=True, use_container_width=True, height=380,
        )

    st.markdown("---")
    st.subheader("Sample of Test Records")
    n_show = st.slider("Rows to preview", 5, 50, 10)
    sample_df = pd.DataFrame(X_test[:n_show], columns=FEATURES)
    sample_df.insert(0, "Actual", np.where(y_test[:n_show] == 0, "Normal", "Anomaly"))
    st.dataframe(sample_df, use_container_width=True)

# ============================================================================
# PAGE: Model Performance
# ============================================================================

elif page == "Model Performance":
    st.title("Model Performance")
    st.caption("Evaluated on the held-out test split")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1 Score", f"{f1:.4f}")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Confusion Matrix")
        cm_path = FIGURES_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_column_width=True)
        else:
            st.warning("confusion_matrix.png not found.")

    with col2:
        st.subheader("Classification Report")
        st.code(report_text, language="text")

    st.markdown("---")
    st.subheader("Top 15 Feature Importances")
    importances = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(15)

    fig = px.bar(
        importances.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: Live Prediction
# ============================================================================

elif page == "Live Prediction":
    st.title("Live Prediction")
    st.caption("Enter network flow statistics manually, or load a random test-set sample")

    if "form_values" not in st.session_state:
        idx0 = 0
        st.session_state.form_values = {f: float(X_test[idx0][i]) for i, f in enumerate(FEATURES)}
        st.session_state.form_actual = "Normal" if y_test[idx0] == 0 else "Anomaly"

    top_col1, top_col2 = st.columns([1, 3])
    with top_col1:
        if st.button("🎲 Load random test sample"):
            idx = np.random.randint(0, len(X_test))
            st.session_state.form_values = {f: float(X_test[idx][i]) for i, f in enumerate(FEATURES)}
            st.session_state.form_actual = "Normal" if y_test[idx] == 0 else "Anomaly"
    with top_col2:
        st.caption(f"Loaded sample's true label: **{st.session_state.form_actual}** "
                   f"(hidden from the model — for your own comparison only)")

    with st.form("prediction_form"):
        st.markdown("**Flow Features**")
        cols = st.columns(3)
        input_values = {}
        for i, feature in enumerate(FEATURES):
            col = cols[i % 3]
            input_values[feature] = col.number_input(
                feature,
                value=st.session_state.form_values[feature],
                format="%.4f",
            )
        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        x = np.array([[input_values[f] for f in FEATURES]])
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)[0]
        proba = model.predict_proba(x_scaled)[0]

        st.markdown("---")
        st.subheader("Result")

        result_col, proba_col = st.columns([1, 2])

        with result_col:
            if pred == 1:
                st.error("🚨 ANOMALY DETECTED")
            else:
                st.success("✅ NORMAL TRAFFIC")
            st.write(f"Predicted class: **{'Anomaly' if pred == 1 else 'Normal'}**")
            st.write(f"True label of loaded sample: **{st.session_state.form_actual}**")

        with proba_col:
            proba_df = pd.DataFrame({
                "Class": ["Normal", "Anomaly"],
                "Probability": proba,
            })
            fig = px.bar(
                proba_df, x="Class", y="Probability", color="Class", text="Probability",
                color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"},
                range_y=[0, 1],
            )
            fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
