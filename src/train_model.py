import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    MODEL_DIR,
    REPORTS_DIR,
    CATEGORIES,
    NORMAL
)
from preprocessing import preprocess, feature_columns
from compare_models import get_models
from evaluate import evaluate_model, BINARY_NAMES


# Scaler + Random Forest in one pipeline, so the saved model takes raw features

def build_random_forest():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", get_models()["Random Forest"])
    ])


# Build a replayable traffic stream for the dashboard from held-out flows:
# stretches of normal traffic with bursts of one attack type mixed in

def build_demo_stream(test, size=20000, seed=42):

    rng = np.random.default_rng(seed)

    pools = {
        category: list(rng.permutation(group.index))
        for category, group in test.groupby("Category")
    }

    attacks = [c for c in CATEGORIES if c != NORMAL and c in pools]

    order = []

    def take(category, count):
        pool = pools[category]
        taken, pools[category] = pool[:count], pool[count:]
        return taken

    while len(order) < size:

        order += take(NORMAL, int(rng.integers(300, 900)))

        category = attacks[int(rng.integers(len(attacks)))]

        burst = take(category, int(rng.integers(60, 250)))
        burst += take(NORMAL, len(burst))

        order += list(rng.permutation(burst))

    return test.loc[order[:size]].reset_index(drop=True)


# Train binary and multi-class Random Forest models on CICIDS2017

def train():

    df = preprocess()

    features = feature_columns(df)

    df[["Label", "Category"]].value_counts().rename("Count").reset_index().to_csv(
        REPORTS_DIR / "cicids_class_distribution.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Category"]
    )

    print(f"\nTraining samples: {len(train_df):,}")
    print(f"Testing samples : {len(test_df):,}")

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST MODELS")
    print("=" * 70)

    print("\nTraining binary model (Normal vs Attack)...")

    binary_model = build_random_forest().fit(
        train_df[features],
        train_df["Target"]
    )

    print("Training multi-class model (attack categories)...")

    multiclass_model = build_random_forest().fit(
        train_df[features],
        train_df["Category"]
    )

    joblib.dump(binary_model, MODEL_DIR / "cicids_binary.pkl", compress=3)
    joblib.dump(multiclass_model, MODEL_DIR / "cicids_multiclass.pkl", compress=3)
    joblib.dump(test_df, MODEL_DIR / "cicids_test.pkl", compress=3)

    print(f"\nModels and test set saved to: {MODEL_DIR}")

    # Typical values of normal traffic, used to explain predictions

    baseline = train_df.loc[train_df["Category"] == NORMAL, features].median()

    with open(MODEL_DIR / "normal_baseline.json", "w") as file:
        json.dump(baseline.to_dict(), file, indent=2)

    joblib.dump(
        build_demo_stream(test_df),
        MODEL_DIR / "demo_stream.pkl",
        compress=3
    )

    pd.DataFrame({
        "Feature": features,
        "Importance": multiclass_model.named_steps["classifier"].feature_importances_
    }).sort_values("Importance", ascending=False).to_csv(
        REPORTS_DIR / "cicids_feature_importance.csv",
        index=False
    )

    evaluate_model(
        binary_model,
        test_df[features],
        test_df["Target"],
        [0, 1],
        BINARY_NAMES,
        "cicids_binary",
        "CICIDS2017 Binary"
    )

    evaluate_model(
        multiclass_model,
        test_df[features],
        test_df["Category"],
        CATEGORIES,
        CATEGORIES,
        "cicids_multiclass",
        "CICIDS2017 Multi-class"
    )


# Run

if __name__ == "__main__":

    train()

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
