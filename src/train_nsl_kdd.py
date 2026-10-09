import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    MODEL_DIR,
    REPORTS_DIR,
    NSL_CATEGORICAL,
    NSL_CATEGORIES
)
from preprocessing import load_nsl_kdd
from evaluate import evaluate_model, BINARY_NAMES
from compare_models import get_models, score_model


NON_FEATURES = ["label", "difficulty", "Category", "Target"]


# One-hot encode the categorical columns and scale the numeric ones

def build_pipeline(classifier, numeric_columns):

    encoder = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NSL_CATEGORICAL
            ),
            ("numeric", StandardScaler(), numeric_columns)
        ],
        sparse_threshold=0
    )

    return Pipeline([
        ("encoder", encoder),
        ("classifier", classifier)
    ])


# Train on KDDTrain+ and test on KDDTest+ (the standard NSL-KDD protocol)

def train():

    train_df, test_df = load_nsl_kdd()

    print("=" * 70)
    print("NSL-KDD")
    print("=" * 70)

    print(f"\nTraining samples: {len(train_df):,}")
    print(f"Testing samples : {len(test_df):,}")

    pd.DataFrame({
        "Train": train_df["Category"].value_counts(),
        "Test": test_df["Category"].value_counts()
    }).reindex(NSL_CATEGORIES).rename_axis("Category").reset_index().to_csv(
        REPORTS_DIR / "nsl_kdd_class_distribution.csv",
        index=False
    )

    X_train = train_df.drop(columns=NON_FEATURES)
    X_test = test_df.drop(columns=NON_FEATURES)

    numeric_columns = [
        column
        for column in X_train.columns
        if column not in NSL_CATEGORICAL
    ]

    # Compare all models on the multi-class task

    rows = []
    fitted = {}

    for name, classifier in get_models().items():

        model = build_pipeline(classifier, numeric_columns)

        rows.append(
            score_model(
                name,
                model,
                X_train,
                train_df["Category"],
                X_test,
                test_df["Category"]
            )
        )

        fitted[name] = model

    pd.DataFrame(rows).to_csv(
        REPORTS_DIR / "nsl_kdd_model_comparison.csv",
        index=False
    )

    # Save and fully evaluate the Random Forest models

    multiclass_model = fitted["Random Forest"]

    binary_model = build_pipeline(
        get_models()["Random Forest"],
        numeric_columns
    ).fit(X_train, train_df["Target"])

    joblib.dump(binary_model, MODEL_DIR / "nsl_kdd_binary.pkl", compress=3)
    joblib.dump(multiclass_model, MODEL_DIR / "nsl_kdd_multiclass.pkl", compress=3)
    joblib.dump(test_df, MODEL_DIR / "nsl_kdd_test.pkl", compress=3)

    evaluate_model(
        binary_model,
        X_test,
        test_df["Target"],
        [0, 1],
        BINARY_NAMES,
        "nsl_kdd_binary",
        "NSL-KDD Binary"
    )

    evaluate_model(
        multiclass_model,
        X_test,
        test_df["Category"],
        NSL_CATEGORIES,
        NSL_CATEGORIES,
        "nsl_kdd_multiclass",
        "NSL-KDD Multi-class"
    )


# Run

if __name__ == "__main__":

    train()

    print("\n" + "=" * 70)
    print("NSL-KDD TRAINING COMPLETE")
    print("=" * 70)
