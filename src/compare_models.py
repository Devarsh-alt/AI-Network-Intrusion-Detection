import time
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import REPORTS_DIR, HAND_PICKED_FEATURES
from preprocessing import preprocess, feature_columns


# Rows kept per attack category in the model comparison, so that slow models
# (K-Nearest Neighbours) finish in reasonable time
SAMPLE_PER_CATEGORY = 60000


def get_models():

    return {
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
    }


# Fit one model and measure accuracy, F1 and timing

def score_model(name, model, X_train, y_train, X_test, y_test):

    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    predict_seconds = time.time() - start

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Train Seconds": round(train_seconds, 2),
        "Predict Seconds": round(predict_seconds, 2)
    }

    # F1 of every class, so weak classes are visible in the comparison
    classes = sorted(set(y_test))

    for label, score in zip(
        classes,
        f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    ):
        row[f"F1 {label}"] = score

    print(
        f"{name:<42} accuracy={row['Accuracy']:.4f} "
        f"macro_f1={row['Macro F1']:.4f} train={train_seconds:.1f}s"
    )

    return row


def scaled(classifier, *steps):

    return Pipeline(
        [("scaler", StandardScaler())]
        + list(steps)
        + [("classifier", classifier)]
    )


# Compare classifiers on a class-capped sample (all features)

def compare_classifiers(df, features):

    sample = pd.concat([
        group.sample(
            min(len(group), SAMPLE_PER_CATEGORY),
            random_state=42
        )
        for _, group in df.groupby("Category")
    ])

    train_df, test_df = train_test_split(
        sample,
        test_size=0.25,
        random_state=42,
        stratify=sample["Category"]
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (class-capped sample, all features)")
    print("=" * 70)

    print(f"\nTraining samples: {len(train_df):,}")
    print(f"Testing samples : {len(test_df):,}\n")

    rows = [
        score_model(
            name,
            scaled(classifier),
            train_df[features],
            train_df["Category"],
            test_df[features],
            test_df["Category"]
        )
        for name, classifier in get_models().items()
    ]

    pd.DataFrame(rows).to_csv(
        REPORTS_DIR / "cicids_model_comparison.csv",
        index=False
    )


# Compare feature-selection methods with Random Forest on the full dataset,
# using the same train/test split as train_model.py

def compare_feature_sets(df, features):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Category"]
    )

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPARISON (Random Forest, full dataset)")
    print("=" * 70)

    print(f"\nTraining samples: {len(train_df):,}")
    print(f"Testing samples : {len(test_df):,}\n")

    def forest():
        return get_models()["Random Forest"]

    rows = []

    def run(name, columns, *steps):

        model = scaled(forest(), *steps)

        row = score_model(
            name,
            model,
            train_df[columns],
            train_df["Category"],
            test_df[columns],
            test_df["Category"]
        )

        row = {"Feature Set": row.pop("Model"), "Features": len(columns), **row}

        if steps:
            row["Features"] = steps[0][1].n_components

        rows.append(row)

        return model

    all_model = run(f"All features ({len(features)})", features)

    # Rank features by the importance the all-features model gives them

    top_features = (
        pd.Series(
            all_model.named_steps["classifier"].feature_importances_,
            index=features
        )
        .sort_values(ascending=False)
        .head(20)
        .index
        .tolist()
    )

    run("Top 20 by Random Forest importance", top_features)

    run("Hand-picked (20)", HAND_PICKED_FEATURES)

    run(
        "Hand-picked without Destination Port (19)",
        [f for f in HAND_PICKED_FEATURES if f != "Destination Port"]
    )

    run(
        "PCA (20 components)",
        features,
        ("pca", PCA(n_components=20, random_state=42))
    )

    pd.DataFrame(rows).to_csv(
        REPORTS_DIR / "cicids_feature_selection.csv",
        index=False
    )

    pd.DataFrame({"Feature": top_features}).to_csv(
        REPORTS_DIR / "cicids_top_features.csv",
        index=False
    )


def compare():

    df = preprocess()

    features = feature_columns(df)

    compare_classifiers(df, features)

    compare_feature_sets(df, features)


# Run

if __name__ == "__main__":

    compare()

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 70)
