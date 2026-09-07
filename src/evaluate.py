import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Paths

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"

FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Evaluate model

def evaluate_model():

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Load model

    model_path = MODEL_DIR / "random_forest.pkl"

    model = joblib.load(model_path)

    print("\nModel loaded successfully.")

    # Load test data

    X_test_path = MODEL_DIR / "X_test.pkl"
    y_test_path = MODEL_DIR / "y_test.pkl"

    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)

    print("Test data loaded successfully.")

    print(f"\nTesting samples: {len(X_test):,}")

    # Generate predictions

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    print("Predictions completed.")

    # Calculate metrics

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Display metrics

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Classification report

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Anomaly"
        ],
        zero_division=0
    )

    print("\n" + report)

    report_path = REPORTS_DIR / "classification_report.txt"

    with open(report_path, "w") as file:
        file.write(report)

    print(f"Report saved to: {report_path}")

    # Confusion matrix

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(cm)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=[
            "Normal",
            "Anomaly"
        ],
        yticklabels=[
            "Normal",
            "Anomaly"
        ]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest Confusion Matrix")

    plt.tight_layout()

    confusion_path = (
        FIGURES_DIR / "confusion_matrix.png"
    )

    plt.savefig(confusion_path)

    plt.close()

    print(f"\nConfusion matrix saved to: {confusion_path}")

    # Save metrics

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    metrics_path = REPORTS_DIR / "metrics.csv"

    metrics.to_csv(
        metrics_path,
        index=False
    )

    print(f"Metrics saved to: {metrics_path}")


# Run

if __name__ == "__main__":

    evaluate_model()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)