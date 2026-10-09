import json
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from config import (
    MODEL_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    CATEGORIES,
    NSL_CATEGORIES
)


BINARY_NAMES = ["Normal", "Attack"]


# Evaluate one model and save its report, metrics and confusion matrix

def evaluate_model(model, X_test, y_test, labels, names, prefix, title):

    print("\n" + "=" * 70)
    print(f"EVALUATION: {title}")
    print("=" * 70)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Binary: scores for the Attack class. Multi-class: macro average.
    average = "binary" if len(labels) == 2 else "macro"

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=None if average == "binary" else labels,
        average=average,
        zero_division=0
    )

    weighted_f1 = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )[2]

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f} ({average})")
    print(f"Recall    : {recall:.4f} ({average})")
    print(f"F1 Score  : {f1:.4f} ({average})")

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=names,
        digits=4,
        zero_division=0
    )

    print("\n" + report)

    with open(REPORTS_DIR / f"{prefix}_classification_report.txt", "w") as file:
        file.write(report)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    plt.figure(figsize=(1.2 * len(names) + 4, 1.0 * len(names) + 3))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{title} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(FIGURES_DIR / f"{prefix}_confusion_matrix.png", dpi=150)

    plt.close()

    metrics = {
        "title": title,
        "average": average,
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "labels": names,
        "per_class": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": cm.tolist()
    }

    with open(REPORTS_DIR / f"{prefix}_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print(f"Results saved with prefix: {prefix}")

    return metrics


# Re-evaluate the saved models on their saved test sets

def evaluate_saved_models():

    test = joblib.load(MODEL_DIR / "cicids_test.pkl")

    binary_model = joblib.load(MODEL_DIR / "cicids_binary.pkl")

    features = list(binary_model.feature_names_in_)

    evaluate_model(
        binary_model,
        test[features],
        test["Target"],
        [0, 1],
        BINARY_NAMES,
        "cicids_binary",
        "CICIDS2017 Binary"
    )

    evaluate_model(
        joblib.load(MODEL_DIR / "cicids_multiclass.pkl"),
        test[features],
        test["Category"],
        CATEGORIES,
        CATEGORIES,
        "cicids_multiclass",
        "CICIDS2017 Multi-class"
    )

    nsl_test_path = MODEL_DIR / "nsl_kdd_test.pkl"

    if nsl_test_path.exists():

        nsl_test = joblib.load(nsl_test_path)

        nsl_features = nsl_test.drop(
            columns=["label", "difficulty", "Category", "Target"]
        )

        evaluate_model(
            joblib.load(MODEL_DIR / "nsl_kdd_binary.pkl"),
            nsl_features,
            nsl_test["Target"],
            [0, 1],
            BINARY_NAMES,
            "nsl_kdd_binary",
            "NSL-KDD Binary"
        )

        evaluate_model(
            joblib.load(MODEL_DIR / "nsl_kdd_multiclass.pkl"),
            nsl_features,
            nsl_test["Category"],
            NSL_CATEGORIES,
            NSL_CATEGORIES,
            "nsl_kdd_multiclass",
            "NSL-KDD Multi-class"
        )


# Run

if __name__ == "__main__":

    evaluate_saved_models()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
