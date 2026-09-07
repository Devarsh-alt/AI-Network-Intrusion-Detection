import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess


# Paths

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Prepare data

def prepare_data():

    X, y = preprocess()

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Testing samples : {len(X_test):,}")

    print("\n" + "=" * 70)
    print("FEATURE SCALING")
    print("=" * 70)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print("\nScaling completed.")

    print(f"\nTraining feature shape: {X_train_scaled.shape}")
    print(f"Testing feature shape : {X_test_scaled.shape}")

    # Save scaler

    scaler_path = MODEL_DIR / "scaler.pkl"

    joblib.dump(
        scaler,
        scaler_path
    )

    print(f"\nScaler saved to: {scaler_path}")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )


# Train Random Forest

def train_random_forest():

    X_train, X_test, y_train, y_test = prepare_data()

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("\nNumber of trees : 100")
    print("Maximum depth   : 20")
    print("Class weight    : balanced")

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("\nRandom Forest training completed.")

    # Save model

    model_path = MODEL_DIR / "random_forest.pkl"

    joblib.dump(
        model,
        model_path
    )

    print(f"Model saved to: {model_path}")

    # Save test data

    X_test_path = MODEL_DIR / "X_test.pkl"
    y_test_path = MODEL_DIR / "y_test.pkl"

    joblib.dump(
        X_test,
        X_test_path
    )

    joblib.dump(
        y_test,
        y_test_path
    )

    print(f"Test features saved to: {X_test_path}")
    print(f"Test labels saved to: {y_test_path}")

    return model


# Run

if __name__ == "__main__":

    train_random_forest()

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)