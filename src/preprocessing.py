import pandas as pd
import numpy as np
from pathlib import Path


# Paths

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Load all CSV files

def load_all_data():

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    dataframes = []

    print("=" * 70)
    print("LOADING CICIDS2017 DATASET")
    print("=" * 70)

    for file in csv_files:

        print(f"\nLoading: {file.name}")

        df = pd.read_csv(
            file,
            low_memory=False
        )

        df.columns = df.columns.str.strip()

        print(f"Rows: {len(df):,}")

        dataframes.append(df)

    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    print("\nTotal rows:", f"{len(data):,}")
    print("Total columns:", len(data.columns))

    return data


# Basic cleaning

def clean_data(df):

    print("\n" + "=" * 70)
    print("BASIC DATA CLEANING")
    print("=" * 70)

    df.columns = df.columns.str.strip()

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"\nDuplicate rows removed: {before - after:,}")

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    missing_before = df.isnull().sum().sum()

    print(f"Missing/invalid values found: {missing_before:,}")

    df = df.dropna()

    missing_after = df.isnull().sum().sum()

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Missing values remaining: {missing_after:,}")

    return df


# Create binary target

def create_binary_label(df):

    print("\n" + "=" * 70)
    print("CREATING BINARY LABEL")
    print("=" * 70)

    df["Label"] = df["Label"].astype(str).str.strip()

    df["Target"] = (
        df["Label"]
        .apply(lambda x: 0 if x == "BENIGN" else 1)
    )

    print("\nBinary label distribution:")

    print(
        df["Target"]
        .value_counts()
        .sort_index()
        .rename({
            0: "Normal",
            1: "Anomaly"
        })
    )

    return df


# Feature selection

def select_features(df):

    print("\n" + "=" * 70)
    print("FEATURE SELECTION")
    print("=" * 70)

    selected_features = [
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "SYN Flag Count",
        "ACK Flag Count",
        "Average Packet Size"
    ]

    selected_features = [
        feature
        for feature in selected_features
        if feature in df.columns
    ]

    print(f"\nSelected features: {len(selected_features)}")

    for feature in selected_features:
        print(f"  - {feature}")

    X = df[selected_features].copy()

    y = df["Target"].copy()

    return X, y


# Main preprocessing pipeline

def preprocess():

    df = load_all_data()

    df = clean_data(df)

    df = create_binary_label(df)

    X, y = select_features(df)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


# Run

if __name__ == "__main__":

    X, y = preprocess()

    print("\nFirst 5 feature rows:")
    print(X.head())

    print("\nFirst 5 targets:")
    print(y.head())

    print("\nTarget distribution:")

    print(
        y.value_counts()
        .sort_index()
        .rename({
            0: "Normal",
            1: "Anomaly"
        })
    )