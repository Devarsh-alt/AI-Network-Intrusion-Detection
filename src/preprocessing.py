import pandas as pd
import numpy as np

from config import (
    RAW_DIR,
    NSL_DIR,
    LABEL_COLUMNS,
    NORMAL,
    NSL_COLUMNS,
    NSL_LABEL_MAP,
    cicids_category
)


# Load CICIDS2017 CSV files

def load_cicids():

    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    dataframes = []

    print("=" * 70)
    print("LOADING CICIDS2017 DATASET")
    print("=" * 70)

    for file in csv_files:

        df = pd.read_csv(
            file,
            low_memory=False,
            encoding="latin-1"
        )

        df.columns = df.columns.str.strip()

        print(f"{file.name}: {len(df):,} rows")

        dataframes.append(df)

    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    # The raw files contain the "Fwd Header Length" column twice
    data = data.drop(
        columns="Fwd Header Length.1",
        errors="ignore"
    )

    print("\nTotal rows:", f"{len(data):,}")
    print("Total columns:", len(data.columns))

    return data


# Basic cleaning

def clean_data(df):

    print("\n" + "=" * 70)
    print("BASIC DATA CLEANING")
    print("=" * 70)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    before = len(df)

    df = df.dropna()

    print(f"\nRows with missing/infinite values removed: {before - len(df):,}")

    before = len(df)

    df = df.drop_duplicates()

    print(f"Duplicate rows removed: {before - len(df):,}")
    print(f"Rows after cleaning: {len(df):,}")

    return df.reset_index(drop=True)


# Create binary and multi-class targets

def create_labels(df):

    df["Label"] = df["Label"].astype(str).str.strip()

    # Attack category (multi-class target)
    df["Category"] = df["Label"].map(cicids_category)

    # 0 = Normal, 1 = Attack (binary target)
    df["Target"] = (df["Category"] != NORMAL).astype(int)

    print("\nCategory distribution:")
    print(df["Category"].value_counts())

    return df


# Model input columns of the cleaned CICIDS2017 data

def feature_columns(df):

    return [
        column
        for column in df.columns
        if column not in LABEL_COLUMNS
    ]


# Main CICIDS2017 preprocessing pipeline

def preprocess():

    df = load_cicids()

    df = clean_data(df)

    df = create_labels(df)

    return df


# NSL-KDD

def load_nsl_kdd():

    frames = []

    for name in ("KDDTrain+.txt", "KDDTest+.txt"):

        path = NSL_DIR / name

        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found in {NSL_DIR}"
            )

        df = pd.read_csv(
            path,
            names=NSL_COLUMNS
        )

        df["Category"] = df["label"].map(NSL_LABEL_MAP)

        if df["Category"].isnull().any():
            unknown = df.loc[df["Category"].isnull(), "label"].unique()
            raise ValueError(f"Unmapped NSL-KDD labels: {unknown}")

        df["Target"] = (df["Category"] != NORMAL).astype(int)

        frames.append(df)

    return frames[0], frames[1]


# Run

if __name__ == "__main__":

    df = preprocess()

    print("\nFirst 5 rows:")
    print(df.head())
