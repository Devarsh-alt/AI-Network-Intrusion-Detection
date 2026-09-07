import pandas as pd
from pathlib import Path




DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"



def get_csv_files():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {DATA_DIR}"
        )

    return csv_files



def explore_dataset():

    csv_files = get_csv_files()

    print("=" * 70)
    print("CICIDS2017 DATASET EXPLORATION")
    print("=" * 70)

    print(f"\nDataset folder:")
    print(DATA_DIR)

    print(f"\nNumber of CSV files: {len(csv_files)}")

    total_rows = 0
    overall_labels = {}


    for file in csv_files:

        print("\n" + "-" * 70)
        print(f"FILE: {file.name}")
        print("-" * 70)

        row_count = 0
        labels = {}

        first_chunk = True

        for chunk in pd.read_csv(
            file,
            low_memory=False,
            chunksize=100000
        ):

            # Remove unnecessary spaces from column names
            chunk.columns = chunk.columns.str.strip()

            row_count += len(chunk)

            if "Label" in chunk.columns:

                counts = chunk["Label"].value_counts()

                for label, count in counts.items():

                    labels[label] = labels.get(label, 0) + count

                    overall_labels[label] = (
                        overall_labels.get(label, 0) + count
                    )

            # Print columns only once
            if first_chunk:

                print(f"Columns    : {len(chunk.columns)}")

                print("\nColumns:")

                for column in chunk.columns:
                    print(f"  {column}")

                first_chunk = False

        total_rows += row_count

        print(f"\nRows       : {row_count:,}")

        print("\nLabel distribution:")

        if labels:

            for label, count in sorted(
                labels.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  {label}: {count:,}")

        else:
            print("  Label column not found.")



    print("\n" + "=" * 70)
    print("OVERALL DATASET SUMMARY")
    print("=" * 70)

    print(f"\nTotal CSV files : {len(csv_files)}")
    print(f"Total rows      : {total_rows:,}")

    print("\nOverall Label Distribution:")

    for label, count in sorted(
        overall_labels.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {label}: {count:,}")


if __name__ == "__main__":
    explore_dataset()