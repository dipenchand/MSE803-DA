from pathlib import Path

import pandas as pd


DATA_GLOB = "PRSA_Data_*.csv"

# Load the dataset from CSV files.
def load_dataset() -> pd.DataFrame:
    files = sorted(Path(".").glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found for pattern {DATA_GLOB!r}")
    return pd.concat((pd.read_csv(file) for file in files), ignore_index=True)


def main() -> None:
    df = load_dataset()
    print("Dataset loaded successfully.\n")

    print("First 5 rows")
    print(df.head().to_string(index=False))
    print("*" * 20)

    print("Column names:")
    print(", ".join(df.columns))
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\n")
    print("*" * 20)

    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")

    print("*" * 20)
    print(f"Total Stations: {df['station'].nunique()}")
    print(f"Name of stations: {', '.join(df['station'].unique())}")



if __name__ == "__main__":
    main()
