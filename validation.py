from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/customers.csv")

REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract",
    "internet_service",
    "churn",
]


def validate_data(file_path=DATA_PATH):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Preserve "None" as a valid category
    df = pd.read_csv(file_path, keep_default_na=False)
    df = df.replace("", pd.NA)

    if df.empty:
        raise ValueError("Dataset is empty")

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values")

    for column in ["tenure", "monthly_charges", "support_calls", "churn"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 and 1")

    if (df["tenure"] < 0).any():
        raise ValueError("Tenure cannot be negative")

    if (df["monthly_charges"] < 0).any():
        raise ValueError("Monthly charges cannot be negative")

    if (df["support_calls"] < 0).any():
        raise ValueError("Support calls cannot be negative")

    if not set(df["contract"].unique()).issubset(
        {"Monthly", "One Year", "Two Year"}
    ):
        raise ValueError("Unexpected contract type")

    if not set(df["internet_service"].unique()).issubset(
        {"DSL", "Fiber", "None"}
    ):
        raise ValueError("Unexpected internet service")

    if df["churn"].nunique() != 2:
        raise ValueError("Target must contain both churn classes")

    print(f"Data validation passed: {len(df)} rows")
    return df


if __name__ == "__main__":
    validate_data()