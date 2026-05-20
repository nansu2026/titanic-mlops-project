import os
import sys
import pandas as pd


REQUIRED_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Survived"
]


def validate_dataset(data):
    if isinstance(data, (str, bytes, os.PathLike)):
        if not os.path.exists(data):
            raise FileNotFoundError(f"Dataset not found: {data}")
        df = pd.read_csv(data)
    else:
        df = data

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset is empty")

    if not set(df["Survived"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column 'Survived' must contain only 0 and 1")

    if (df["Age"].dropna() < 0).any():
        raise ValueError("Age cannot contain negative values")

    return True


def validate_data(data):
    return validate_dataset(data)


if __name__ == "__main__":
    DATA_PATH = "data/titanic.csv"

    try:
        validate_dataset(DATA_PATH)
        print("Data validation passed successfully")
    except Exception as e:
        print(f"Data validation failed: {e}")
        sys.exit(1)