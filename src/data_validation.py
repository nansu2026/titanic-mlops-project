import json
import os
import pandas as pd

FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
TARGET = "Survived"
EXPECTED_COLUMNS = FEATURES + [TARGET]

VALID_PCLASS = {1, 2, 3}
VALID_SEX = {"male", "female"}
VALID_EMBARKED = {"S", "C", "Q"}
VALID_TARGET = {0, 1}


def validate_schema(df: pd.DataFrame):
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    return True


def validate_values(df: pd.DataFrame):
    df = df.copy()

    if df[TARGET].isnull().sum() > 0:
        raise ValueError("Target column contains missing values.")

    if not set(df[TARGET].dropna().unique()).issubset(VALID_TARGET):
        raise ValueError("Survived must contain only 0 or 1.")

    if not set(df["Pclass"].dropna().unique()).issubset(VALID_PCLASS):
        raise ValueError("Pclass must contain only 1, 2, or 3.")

    if not set(df["Sex"].dropna().unique()).issubset(VALID_SEX):
        raise ValueError("Sex must contain only male or female.")

    if not set(df["Embarked"].dropna().unique()).issubset(VALID_EMBARKED):
        raise ValueError("Embarked must contain only S, C, or Q.")

    numeric_columns = ["Age", "Fare"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if df[column].isnull().all():
            raise ValueError(f"{column} has no valid numeric values.")

        if (df[column].dropna() < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

    return True


def create_data_profile(df: pd.DataFrame):
    profile = {
        "row_count": int(len(df)),
        "missing_values": df[EXPECTED_COLUMNS].isnull().sum().to_dict(),
        "target_distribution": df[TARGET].value_counts(normalize=True).to_dict(),
        "numeric_summary": df[["Age", "Fare"]].describe().to_dict(),
        "categorical_summary": {
            "Pclass": df["Pclass"].value_counts().to_dict(),
            "Sex": df["Sex"].value_counts().to_dict(),
            "Embarked": df["Embarked"].value_counts().to_dict()
        }
    }

    os.makedirs("metrics", exist_ok=True)

    with open("metrics/data_profile.json", "w") as file:
        json.dump(profile, file, indent=4)

    return profile


def validate_dataset(df: pd.DataFrame):
    validate_schema(df)
    validate_values(df)
    profile = create_data_profile(df)
    return profile