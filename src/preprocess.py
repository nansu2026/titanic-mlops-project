import pandas as pd

FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
TARGET = "Survived"

EXPECTED_COLUMNS = FEATURES + [TARGET]


def load_data(path="data/titanic.csv"):
    return pd.read_csv(path)


def validate_data(df):
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[TARGET].isnull().sum() > 0:
        raise ValueError("Target column contains missing values.")

    valid_classes = {0, 1}
    actual_classes = set(df[TARGET].unique())

    if not actual_classes.issubset(valid_classes):
        raise ValueError("Target column must contain only 0 and 1.")

    return True


def preprocess_data(df):
    df = df.copy()

    validate_data(df)

    df = df[EXPECTED_COLUMNS]

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    X = df[FEATURES]
    y = df[TARGET]

    return X, y