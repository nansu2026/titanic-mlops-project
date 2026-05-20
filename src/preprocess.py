import pandas as pd
from data_validation import validate_dataset

FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
TARGET = "Survived"
EXPECTED_COLUMNS = FEATURES + [TARGET]


def load_data(path="data/titanic.csv"):
    return pd.read_csv(path)


def preprocess_data(df):
    df = df.copy()

    validate_dataset(df)

    df = df[EXPECTED_COLUMNS]

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    X = df[FEATURES]
    y = df[TARGET]

    return X, y