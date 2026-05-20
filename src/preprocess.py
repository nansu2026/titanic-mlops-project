import pandas as pd
from data_validation import validate_dataset


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    validate_dataset(df)

    df = df.copy()

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Features
    X = df[[
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked"
    ]]

    # Target
    y = df["Survived"]

    return X, y