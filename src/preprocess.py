import pandas as pd

FEATURES = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
TARGET = "Survived"


def load_data(path="data/titanic.csv"):
    return pd.read_csv(path)


def preprocess_data(df):
    df = df.copy()

    df = df[FEATURES + [TARGET]]

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    X = df[FEATURES]
    y = df[TARGET]

    return X, y