import os
import json
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, preprocess_data


def build_preprocessor():
    numeric_features = ["Pclass", "Age", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def train_models():
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    version = datetime.now().strftime("v%Y%m%d%H%M%S")

    results = {}
    best_model_name = None
    best_accuracy = 0
    best_pipeline = None

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)

        model_file = f"models/{model_name}_{version}.pkl"
        joblib.dump(pipeline, model_file)

        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "model_file": model_file
        }

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_pipeline = pipeline

    joblib.dump(best_pipeline, "models/best_model.pkl")

    registry = {
        "version": version,
        "best_model": best_model_name,
        "best_model_file": "models/best_model.pkl",
        "metrics": results
    }

    with open("models/model_registry.json", "w") as file:
        json.dump(registry, file, indent=4)

    with open("metrics/model_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("Training completed.")
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_accuracy}")


if __name__ == "__main__":
    train_models()