import os
import json
import joblib
import yaml
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, preprocess_data


def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )


def get_previous_best_accuracy(registry_path):
    if os.path.exists(registry_path):
        with open(registry_path, "r") as file:
            old_registry = json.load(file)
            return old_registry.get("production_accuracy", 0)

    return 0


def train_models():
    config = load_config()

    data_path = config["data"]["path"]
    model_path = config["model"]["path"]
    registry_path = config["model"]["registry_path"]
    minimum_accuracy = config["model"]["minimum_accuracy"]

    mlflow_tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    registered_model_name = config["mlflow"]["registered_model_name"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    df = load_data(data_path)
    X, y = preprocess_data(df)

    data_profile = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df["Survived"].value_counts().to_dict(),
    }

    with open("metrics/data_profile.json", "w") as file:
        json.dump(data_profile, file, indent=4)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
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
    best_model_file = None
    best_run_id = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{version}") as run:
            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            conf_matrix = confusion_matrix(y_test, predictions).tolist()

            model_file = f"models/{model_name}_{version}.pkl"
            joblib.dump(pipeline, model_file)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("version", version)
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("registered_model_name", registered_model_name)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.log_artifact("metrics/data_profile.json")

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=registered_model_name
            )

            results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": conf_matrix,
                "model_file": model_file,
                "mlflow_run_id": run.info.run_id
            }

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                best_pipeline = pipeline
                best_model_file = model_file
                best_run_id = run.info.run_id

    previous_best_accuracy = get_previous_best_accuracy(registry_path)

    if best_accuracy < minimum_accuracy:
        raise ValueError(
            f"Best model accuracy {best_accuracy} is below required minimum {minimum_accuracy}"
        )

    if best_accuracy >= previous_best_accuracy:
        joblib.dump(best_pipeline, model_path)
        production_model_updated = True
        production_accuracy = best_accuracy
        production_model_name = best_model_name
        production_source_file = best_model_file
    else:
        production_model_updated = False
        production_accuracy = previous_best_accuracy
        production_model_name = "previous_production_model"
        production_source_file = model_path

    registry = {
        "version": version,
        "training_time": datetime.now().isoformat(),
        "mlflow_experiment": experiment_name,
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "mlflow_registered_model": registered_model_name,
        "best_mlflow_run_id": best_run_id,
        "new_training_best_model": best_model_name,
        "new_training_best_accuracy": best_accuracy,
        "previous_production_accuracy": previous_best_accuracy,
        "minimum_required_accuracy": minimum_accuracy,
        "production_model_updated": production_model_updated,
        "production_model_name": production_model_name,
        "production_accuracy": production_accuracy,
        "production_model_file": model_path,
        "production_source_file": production_source_file,
        "data_profile_file": "metrics/data_profile.json",
        "metrics": results
    }

    with open(registry_path, "w") as file:
        json.dump(registry, file, indent=4)

    with open("metrics/model_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("Training completed.")
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_accuracy}")
    print(f"MLflow run ID: {best_run_id}")
    print(f"Production model updated: {production_model_updated}")


if __name__ == "__main__":
    train_models()