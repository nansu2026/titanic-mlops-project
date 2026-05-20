import os
import json
import joblib
import yaml
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, preprocess_data
from data_versioning import save_data_version
from drift_detection import create_data_profile, detect_data_drift


def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def build_preprocessor():
    numeric_features = ["Pclass", "Age", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])


def get_previous_best_accuracy(registry_path):
    if os.path.exists(registry_path):
        with open(registry_path, "r") as file:
            registry = json.load(file)
            return registry.get("production_accuracy", 0)
    return 0


def append_registry_history(history_path, registry):
    history = []

    if os.path.exists(history_path):
        with open(history_path, "r") as file:
            history = json.load(file)

    history.append(registry)

    with open(history_path, "w") as file:
        json.dump(history, file, indent=4)


def train_models():
    config = load_config()

    data_path = config["data"]["path"]
    version_file = config["data"]["version_file"]

    reference_profile = config["data"]["reference_profile"]
    reference_dataset = config["data"]["reference_dataset"]

    drift_report_path = config["data"]["drift_report"]
    drift_p_value_threshold = config["data"]["drift_p_value_threshold"]

    model_path = config["model"]["path"]
    registry_path = config["model"]["registry_path"]
    history_path = config["model"]["history_path"]
    minimum_accuracy = config["model"]["minimum_accuracy"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    df = load_data(data_path)

    data_version = save_data_version(data_path, version_file)

    current_profile = create_data_profile(df)

    with open("metrics/data_profile.json", "w") as file:
        json.dump(current_profile, file, indent=4)

    with open(reference_profile, "w") as file:
        json.dump(current_profile, file, indent=4)

    drift_report = detect_data_drift(
        current_df=df,
        reference_data_path=reference_dataset,
        drift_report_path=drift_report_path,
        p_value_threshold=drift_p_value_threshold
    )

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    version = datetime.now().strftime("v%Y%m%d%H%M%S")

    results = {}
    best_accuracy = 0
    best_pipeline = None
    best_model_name = None
    best_model_file = None
    best_run_id = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{version}") as run:
            pipeline = Pipeline([
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
            mlflow.log_param("data_hash", data_version["data_hash"])
            mlflow.log_param("drift_detected", drift_report["drift_detected"])

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.log_artifact(version_file)
            mlflow.log_artifact("metrics/data_profile.json")
            mlflow.log_artifact(reference_profile)
            mlflow.log_artifact(drift_report_path)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=config["mlflow"]["registered_model_name"]
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
                best_pipeline = pipeline
                best_model_name = model_name
                best_model_file = model_file
                best_run_id = run.info.run_id

    previous_accuracy = get_previous_best_accuracy(registry_path)

    if best_accuracy < minimum_accuracy:
        raise ValueError(
            f"Training failed. Best accuracy {best_accuracy} is below minimum {minimum_accuracy}"
        )

    if best_accuracy >= previous_accuracy:
        joblib.dump(best_pipeline, model_path)
        production_updated = True
        production_accuracy = best_accuracy
    else:
        production_updated = False
        production_accuracy = previous_accuracy

    registry = {
        "version": version,
        "training_time": datetime.now().isoformat(),
        "data_hash": data_version["data_hash"],
        "data_version_file": version_file,
        "reference_profile_file": reference_profile,
        "reference_dataset_file": reference_dataset,
        "drift_report_file": drift_report_path,
        "drift_detected": drift_report["drift_detected"],
        "best_model_name": best_model_name,
        "best_model_file": best_model_file,
        "best_mlflow_run_id": best_run_id,
        "new_training_accuracy": best_accuracy,
        "previous_production_accuracy": previous_accuracy,
        "production_accuracy": production_accuracy,
        "production_model_updated": production_updated,
        "production_model_file": model_path,
        "minimum_required_accuracy": minimum_accuracy,
        "metrics": results
    }

    with open(registry_path, "w") as file:
        json.dump(registry, file, indent=4)

    append_registry_history(history_path, registry)

    with open("metrics/model_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("Strict ML training pipeline completed successfully.")
    print(f"Data hash: {data_version['data_hash']}")
    print(f"Drift detected: {drift_report['drift_detected']}")
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_accuracy}")
    print(f"Production model updated: {production_updated}")


if __name__ == "__main__":
    train_models()