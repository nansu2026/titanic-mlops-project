import json
import yaml
import joblib
from pathlib import Path
from flask import Flask, request, jsonify
import pandas as pd


app = Flask(__name__)


def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


config = load_config()

MODEL_PATH = Path(config["model"]["path"])
REGISTRY_PATH = Path(config["model"]["registry_path"])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Please run: python src/train.py"
        )
    return joblib.load(MODEL_PATH)


def load_model_metadata():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as file:
            return json.load(file)

    return {
        "message": "Model registry not found",
        "model_path": str(MODEL_PATH)
    }


model = load_model()


@app.route("/")
def home():
    return jsonify({
        "message": "Titanic MLOps API is running",
        "form_url": "/form",
        "health_url": "/health",
        "metadata_url": "/metadata",
        "predict_url": "/predict"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }), 200


@app.route("/metadata")
def metadata():
    return jsonify(load_model_metadata())


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required_fields = ["Pclass", "Sex", "Age", "Fare", "Embarked"]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    input_data = pd.DataFrame([{
        "Pclass": int(data["Pclass"]),
        "Sex": data["Sex"],
        "Age": float(data["Age"]),
        "Fare": float(data["Fare"]),
        "Embarked": data["Embarked"]
    }])

    prediction = model.predict(input_data)[0]

    return jsonify({
        "survival_prediction": int(prediction)
    })


@app.route("/form")
def form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titanic MLOps Prediction</title>
    </head>
    <body>
        <h2>Titanic Survival Prediction</h2>

        <form action="/predict_form" method="post">
            <label>Passenger Class</label>
            <select name="Pclass" required>
                <option value="1">1st Class</option>
                <option value="2">2nd Class</option>
                <option value="3" selected>3rd Class</option>
            </select><br><br>

            <label>Sex</label>
            <select name="Sex" required>
                <option value="male" selected>Male</option>
                <option value="female">Female</option>
            </select><br><br>

            <label>Age</label>
            <input type="number" step="0.1" name="Age" value="25" required><br><br>

            <label>Fare</label>
            <input type="number" step="0.1" name="Fare" value="7.25" required><br><br>

            <label>Embarked</label>
            <select name="Embarked" required>
                <option value="S" selected>S - Southampton</option>
                <option value="C">C - Cherbourg</option>
                <option value="Q">Q - Queenstown</option>
            </select><br><br>

            <button type="submit">Predict Survival</button>
        </form>

        <br>
        <a href="/health">Health</a>
        <a href="/metadata">Model Metadata</a>
    </body>
    </html>
    """


@app.route("/predict_form", methods=["POST"])
def predict_form():
    input_data = pd.DataFrame([{
        "Pclass": int(request.form["Pclass"]),
        "Sex": request.form["Sex"],
        "Age": float(request.form["Age"]),
        "Fare": float(request.form["Fare"]),
        "Embarked": request.form["Embarked"]
    }])

    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"

    return f"""
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Prediction Result</h2>
        <h3>{result}</h3>
        <p>Prediction value: {int(prediction)}</p>
        <a href="/form">Try Again</a>
    </body>
    </html>
    """


if __name__ == "__main__":
    server_config = config.get("server", {})
    app.run(
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 5000),
        debug=True
    )