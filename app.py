import os
import json
import yaml
import joblib
import pandas as pd

from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)


# =========================
# Load configuration
# =========================
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_PATH = config["model"]["path"]
REGISTRY_PATH = config["model"]["registry_path"]

# =========================
# Load trained model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Train model first."
    )

model = joblib.load(MODEL_PATH)


# =========================
# Home Route
# =========================
@app.route("/")
def home():
    return jsonify({
        "message": "Titanic MLOps API is running",
        "health_url": "/health",
        "predict_url": "/predict",
        "form_url": "/form",
        "metadata_url": "/metadata"
    })


# =========================
# Health Check
# =========================
@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


# =========================
# Metadata Route
# =========================
@app.route("/metadata")
def metadata():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as file:
            registry = json.load(file)

        return jsonify(registry)

    return jsonify({
        "error": "Registry file not found"
    }), 404


# =========================
# Prediction API
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Invalid request. Please send JSON data using POST."
        }), 400

    required_fields = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": f"Missing fields: {missing_fields}"
        }), 400

    try:
        pclass = int(data["Pclass"])
        sex = str(data["Sex"]).lower()
        age = float(data["Age"])
        fare = float(data["Fare"])
        embarked = str(data["Embarked"]).upper()

        # =========================
        # Input Validation
        # =========================

        if pclass not in [1, 2, 3]:
            return jsonify({
                "error": "Pclass must be 1, 2, or 3"
            }), 400

        if sex not in ["male", "female"]:
            return jsonify({
                "error": "Sex must be male or female"
            }), 400

        if embarked not in ["S", "C", "Q"]:
            return jsonify({
                "error": "Embarked must be S, C, or Q"
            }), 400

        if age < 0:
            return jsonify({
                "error": "Age cannot be negative"
            }), 400

        if fare < 0:
            return jsonify({
                "error": "Fare cannot be negative"
            }), 400

        input_df = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "Fare": fare,
            "Embarked": embarked
        }])

        prediction = model.predict(input_df)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(input_df)[0][1]
            )

        return jsonify({
            "prediction": int(prediction),
            "survived": bool(prediction),
            "survival_probability": probability
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


# =========================
# HTML Form UI
# =========================
@app.route("/form")
def form():

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titanic Survival Prediction</title>

        <style>
            body {
                font-family: Arial;
                background-color: #f4f4f4;
                padding: 40px;
            }

            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                max-width: 500px;
                margin: auto;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }

            h1 {
                text-align: center;
                color: #333;
            }

            input, select {
                width: 100%;
                padding: 10px;
                margin-top: 10px;
                margin-bottom: 20px;
                border-radius: 5px;
                border: 1px solid #ccc;
            }

            button {
                width: 100%;
                padding: 12px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }

            button:hover {
                background-color: #0056b3;
            }

            .result {
                margin-top: 20px;
                padding: 15px;
                background-color: #e9ecef;
                border-radius: 5px;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <h1>Titanic Survival Prediction</h1>

            <form id="predictionForm">

                <label>Pclass</label>
                <select name="Pclass">
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3" selected>3</option>
                </select>

                <label>Sex</label>
                <select name="Sex">
                    <option value="male" selected>Male</option>
                    <option value="female">Female</option>
                </select>

                <label>Age</label>
                <input type="number" name="Age" value="22" required>

                <label>Fare</label>
                <input type="number" step="0.01" name="Fare" value="7.25" required>

                <label>Embarked</label>
                <select name="Embarked">
                    <option value="S" selected>S</option>
                    <option value="C">C</option>
                    <option value="Q">Q</option>
                </select>

                <button type="submit">Predict Survival</button>

            </form>

            <div class="result" id="result"></div>

        </div>

        <script>

            document.getElementById("predictionForm")
                .addEventListener("submit", async function(event) {

                event.preventDefault();

                const formData = new FormData(event.target);

                const data = {
                    Pclass: parseInt(formData.get("Pclass")),
                    Sex: formData.get("Sex"),
                    Age: parseFloat(formData.get("Age")),
                    Fare: parseFloat(formData.get("Fare")),
                    Embarked: formData.get("Embarked")
                };

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                document.getElementById("result").innerHTML =
                    "<pre>" + JSON.stringify(result, null, 2) + "</pre>";
            });

        </script>

    </body>
    </html>
    """

    return render_template_string(html)


# =========================
# Run Flask app
# =========================
if __name__ == "__main__":
    app.run(
        host=config["server"]["host"],
        port=config["server"]["port"],
        debug=False
    )