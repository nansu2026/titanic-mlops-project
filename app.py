import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "models/best_model.pkl"
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return jsonify({
        "message": "Titanic Survival Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_data = pd.DataFrame([{
        "Pclass": data["Pclass"],
        "Sex": data["Sex"],
        "Age": data["Age"],
        "Fare": data["Fare"],
        "Embarked": data["Embarked"]
    }])

    prediction = model.predict(input_data)[0]

    return jsonify({
        "survival_prediction": int(prediction)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)