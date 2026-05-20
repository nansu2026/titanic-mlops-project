import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model
model = joblib.load("models/best_model.pkl")


# Home route
@app.route("/")
def home():
    return jsonify({
        "message": "Titanic Survival Prediction API is running"
    })


# API prediction route
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


# Browser form page
@app.route("/form")
def form():
    return """
    <html>
    <head>
        <title>Titanic Survival Prediction</title>
    </head>

    <body>

        <h1>Titanic Survival Prediction</h1>

        <form action="/predict_form" method="post">

            <label>Pclass:</label><br>
            <input type="number" name="Pclass" value="1"><br><br>

            <label>Sex:</label><br>
            <input type="text" name="Sex" value="female"><br><br>

            <label>Age:</label><br>
            <input type="number" name="Age" value="22"><br><br>

            <label>Fare:</label><br>
            <input type="number" step="0.01" name="Fare" value="100"><br><br>

            <label>Embarked:</label><br>
            <input type="text" name="Embarked" value="S"><br><br>

            <input type="submit" value="Predict Survival">

        </form>

    </body>
    </html>
    """


# Browser prediction result
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

    result = "Survived" if int(prediction) == 1 else "Did Not Survive"

    return f"""
    <html>
    <head>
        <title>Prediction Result</title>
    </head>

    <body>

        <h1>Titanic Survival Prediction Result</h1>

        <h2>Prediction: {result}</h2>

        <p>Raw prediction value: {int(prediction)}</p>

        <br>

        <a href="/form">Try Another Prediction</a>

    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)