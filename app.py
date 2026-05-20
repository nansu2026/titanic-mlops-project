import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("models/best_model.pkl")


@app.route("/")
def home():
    return jsonify({
        "message": "Titanic Survival Prediction API is running",
        "browser_form": "/form"
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


@app.route("/form")
def form():
    return """
    <html>
    <head>
        <title>Titanic Survival Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #dbeafe, #f8fafc);
                margin: 0;
                padding: 0;
            }

            .container {
                width: 420px;
                margin: 60px auto;
                background: white;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.12);
            }

            h1 {
                text-align: center;
                color: #1e3a8a;
                margin-bottom: 10px;
            }

            p {
                text-align: center;
                color: #555;
                font-size: 14px;
                margin-bottom: 25px;
            }

            label {
                font-weight: bold;
                color: #333;
            }

            input, select {
                width: 100%;
                padding: 10px;
                margin-top: 6px;
                margin-bottom: 16px;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 14px;
            }

            button {
                width: 100%;
                background: #2563eb;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
            }

            button:hover {
                background: #1d4ed8;
            }

            .footer {
                margin-top: 20px;
                text-align: center;
                font-size: 12px;
                color: #777;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h1>Titanic Survival Prediction</h1>
            <p>Enter passenger details to predict survival.</p>

            <form action="/predict_form" method="post">

                <label>Passenger Class</label>
                <select name="Pclass">
                    <option value="1">1st Class</option>
                    <option value="2">2nd Class</option>
                    <option value="3">3rd Class</option>
                </select>

                <label>Sex</label>
                <select name="Sex">
                    <option value="female">Female</option>
                    <option value="male">Male</option>
                </select>

                <label>Age</label>
                <input type="number" name="Age" value="22" min="0" max="100">

                <label>Fare</label>
                <input type="number" step="0.01" name="Fare" value="100">

                <label>Embarked</label>
                <select name="Embarked">
                    <option value="S">Southampton (S)</option>
                    <option value="C">Cherbourg (C)</option>
                    <option value="Q">Queenstown (Q)</option>
                </select>

                <button type="submit">Predict Survival</button>
            </form>

            <div class="footer">
                Titanic MLOps Pipeline | Flask + Docker + GitHub Actions
            </div>
        </div>
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
    result = "Survived" if int(prediction) == 1 else "Did Not Survive"

    if int(prediction) == 1:
        color = "#16a34a"
        emoji = "✅"
    else:
        color = "#dc2626"
        emoji = "❌"

    return f"""
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #dbeafe, #f8fafc);
                margin: 0;
                padding: 0;
            }}

            .container {{
                width: 420px;
                margin: 80px auto;
                background: white;
                padding: 35px;
                border-radius: 16px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.12);
                text-align: center;
            }}

            h1 {{
                color: #1e3a8a;
            }}

            .result {{
                font-size: 28px;
                font-weight: bold;
                color: {color};
                margin: 25px 0;
            }}

            .raw {{
                color: #555;
                margin-bottom: 25px;
            }}

            a {{
                display: inline-block;
                background: #2563eb;
                color: white;
                padding: 12px 18px;
                border-radius: 8px;
                text-decoration: none;
            }}

            a:hover {{
                background: #1d4ed8;
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <h1>Titanic Survival Prediction Result</h1>
            <div class="result">{emoji} Prediction: {result}</div>
            <div class="raw">Raw prediction value: {int(prediction)}</div>
            <a href="/form">Try Another Prediction</a>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)