from app import app


def test_home_route():
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Titanic Survival Prediction API is running" in response.data


def test_predict_route():
    client = app.test_client()

    sample_data = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 25,
        "Fare": 7.25,
        "Embarked": "S"
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert "survival_prediction" in response.get_json()