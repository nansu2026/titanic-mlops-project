import pandas as pd
from src.preprocess import preprocess_data


def test_preprocess_removes_missing_values():
    data = {
        "Survived": [0, 1, 1],
        "Pclass": [3, 1, 2],
        "Sex": ["male", "female", "female"],
        "Age": [22, None, 30],
        "Fare": [7.25, 71.28, None],
        "Embarked": ["S", None, "C"]
    }

    df = pd.DataFrame(data)
    X, y = preprocess_data(df)

    assert X["Age"].isnull().sum() == 0
    assert X["Fare"].isnull().sum() == 0
    assert X["Embarked"].isnull().sum() == 0
    assert len(X) == len(y)