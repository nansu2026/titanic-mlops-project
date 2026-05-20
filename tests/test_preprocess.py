import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.abspath("src"))

from preprocess import preprocess_data
from data_validation import validate_dataset


def test_preprocess_data():
    df = pd.DataFrame({
        "Pclass": [1, 3],
        "Sex": ["female", "male"],
        "Age": [25, 30],
        "Fare": [100, 7.25],
        "Embarked": ["C", "S"],
        "Survived": [1, 0]
    })

    X, y = preprocess_data(df)

    assert X.shape[0] == 2
    assert y.shape[0] == 2


def test_missing_columns():
    df = pd.DataFrame({
        "Pclass": [1],
        "Sex": ["female"]
    })

    with pytest.raises(ValueError):
        validate_dataset(df)


def test_invalid_target():
    df = pd.DataFrame({
        "Pclass": [1],
        "Sex": ["male"],
        "Age": [22],
        "Fare": [7.25],
        "Embarked": ["S"],
        "Survived": [5]
    })

    with pytest.raises(ValueError):
        validate_dataset(df)


def test_negative_age():
    df = pd.DataFrame({
        "Pclass": [1],
        "Sex": ["male"],
        "Age": [-10],
        "Fare": [7.25],
        "Embarked": ["S"],
        "Survived": [0]
    })

    with pytest.raises(ValueError):
        validate_dataset(df)