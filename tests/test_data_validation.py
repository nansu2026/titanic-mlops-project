import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.abspath("src"))

from data_validation import validate_dataset


def test_valid_dataset_passes():
    df = pd.DataFrame({
        "Pclass": [1, 2, 3],
        "Sex": ["male", "female", "male"],
        "Age": [22, 38, 26],
        "Fare": [7.25, 71.28, 8.05],
        "Embarked": ["S", "C", "Q"],
        "Survived": [0, 1, 0]
    })

    assert validate_dataset(df) is not None


def test_missing_column_fails():
    df = pd.DataFrame({
        "Pclass": [1],
        "Sex": ["male"],
        "Age": [22],
        "Fare": [7.25],
        "Survived": [0]
    })

    with pytest.raises(ValueError):
        validate_dataset(df)


def test_invalid_target_fails():
    df = pd.DataFrame({
        "Pclass": [1],
        "Sex": ["male"],
        "Age": [22],
        "Fare": [7.25],
        "Embarked": ["S"],
        "Survived": [2]
    })

    with pytest.raises(ValueError):
        validate_dataset(df)