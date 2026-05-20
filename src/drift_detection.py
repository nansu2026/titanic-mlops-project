import json
import os
from datetime import datetime

import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


NUMERIC_COLUMNS = ["Pclass", "Age", "Fare"]
CATEGORICAL_COLUMNS = ["Sex", "Embarked", "Survived"]


def create_data_profile(df):
    profile = {
        "created_at": datetime.now().isoformat(),
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": {},
        "categorical_distribution": {}
    }

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            profile["numeric_summary"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "median": float(df[col].median()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            profile["categorical_distribution"][col] = (
                df[col].astype(str).value_counts(normalize=True).to_dict()
            )

    return profile


def detect_data_drift(current_df, reference_data_path, drift_report_path, p_value_threshold=0.05):
    os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)

    if not os.path.exists(reference_data_path):
        current_df.to_csv(reference_data_path, index=False)

        report = {
            "created_at": datetime.now().isoformat(),
            "drift_detected": False,
            "reason": "No reference dataset existed. Current data saved as reference.",
            "numeric_tests": {},
            "categorical_tests": {}
        }

        with open(drift_report_path, "w") as file:
            json.dump(report, file, indent=4)

        return report

    reference_df = pd.read_csv(reference_data_path)

    numeric_tests = {}
    categorical_tests = {}
    drift_detected = False

    for col in NUMERIC_COLUMNS:
        if col in current_df.columns and col in reference_df.columns:
            current_values = current_df[col].dropna()
            reference_values = reference_df[col].dropna()

            statistic, p_value = ks_2samp(reference_values, current_values)

            numeric_tests[col] = {
                "test": "Kolmogorov-Smirnov",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < p_value_threshold)
            }

            if p_value < p_value_threshold:
                drift_detected = True

    for col in CATEGORICAL_COLUMNS:
        if col in current_df.columns and col in reference_df.columns:
            current_counts = current_df[col].astype(str).value_counts()
            reference_counts = reference_df[col].astype(str).value_counts()

            all_categories = sorted(set(current_counts.index).union(set(reference_counts.index)))

            table = [
                [int(reference_counts.get(cat, 0)) for cat in all_categories],
                [int(current_counts.get(cat, 0)) for cat in all_categories]
            ]

            try:
                statistic, p_value, _, _ = chi2_contingency(table)

                categorical_tests[col] = {
                    "test": "Chi-square",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift_detected": bool(p_value < p_value_threshold)
                }

                if p_value < p_value_threshold:
                    drift_detected = True

            except ValueError:
                categorical_tests[col] = {
                    "test": "Chi-square",
                    "error": "Not enough category variation to calculate drift.",
                    "drift_detected": False
                }

    report = {
        "created_at": datetime.now().isoformat(),
        "drift_detected": drift_detected,
        "p_value_threshold": p_value_threshold,
        "numeric_tests": numeric_tests,
        "categorical_tests": categorical_tests
    }

    with open(drift_report_path, "w") as file:
        json.dump(report, file, indent=4)

    return report