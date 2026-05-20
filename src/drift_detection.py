import json
import os
import pandas as pd


def create_data_profile(df):
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    profile = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_means": df[numeric_columns].mean().to_dict(),
        "numeric_std": df[numeric_columns].std().to_dict()
    }

    if "Survived" in df.columns:
        profile["target_distribution"] = df["Survived"].value_counts(normalize=True).to_dict()

    return profile


def detect_data_drift(current_profile, reference_profile_path, drift_report_path, threshold):
    os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)

    if not os.path.exists(reference_profile_path):
        with open(reference_profile_path, "w") as file:
            json.dump(current_profile, file, indent=4)

        report = {
            "drift_detected": False,
            "reason": "No reference profile existed. Current profile saved as reference.",
            "drift_score": 0
        }

    else:
        with open(reference_profile_path, "r") as file:
            reference_profile = json.load(file)

        drift_score = 0
        drift_details = {}

        for col, current_mean in current_profile.get("numeric_means", {}).items():
            old_mean = reference_profile.get("numeric_means", {}).get(col)

            if old_mean is not None and old_mean != 0:
                change = abs(current_mean - old_mean) / abs(old_mean)
                drift_details[col] = change

                if change > threshold:
                    drift_score += 1

        report = {
            "drift_detected": drift_score > 0,
            "drift_score": drift_score,
            "threshold": threshold,
            "details": drift_details
        }

    with open(drift_report_path, "w") as file:
        json.dump(report, file, indent=4)

    return report