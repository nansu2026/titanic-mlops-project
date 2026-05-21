import json
import os
import subprocess
import sys

import pandas as pd

from drift_detection import detect_data_drift


CURRENT_DATA = "data/titanic.csv"
REFERENCE_DATA = "metrics/reference_dataset.csv"
DRIFT_REPORT = "metrics/drift_report.json"
MODEL_PATH = "models/best_model.pkl"
THRESHOLD = 0.05


def run_training(reason):
    print(f"Starting training. Reason: {reason}")
    subprocess.run([sys.executable, "src/train.py"], check=True)
    print("Training completed successfully.")


def main():
    print("Continuous training trigger started.")

    os.makedirs("metrics", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        run_training("No existing trained model found.")
        return

    if not os.path.exists(REFERENCE_DATA):
        run_training("No reference dataset found.")
        return

    current_df = pd.read_csv(CURRENT_DATA)

    report = detect_data_drift(
        current_df=current_df,
        reference_data_path=REFERENCE_DATA,
        drift_report_path=DRIFT_REPORT,
        p_value_threshold=THRESHOLD,
    )

    print(json.dumps(report, indent=4))

    if report.get("drift_detected"):
        run_training("Data drift detected.")
    else:
        print("No data drift detected. Training skipped.")


if __name__ == "__main__":
    main()