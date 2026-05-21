import json
import subprocess
import sys
import pandas as pd

from drift_detection import detect_data_drift


CURRENT_DATA = "data/titanic.csv"
REFERENCE_DATA = "metrics/reference_dataset.csv"
DRIFT_REPORT = "metrics/drift_report.json"
THRESHOLD = 0.05


def main():
    print("Checking live data drift before retraining...")

    current_df = pd.read_csv(CURRENT_DATA)

    report = detect_data_drift(
        current_df=current_df,
        reference_data_path=REFERENCE_DATA,
        drift_report_path=DRIFT_REPORT,
        p_value_threshold=THRESHOLD
    )

    print(json.dumps(report, indent=4))

    if report["drift_detected"]:
        print("Data drift detected. Starting automatic retraining...")
        subprocess.run([sys.executable, "src/train.py"], check=True)
        print("Retraining completed because data drift was detected.")
    else:
        print("No data drift detected. Retraining skipped.")


if __name__ == "__main__":
    main()