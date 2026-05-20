import hashlib
import json
import os
from datetime import datetime


def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


def save_data_version(data_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    version = {
        "data_path": data_path,
        "data_hash": calculate_file_hash(data_path),
        "version_time": datetime.now().isoformat()
    }

    with open(output_path, "w") as file:
        json.dump(version, file, indent=4)

    return version