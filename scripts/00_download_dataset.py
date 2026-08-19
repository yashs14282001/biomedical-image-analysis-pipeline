from __future__ import annotations

import sys
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

from config import DATASET_EXTRACT_DIR, DATASET_URL, DATASET_ZIP, RAW_DIR


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_ZIP.exists():
        print(f"Downloading dataset from:\n{DATASET_URL}")
        with requests.get(DATASET_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(DATASET_ZIP, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Saved: {DATASET_ZIP}")
    else:
        print(f"Dataset ZIP already exists: {DATASET_ZIP}")

    DATASET_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if not any(DATASET_EXTRACT_DIR.iterdir()):
        with zipfile.ZipFile(DATASET_ZIP, "r") as archive:
            archive.extractall(DATASET_EXTRACT_DIR)
        print(f"Extracted to: {DATASET_EXTRACT_DIR}")
    else:
        print(f"Extracted dataset already present: {DATASET_EXTRACT_DIR}")


if __name__ == "__main__":
    main()
