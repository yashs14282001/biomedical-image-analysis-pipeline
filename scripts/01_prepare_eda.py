from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATASET_EXTRACT_DIR, OUTPUT_DIR, PROCESSED_DIR
from src.data import prepare_dataset
from src.eda import make_eda_figures
from src.paths import ensure_project_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess nuclei images and create EDA outputs.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_EXTRACT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    manifest = prepare_dataset(
        args.dataset_root,
        PROCESSED_DIR,
        overwrite=args.overwrite,
    )
    stats = make_eda_figures(manifest, PROCESSED_DIR, OUTPUT_DIR / "eda")
    print("\nDataset split counts:")
    print(manifest.groupby("split").size())
    print(f"\nManifest: {PROCESSED_DIR / 'manifest.csv'}")
    print(f"EDA outputs: {OUTPUT_DIR / 'eda'}")
    print("\nIntensity statistics summary:")
    print(stats.groupby("split")[["mean_intensity", "std_intensity"]].mean().round(4))


if __name__ == "__main__":
    main()
