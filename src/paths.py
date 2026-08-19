from pathlib import Path
from config import OUTPUT_DIR, MODEL_DIR, PROCESSED_DIR


def ensure_project_dirs() -> None:
    """Create all generated-data directories used by the pipeline."""
    directories = [
        PROCESSED_DIR,
        MODEL_DIR,
        OUTPUT_DIR / "eda",
        OUTPUT_DIR / "task1_vlm",
        OUTPUT_DIR / "task2_classical",
        OUTPUT_DIR / "task3_unet",
        OUTPUT_DIR / "task4_hybrid",
        OUTPUT_DIR / "extensions",
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
