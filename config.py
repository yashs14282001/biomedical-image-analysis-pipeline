from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
PROMPT_DIR = ROOT / "prompts"

IMAGE_SIZE = 256
RANDOM_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_WORKERS = 0  # 0 is safest on Windows/VS Code.

VISION_MODEL = "llama3.2-vision"
TEXT_MODEL = "llama3.2:3b"

DATASET_URL = (
    "https://github.com/Nickolay-K/Assingnment-3-dataset/"
    "raw/refs/heads/main/nuclei_dataset.zip"
)
DATASET_ZIP = RAW_DIR / "nuclei_dataset.zip"
DATASET_EXTRACT_DIR = RAW_DIR / "nuclei_dataset"
