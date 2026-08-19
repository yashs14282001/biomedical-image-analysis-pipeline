from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR
from src.data import load_manifest
from src.hybrid import run_hybrid_pipeline
from src.llm_utils import ollama_error_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: run full U-Net -> features -> LLM hybrid pipeline.")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "unet_bce_dice.pt")
    parser.add_argument("--no-llm", action="store_true", help="Debug only: make deterministic records without Ollama.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run scripts/04_task3_train_unet.py first."
        )

    manifest = load_manifest(PROCESSED_DIR)
    try:
        frame = run_hybrid_pipeline(
            manifest,
            PROCESSED_DIR,
            args.checkpoint,
            OUTPUT_DIR / "task4_hybrid",
            use_llm=not args.no_llm,
        )
    except Exception as exc:
        if not args.no_llm:
            raise RuntimeError(ollama_error_message(exc)) from exc
        raise

    print("\nHybrid test records:")
    print(frame.head().to_string(index=False))
    print(f"\nAggregated CSV: {OUTPUT_DIR / 'task4_hybrid' / 'hybrid_test_records.csv'}")


if __name__ == "__main__":
    main()
