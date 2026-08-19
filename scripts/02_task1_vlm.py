from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import OUTPUT_DIR, PROCESSED_DIR, VISION_MODEL
from src.data import load_manifest
from src.llm_utils import ollama_error_message, run_naive_vlm, run_optimised_vlm


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: direct multimodal LLM descriptions.")
    parser.add_argument("--model", default=VISION_MODEL)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    output_dir = OUTPUT_DIR / "task1_vlm"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(PROCESSED_DIR)
    row = manifest[manifest["split"] == "train"].iloc[0]
    image_path = PROCESSED_DIR / row["image_path"]

    try:
        naive = run_naive_vlm(image_path, model=args.model)
        (output_dir / "naive_prompt_output.txt").write_text(naive, encoding="utf-8")

        repeated = []
        raw_outputs = []
        for run_index in range(args.repeats):
            record, raw = run_optimised_vlm(image_path, model=args.model, temperature=0.7)
            repeated.append({"run": run_index + 1, "record": record})
            raw_outputs.append(raw)
            with open(output_dir / f"optimized_run_{run_index + 1}.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

        with open(output_dir / "optimized_repeated_runs.json", "w", encoding="utf-8") as f:
            json.dump(repeated, f, indent=2)
        (output_dir / "representative_image.txt").write_text(str(image_path), encoding="utf-8")

        unique = len({json.dumps(item["record"], sort_keys=True) for item in repeated})
        variability = {
            "n_runs": args.repeats,
            "n_unique_structured_outputs": unique,
            "outputs_identical": unique == 1,
            "note": "Generative runs can differ; exact variability depends on model version and sampling.",
        }
        with open(output_dir / "run_to_run_variability.json", "w", encoding="utf-8") as f:
            json.dump(variability, f, indent=2)

        print("Naive output:\n", naive)
        print("\nOptimised structured runs:")
        print(json.dumps(repeated, indent=2))
        print("\nVariability summary:", variability)
    except Exception as exc:
        raise RuntimeError(ollama_error_message(exc)) from exc


if __name__ == "__main__":
    main()
