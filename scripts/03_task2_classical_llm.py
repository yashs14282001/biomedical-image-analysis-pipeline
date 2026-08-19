from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from config import OUTPUT_DIR, PROCESSED_DIR, TEXT_MODEL
from src.classical import read_grayscale01, run_classical
from src.data import load_manifest
from src.llm_utils import ollama_error_message, run_numbers_first


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: Otsu + regionprops + numbers-only LLM.")
    parser.add_argument("--model", default=TEXT_MODEL)
    parser.add_argument("--no-llm", action="store_true", help="Debug feature extraction without Ollama.")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR / "task2_classical"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(PROCESSED_DIR)
    row = manifest[manifest["split"] == "train"].iloc[0]
    image = read_grayscale01(PROCESSED_DIR / row["image_path"])
    result = run_classical(image)

    result.features.to_csv(output_dir / "representative_regionprops.csv", index=False)
    (output_dir / "feature_summary.txt").write_text(result.summary, encoding="utf-8")
    with open(output_dir / "measured_facts.json", "w", encoding="utf-8") as f:
        json.dump(result.facts, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(result.binary_mask, cmap="gray")
    axes[1].set_title(f"Otsu + cleanup\nthreshold={result.threshold:.3f}")
    axes[2].imshow(result.labels, cmap="nipy_spectral")
    axes[2].set_title(f"Connected components: {len(result.features)}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "classical_segmentation_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    numbers_record = None
    if not args.no_llm:
        try:
            numbers_record, raw, narrative = run_numbers_first(
                result.summary, result.facts, model=args.model
            )
            with open(output_dir / "numbers_first_record.json", "w", encoding="utf-8") as f:
                json.dump(numbers_record, f, indent=2)
            (output_dir / "numbers_first_raw_llm.txt").write_text(raw, encoding="utf-8")
            (output_dir / "numbers_first_narrative.txt").write_text(narrative, encoding="utf-8")
            print("Numbers-first JSON:\n", json.dumps(numbers_record, indent=2))
            print("\nNarrative:\n", narrative)
        except Exception as exc:
            raise RuntimeError(ollama_error_message(exc)) from exc

    # Bring Task 1 and Task 2 records together for report-ready comparison.
    task1_runs = OUTPUT_DIR / "task1_vlm" / "optimized_repeated_runs.json"
    if task1_runs.exists() and numbers_record is not None:
        direct = json.loads(task1_runs.read_text(encoding="utf-8"))[0]["record"]
        comparison = {
            "direct_vlm_record": direct,
            "numbers_first_record": numbers_record,
            "comparison_note": (
                "The direct VLM can describe appearance not represented in region statistics, "
                "while the numbers-first record is easier to audit because every claim is constrained by measured features."
            ),
        }
        with open(output_dir / "task1_vs_task2_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()
