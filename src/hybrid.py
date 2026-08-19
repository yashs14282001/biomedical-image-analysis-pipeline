from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.classical import region_features, summarise_features
from src.train_utils import load_trained_unet, predict_mask


def read_image01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(path)


def run_hybrid_pipeline(
    manifest: pd.DataFrame,
    processed_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    use_llm: bool = True,
) -> pd.DataFrame:
    """U-Net -> region features -> structured record -> narrative on unseen test images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "predicted_masks"
    features_dir = output_dir / "region_features"
    records_dir = output_dir / "records"
    narratives_dir = output_dir / "narratives"
    raw_dir = output_dir / "raw_llm"
    for d in [masks_dir, features_dir, records_dir, narratives_dir, raw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    test_rows = manifest[manifest["split"] == "test"].copy()
    if test_rows.empty:
        raise ValueError("No test images found in manifest.")

    model, device = load_trained_unet(checkpoint_path)
    aggregate = []

    for _, row in test_rows.iterrows():
        image_id = str(row["image_id"])
        image = read_image01(processed_root / row["image_path"])
        mask, probability = predict_mask(model, image, device)
        save_binary_mask(mask, masks_dir / f"{image_id}_mask.png")

        _, features = region_features(mask, image)
        features.to_csv(features_dir / f"{image_id}_regionprops.csv", index=False)
        summary, facts = summarise_features(features, mask)

        if use_llm:
            from src.llm_utils import narrative_from_record, run_hybrid_record
            record, raw = run_hybrid_record(image_id, summary, facts)
            narrative = narrative_from_record(record, summary)
            (raw_dir / f"{image_id}_record_raw.txt").write_text(raw, encoding="utf-8")
        else:
            # Useful for debugging segmentation without Ollama. The marked assignment run
            # should use the LLM path above.
            record = {
                "image_id": image_id,
                "n_objects": int(facts["n_objects"]),
                "mean_area": round(float(facts["mean_area"]), 3),
                "density_class": facts["density_class_rule"],
                "quality_flag": facts["quality_flag_rule"],
            }
            narrative = (
                "Debug mode (no LLM): " + summary
            )

        # Store measured audit fields alongside the required record in the CSV.
        audit = {
            "foreground_fraction": facts["foreground_fraction"],
            "mean_eccentricity": facts["mean_eccentricity"],
            "mean_solidity": facts["mean_solidity"],
            "mean_object_intensity": facts["mean_intensity"],
            "mean_prediction_probability": float(probability.mean()),
        }
        final_row = {**record, **audit, "narrative": narrative}
        aggregate.append(final_row)

        with open(records_dir / f"{image_id}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        (narratives_dir / f"{image_id}.txt").write_text(narrative, encoding="utf-8")
        (records_dir / f"{image_id}_feature_summary.txt").write_text(summary, encoding="utf-8")

    frame = pd.DataFrame(aggregate)
    frame.to_csv(output_dir / "hybrid_test_records.csv", index=False)
    with open(output_dir / "hybrid_test_records.jsonl", "w", encoding="utf-8") as f:
        for record in aggregate:
            f.write(json.dumps(record) + "\n")
    return frame
