from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR
from src.classical import region_features, summarise_features
from src.data import load_manifest
from src.llm_utils import narrative_from_record, ollama_error_message, run_hybrid_record
from src.robustness import corrupt_image, image_change_metrics
from src.train_utils import binary_dice_np, load_trained_unet, predict_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Extra credit: trace corruption through the hybrid pipeline.")
    parser.add_argument("--corruption", choices=["blur", "low_contrast", "noise"], default="blur")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "unet_bce_dice.pt")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    out = OUTPUT_DIR / "extensions" / f"robustness_{args.corruption}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(PROCESSED_DIR)
    row = manifest[manifest["split"] == "test"].iloc[0]
    image_id = str(row["image_id"])
    clean = np.asarray(Image.open(PROCESSED_DIR / row["image_path"]).convert("L"), dtype=np.float32) / 255.0
    corrupt = corrupt_image(clean, args.corruption)

    model, device = load_trained_unet(args.checkpoint)
    clean_mask, _ = predict_mask(model, clean, device)
    corrupt_mask, _ = predict_mask(model, corrupt, device)

    _, clean_features = region_features(clean_mask, clean)
    _, corrupt_features = region_features(corrupt_mask, corrupt)
    clean_summary, clean_facts = summarise_features(clean_features, clean_mask)
    corrupt_summary, corrupt_facts = summarise_features(corrupt_features, corrupt_mask)

    trace = image_change_metrics(clean, corrupt)
    trace.update({
        "image_id": image_id,
        "corruption": args.corruption,
        "clean_mask_vs_corrupt_mask_dice": binary_dice_np(clean_mask, corrupt_mask),
        "clean_n_objects": clean_facts["n_objects"],
        "corrupt_n_objects": corrupt_facts["n_objects"],
        "clean_mean_area": clean_facts["mean_area"],
        "corrupt_mean_area": corrupt_facts["mean_area"],
        "clean_foreground_fraction": clean_facts["foreground_fraction"],
        "corrupt_foreground_fraction": corrupt_facts["foreground_fraction"],
    })

    # A simple auditable rule for the earliest stage where corruption is measurably visible.
    image_detected = trace["ssim_clean_vs_corrupt"] < 0.95 or abs(
        trace["clean_std_intensity"] - trace["corrupt_std_intensity"]
    ) > 0.05
    mask_detected = trace["clean_mask_vs_corrupt_mask_dice"] < 0.90
    object_delta = abs(clean_facts["n_objects"] - corrupt_facts["n_objects"])
    feature_detected = object_delta >= max(1, int(0.2 * max(clean_facts["n_objects"], 1)))

    if image_detected:
        earliest = "input/image statistics"
    elif mask_detected:
        earliest = "U-Net mask"
    elif feature_detected:
        earliest = "region feature table"
    else:
        earliest = "not strongly detectable by configured numeric rules"
    trace["earliest_detectable_stage"] = earliest

    if not args.no_llm:
        try:
            clean_record, clean_raw = run_hybrid_record(image_id + "_clean", clean_summary, clean_facts)
            corrupt_record, corrupt_raw = run_hybrid_record(image_id + "_corrupt", corrupt_summary, corrupt_facts)
            clean_narrative = narrative_from_record(clean_record, clean_summary)
            corrupt_narrative = narrative_from_record(corrupt_record, corrupt_summary)
            (out / "clean_record.json").write_text(json.dumps(clean_record, indent=2), encoding="utf-8")
            (out / "corrupt_record.json").write_text(json.dumps(corrupt_record, indent=2), encoding="utf-8")
            (out / "clean_narrative.txt").write_text(clean_narrative, encoding="utf-8")
            (out / "corrupt_narrative.txt").write_text(corrupt_narrative, encoding="utf-8")
            trace["llm_record_changed"] = clean_record != corrupt_record
        except Exception as exc:
            raise RuntimeError(ollama_error_message(exc)) from exc

    pd.DataFrame([trace]).to_csv(out / "robustness_trace.csv", index=False)
    (out / "clean_feature_summary.txt").write_text(clean_summary, encoding="utf-8")
    (out / "corrupt_feature_summary.txt").write_text(corrupt_summary, encoding="utf-8")
    clean_features.to_csv(out / "clean_regionprops.csv", index=False)
    corrupt_features.to_csv(out / "corrupt_regionprops.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes[0, 0].imshow(clean, cmap="gray")
    axes[0, 0].set_title("Clean image")
    axes[0, 1].imshow(corrupt, cmap="gray")
    axes[0, 1].set_title(f"Corrupted: {args.corruption}")
    axes[1, 0].imshow(clean_mask, cmap="gray")
    axes[1, 0].set_title("Clean U-Net mask")
    axes[1, 1].imshow(corrupt_mask, cmap="gray")
    axes[1, 1].set_title(f"Corrupt mask\nmask Dice={trace['clean_mask_vs_corrupt_mask_dice']:.3f}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "robustness_propagation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(pd.DataFrame([trace]).T.to_string(header=False))
    print(f"\nEarliest detectable stage: {earliest}")


if __name__ == "__main__":
    main()
