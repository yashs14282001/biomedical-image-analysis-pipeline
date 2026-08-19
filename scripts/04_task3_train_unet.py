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

from config import EPOCHS, MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR
from src.data import load_manifest
from src.plotting import plot_training_curves, plot_validation_predictions
from src.train_utils import (
    binary_dice_np,
    binary_iou_np,
    evaluate_otsu_on_rows,
    predict_mask,
    train_unet,
)


def make_method_example_panel(model, device, comparison: pd.DataFrame, val_rows: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    comparison = comparison.copy()
    comparison["unet_minus_otsu_dice"] = comparison["unet_dice"] - comparison["otsu_dice"]
    unet_example = comparison.sort_values("unet_minus_otsu_dice", ascending=False).iloc[0]
    otsu_example = comparison.sort_values("unet_minus_otsu_dice", ascending=True).iloc[0]
    chosen = pd.DataFrame([
        {"winner_label": "U-Net stronger example", **unet_example.to_dict()},
        {"winner_label": "Otsu stronger example", **otsu_example.to_dict()},
    ])

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for r, example in chosen.iterrows():
        row = val_rows[val_rows["image_id"] == example["image_id"]].iloc[0]
        image = np.asarray(Image.open(PROCESSED_DIR / row["image_path"]).convert("L"), dtype=np.float32) / 255.0
        true = (np.asarray(Image.open(PROCESSED_DIR / row["mask_path"]).convert("L")) > 0).astype(np.uint8)
        from src.classical import otsu_segment
        otsu, _, _ = otsu_segment(image)
        unet, _ = predict_mask(model, image, device)
        panels = [
            (image, "Input"),
            (true, "Ground truth"),
            (otsu, f"Otsu\nDice={example['otsu_dice']:.3f}"),
            (unet, f"U-Net\nDice={example['unet_dice']:.3f}"),
        ]
        for c, (arr, title) in enumerate(panels):
            axes[r, c].imshow(arr, cmap="gray")
            axes[r, c].set_title(f"{example['winner_label']}\n{example['image_id']}\n{title}" if c == 0 else title)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: train and evaluate the small U-Net.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--loss", choices=["bce", "dice", "bce_dice"], default="bce_dice")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    args = parser.parse_args()

    output_dir = OUTPUT_DIR / "task3_unet"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(PROCESSED_DIR)

    model, history, metrics, checkpoint = train_unet(
        manifest=manifest,
        processed_root=PROCESSED_DIR,
        output_dir=output_dir,
        model_dir=MODEL_DIR,
        epochs=args.epochs,
        loss_name=args.loss,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
    )

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    val_rows = manifest[(manifest["split"] == "val") & manifest["mask_path"].fillna("").ne("")]

    plot_training_curves(
        history,
        output_dir / f"loss_curves_{args.loss}.png",
        title_suffix=f"({args.loss})",
    )
    plot_validation_predictions(
        model,
        device,
        val_rows,
        PROCESSED_DIR,
        output_dir / f"validation_predictions_{args.loss}.png",
        n=3,
    )

    unet_per = pd.DataFrame(metrics["per_image"]).rename(columns={"dice": "unet_dice", "iou": "unet_iou"})
    otsu_per = evaluate_otsu_on_rows(val_rows, PROCESSED_DIR)
    comparison = otsu_per.merge(unet_per, on="image_id", how="inner")
    comparison.to_csv(output_dir / "otsu_vs_unet_validation.csv", index=False)

    method_summary = pd.DataFrame([
        {
            "method": "Otsu + morphology",
            "mean_dice": comparison["otsu_dice"].mean(),
            "mean_iou": comparison["otsu_iou"].mean(),
        },
        {
            "method": f"U-Net ({args.loss})",
            "mean_dice": comparison["unet_dice"].mean(),
            "mean_iou": comparison["unet_iou"].mean(),
        },
    ])
    method_summary.to_csv(output_dir / "method_comparison_metrics.csv", index=False)

    examples = make_method_example_panel(
        model,
        device,
        comparison,
        val_rows,
        output_dir / "otsu_vs_unet_examples.png",
    )
    examples.to_csv(output_dir / "otsu_vs_unet_example_ids.csv", index=False)

    print("\nValidation metrics:")
    print(method_summary.round(4).to_string(index=False))
    print("\nReport-ready comparison examples:")
    print(examples[["winner_label", "image_id", "otsu_dice", "unet_dice"]].round(4).to_string(index=False))
    print(f"\nBest model checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
