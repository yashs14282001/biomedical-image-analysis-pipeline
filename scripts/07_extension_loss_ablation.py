from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR
from src.data import load_manifest
from src.plotting import plot_loss_ablation, plot_training_curves
from src.train_utils import train_unet


def main() -> None:
    parser = argparse.ArgumentParser(description="Extra credit: compare BCE, Dice, and BCE+Dice losses.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    args = parser.parse_args()

    out = OUTPUT_DIR / "extensions" / "loss_ablation"
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(PROCESSED_DIR)
    rows = []
    for loss_name in ["bce", "dice", "bce_dice"]:
        model, history, metrics, checkpoint = train_unet(
            manifest,
            PROCESSED_DIR,
            out,
            MODEL_DIR,
            epochs=args.epochs,
            loss_name=loss_name,
            batch_size=args.batch_size,
            base_channels=args.base_channels,
        )
        plot_training_curves(history, out / f"loss_curves_{loss_name}.png", title_suffix=f"({loss_name})")
        rows.append({
            "loss": loss_name,
            "mean_dice": metrics["mean_dice"],
            "mean_iou": metrics["mean_iou"],
            "checkpoint": str(checkpoint),
        })

    comparison = pd.DataFrame(rows).sort_values("mean_dice", ascending=False)
    comparison.to_csv(out / "loss_ablation_metrics.csv", index=False)
    plot_loss_ablation(comparison, out / "loss_ablation_bar_chart.png")
    best = comparison.iloc[0].to_dict()
    with open(out / "best_loss.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(comparison.round(4).to_string(index=False))
    print(f"\nBest validation Dice loss: {best['loss']}")


if __name__ == "__main__":
    main()
