from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.train_utils import predict_mask


def plot_training_curves(history: pd.DataFrame, output_path: Path, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"U-Net loss curves {title_suffix}".strip())
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metric_path = output_path.with_name(output_path.stem.replace("loss", "dice_iou") + output_path.suffix)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(history["epoch"], history["val_dice"], label="Validation Dice")
    ax.plot(history["epoch"], history["val_iou"], label="Validation IoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"U-Net validation segmentation scores {title_suffix}".strip())
    ax.legend()
    fig.tight_layout()
    fig.savefig(metric_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation_predictions(model, device, val_rows: pd.DataFrame, processed_root: Path, output_path: Path, n: int = 3) -> None:
    rows = val_rows.head(n)
    if rows.empty:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    for r, (_, row) in enumerate(rows.iterrows()):
        image = np.asarray(Image.open(processed_root / row["image_path"]).convert("L"), dtype=np.float32) / 255.0
        true = np.asarray(Image.open(processed_root / row["mask_path"]).convert("L")) > 0
        pred, _ = predict_mask(model, image, device)
        for c, (arr, title) in enumerate(
            [(image, "Input"), (true, "Ground truth"), (pred, "U-Net prediction")]
        ):
            axes[r, c].imshow(arr, cmap="gray")
            axes[r, c].set_title(f"{row['image_id']} — {title}")
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loss_ablation(comparison: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(comparison))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, comparison["mean_dice"], width, label="Dice")
    ax.bar(x + width / 2, comparison["mean_iou"], width, label="IoU")
    ax.set_xticks(x, comparison["loss"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Validation score")
    ax.set_title("U-Net loss ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
