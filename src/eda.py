from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def _read01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def make_eda_figures(manifest: pd.DataFrame, processed_root: Path, output_dir: Path, n_samples: int = 6) -> pd.DataFrame:
    """Create the assignment EDA: sample image panel, intensity histogram, and simple statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = manifest[manifest["split"] == "train"].head(n_samples)
    if train_rows.empty:
        raise ValueError("No training images found in manifest.")

    fig, axes = plt.subplots(2, int(np.ceil(len(train_rows) / 2)), figsize=(12, 6))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes, train_rows.iterrows()):
        image = _read01(processed_root / row["image_path"])
        ax.imshow(image, cmap="gray")
        ax.set_title(row["image_id"])
        ax.axis("off")
    for ax in axes[len(train_rows):]:
        ax.axis("off")
    fig.suptitle("Representative preprocessed nuclei images (grayscale, 256×256)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_sample_images.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    all_pixels = []
    rows = []
    for _, row in manifest.iterrows():
        image = _read01(processed_root / row["image_path"])
        all_pixels.append(image.ravel())
        rows.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "mean_intensity": float(image.mean()),
                "std_intensity": float(image.std()),
                "min_intensity": float(image.min()),
                "max_intensity": float(image.max()),
            }
        )

    pixels = np.concatenate(all_pixels)
    fig = plt.figure(figsize=(7.2, 4.5))
    plt.hist(pixels, bins=64)
    plt.xlabel("Normalised grayscale intensity")
    plt.ylabel("Pixel count")
    plt.title("Intensity distribution across preprocessed images")
    plt.tight_layout()
    fig.savefig(output_dir / "eda_intensity_histogram.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    stats = pd.DataFrame(rows)
    stats.to_csv(output_dir / "eda_image_statistics.csv", index=False)

    split_counts = manifest.groupby("split").size().rename("n_images").reset_index()
    split_counts.to_csv(output_dir / "dataset_split_counts.csv", index=False)
    return stats
