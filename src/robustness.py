from __future__ import annotations

import numpy as np
from skimage import exposure, filters, util
from skimage.metrics import structural_similarity


def corrupt_image(image: np.ndarray, corruption: str = "blur", seed: int = 42) -> np.ndarray:
    corruption = corruption.lower()
    if corruption == "blur":
        return filters.gaussian(image, sigma=4, preserve_range=True).astype(np.float32)
    if corruption == "low_contrast":
        midpoint = float(image.mean())
        return np.clip(midpoint + 0.20 * (image - midpoint), 0, 1).astype(np.float32)
    if corruption == "noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.18, size=image.shape)
        return np.clip(image + noise, 0, 1).astype(np.float32)
    raise ValueError("corruption must be blur, low_contrast, or noise")


def image_change_metrics(clean: np.ndarray, corrupted: np.ndarray) -> dict:
    return {
        "clean_mean_intensity": float(clean.mean()),
        "corrupt_mean_intensity": float(corrupted.mean()),
        "clean_std_intensity": float(clean.std()),
        "corrupt_std_intensity": float(corrupted.std()),
        "mean_absolute_pixel_change": float(np.mean(np.abs(clean - corrupted))),
        "ssim_clean_vs_corrupt": float(structural_similarity(clean, corrupted, data_range=1.0)),
    }
