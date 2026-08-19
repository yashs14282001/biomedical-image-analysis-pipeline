from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure, morphology
from skimage.filters import threshold_otsu


REGION_PROPERTIES = [
    "label",
    "area",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "perimeter",
    "major_axis_length",
    "minor_axis_length",
    "extent",
    "equivalent_diameter_area",
]


@dataclass
class ClassicalResult:
    threshold: float
    polarity: str
    binary_mask: np.ndarray
    labels: np.ndarray
    features: pd.DataFrame
    summary: str
    facts: dict


def read_grayscale01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def otsu_segment(
    image: np.ndarray,
    min_object_size: int = 20,
    hole_size: int = 20,
    polarity: str = "auto",
) -> tuple[np.ndarray, float, str]:
    """Otsu segmentation plus morphology; auto polarity selects the minority phase as foreground."""
    threshold = float(threshold_otsu(image))
    bright = image > threshold
    dark = image < threshold

    if polarity == "bright":
        mask, chosen = bright, "bright"
    elif polarity == "dark":
        mask, chosen = dark, "dark"
    elif polarity == "auto":
        # Nuclei typically occupy less of the field than background. This avoids hard-coding
        # whether the stain makes nuclei brighter or darker than their surroundings.
        mask, chosen = (bright, "bright") if bright.mean() <= dark.mean() else (dark, "dark")
    else:
        raise ValueError("polarity must be 'auto', 'bright', or 'dark'")

    mask = morphology.remove_small_objects(mask.astype(bool), max_size=max(0, min_object_size - 1))
    mask = morphology.remove_small_holes(mask, max_size=max(0, hole_size - 1))
    mask = morphology.opening(mask, morphology.disk(1))
    mask = morphology.closing(mask, morphology.disk(1))
    return mask.astype(np.uint8), threshold, chosen


def region_features(mask: np.ndarray, intensity_image: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    labels = measure.label(mask > 0)
    table = measure.regionprops_table(
        labels,
        intensity_image=intensity_image,
        properties=REGION_PROPERTIES,
    )
    return labels, pd.DataFrame(table)


def density_class_from_fraction(foreground_fraction: float) -> str:
    if foreground_fraction < 0.10:
        return "low"
    if foreground_fraction < 0.30:
        return "medium"
    return "high"


def quality_flag_from_facts(n_objects: int, foreground_fraction: float, mean_area: float) -> str:
    # This is deliberately conservative and auditable; the LLM may describe the flag,
    # but the deterministic values remain available as the source of truth.
    if n_objects == 0 or foreground_fraction < 0.002 or foreground_fraction > 0.80:
        return "review"
    if mean_area <= 1:
        return "review"
    return "acceptable"


def summarise_features(features: pd.DataFrame, mask: np.ndarray) -> tuple[str, dict]:
    n = int(len(features))
    foreground_fraction = float((mask > 0).mean())
    if n == 0:
        facts = {
            "n_objects": 0,
            "foreground_fraction": foreground_fraction,
            "mean_area": 0.0,
            "median_area": 0.0,
            "mean_eccentricity": 0.0,
            "mean_solidity": 0.0,
            "mean_intensity": 0.0,
            "density_class_rule": density_class_from_fraction(foreground_fraction),
            "quality_flag_rule": "review",
        }
    else:
        facts = {
            "n_objects": n,
            "foreground_fraction": foreground_fraction,
            "mean_area": float(features["area"].mean()),
            "median_area": float(features["area"].median()),
            "mean_eccentricity": float(features["eccentricity"].mean()),
            "mean_solidity": float(features["solidity"].mean()),
            "mean_intensity": float(features["mean_intensity"].mean()),
            "density_class_rule": density_class_from_fraction(foreground_fraction),
            "quality_flag_rule": quality_flag_from_facts(
                n, foreground_fraction, float(features["area"].mean())
            ),
        }

    summary = (
        f"Connected objects: {facts['n_objects']}. "
        f"Foreground fraction: {facts['foreground_fraction']:.4f}. "
        f"Mean object area: {facts['mean_area']:.2f} pixels; median area: {facts['median_area']:.2f} pixels. "
        f"Mean eccentricity: {facts['mean_eccentricity']:.3f}. "
        f"Mean solidity: {facts['mean_solidity']:.3f}. "
        f"Mean object intensity: {facts['mean_intensity']:.3f}."
    )
    return summary, facts


def run_classical(image: np.ndarray, **kwargs) -> ClassicalResult:
    mask, threshold, polarity = otsu_segment(image, **kwargs)
    labels, features = region_features(mask, image)
    summary, facts = summarise_features(features, mask)
    return ClassicalResult(threshold, polarity, mask, labels, features, summary, facts)
