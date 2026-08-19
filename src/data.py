from __future__ import annotations

import json
import random
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from config import IMAGE_SIZE, RANDOM_SEED, TEST_FRACTION, VAL_FRACTION

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_HINTS = {
    "mask", "masks", "label", "labels", "annotation", "annotations",
    "ground_truth", "groundtruth", "gt", "segmentation", "segmentations",
}
IMAGE_HINTS = {"image", "images", "img", "imgs", "raw"}
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


@dataclass
class Sample:
    image_path: Path
    mask_paths: list[Path]
    source_split: str | None = None


class DatasetDiscoveryError(RuntimeError):
    pass


def extract_zip(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    return destination


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _tokens(path: Path) -> set[str]:
    values: set[str] = set()
    for part in path.parts:
        values.update(re.split(r"[^a-z0-9]+", part.lower()))
    return {v for v in values if v}


def _looks_like_mask(path: Path) -> bool:
    tokens = _tokens(path)
    parts = {part.lower() for part in path.parts}
    stem = path.stem.lower()
    return bool(tokens & MASK_HINTS) or bool(parts & MASK_HINTS) or any(
        marker in stem for marker in ["_mask", "mask_", "_label", "label_", "_gt", "gt_"]
    )


def _detect_split(path: Path) -> str | None:
    for part in path.parts:
        key = part.lower()
        if key in SPLIT_ALIASES:
            return SPLIT_ALIASES[key]
    return None


def _normalise_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(
        r"^(image|img|mask|label|annotation|gt|segmentation)[-_ ]*", "", stem
    )
    stem = re.sub(
        r"[-_ ]*(image|img|mask|label|annotation|gt|segmentation)$", "", stem
    )
    return re.sub(r"[^a-z0-9]+", "", stem)


def _discover_dsb_style(root: Path) -> tuple[list[Sample], set[Path]]:
    """Find sample-folder layouts such as sample/images/x.png + sample/masks/*.png."""
    samples: list[Sample] = []
    consumed: set[Path] = set()
    for mask_dir in root.rglob("*"):
        if not mask_dir.is_dir() or mask_dir.name.lower() not in MASK_HINTS:
            continue
        parent = mask_dir.parent
        image_dirs = [d for d in parent.iterdir() if d.is_dir() and d.name.lower() in IMAGE_HINTS]
        if not image_dirs:
            continue
        image_files = [p for d in image_dirs for p in d.iterdir() if _is_image_file(p)]
        mask_files = [p for p in mask_dir.iterdir() if _is_image_file(p)]
        if len(image_files) == 1 and mask_files:
            samples.append(
                Sample(
                    image_path=image_files[0],
                    mask_paths=sorted(mask_files),
                    source_split=_detect_split(image_files[0]),
                )
            )
            consumed.add(image_files[0])
            consumed.update(mask_files)
    return samples, consumed


def discover_samples(root: Path) -> tuple[list[Sample], list[Sample]]:
    """
    Discover paired image/mask samples and unpaired test images.

    The function supports common biomedical layouts:
      * images/*.png and masks/*.png with matching stems
      * train/images + train/masks (and val/test equivalents)
      * Data-Science-Bowl-like sample/images/x.png + sample/masks/*.png
      * flat image_001.png + mask_001.png naming
    """
    root = Path(root)
    if not root.exists():
        raise DatasetDiscoveryError(f"Dataset directory does not exist: {root}")

    dsb_samples, consumed = _discover_dsb_style(root)
    all_files = [p for p in root.rglob("*") if _is_image_file(p) and p not in consumed]
    masks = [p for p in all_files if _looks_like_mask(p)]
    images = [p for p in all_files if not _looks_like_mask(p)]

    mask_index: dict[tuple[str | None, str], list[Path]] = {}
    for mask in masks:
        split_key = _detect_split(mask)
        norm_key = _normalise_key(mask)
        mask_index.setdefault((split_key, norm_key), []).append(mask)
        # Also allow a split-agnostic fallback when the source actually has a split label.
        if split_key is not None:
            mask_index.setdefault((None, norm_key), []).append(mask)

    paired: list[Sample] = list(dsb_samples)
    unpaired_test: list[Sample] = []
    used_masks: set[Path] = set()

    for image in sorted(images):
        split = _detect_split(image)
        key = _normalise_key(image)
        candidates = mask_index.get((split, key), []) or mask_index.get((None, key), [])
        candidates = [p for p in candidates if p not in used_masks]

        if candidates:
            # In parallel images/masks layouts there is normally one mask per image.
            # If several match, combine them as a binary union later.
            paired.append(Sample(image, sorted(candidates), split))
            used_masks.update(candidates)
        elif split == "test":
            unpaired_test.append(Sample(image, [], split))

    # A final fallback for equal-length image/mask folders whose names are not stem-matched.
    if not paired and images and masks and len(images) == len(masks):
        image_sorted = sorted(images)
        mask_sorted = sorted(masks)
        paired = [
            Sample(img, [msk], _detect_split(img))
            for img, msk in zip(image_sorted, mask_sorted)
        ]

    if not paired:
        preview = "\n".join(str(p.relative_to(root)) for p in sorted(all_files)[:25])
        raise DatasetDiscoveryError(
            "Could not automatically identify paired images and masks. "
            "Check the extracted dataset layout. First files found:\n" + preview
        )

    return paired, unpaired_test


def load_grayscale(path: Path, size: int = IMAGE_SIZE, *, is_mask: bool = False) -> np.ndarray:
    image = Image.open(path).convert("L")
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    image = image.resize((size, size), resample=resample)
    arr = np.asarray(image, dtype=np.float32)
    if is_mask:
        return (arr > 0).astype(np.uint8)
    return arr / 255.0


def combine_masks(mask_paths: Iterable[Path], size: int = IMAGE_SIZE) -> np.ndarray:
    combined = np.zeros((size, size), dtype=np.uint8)
    for path in mask_paths:
        combined = np.maximum(combined, load_grayscale(path, size=size, is_mask=True))
    return combined


def save_grayscale(arr: np.ndarray, path: Path, *, is_mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_mask:
        data = (arr > 0).astype(np.uint8) * 255
    else:
        data = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(data, mode="L").save(path)


def _assign_random_splits(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    samples = samples.copy()
    rng.shuffle(samples)
    n = len(samples)
    n_test = max(1, round(n * TEST_FRACTION)) if n >= 5 else 1
    n_val = max(1, round(n * VAL_FRACTION)) if n >= 4 else 1
    if n_test + n_val >= n:
        n_test, n_val = 1, 1 if n > 2 else 0
    return {
        "test": samples[:n_test],
        "val": samples[n_test:n_test + n_val],
        "train": samples[n_test + n_val:],
    }


def assign_splits(
    paired: list[Sample],
    unpaired_test: list[Sample],
    seed: int = RANDOM_SEED,
) -> dict[str, list[Sample]]:
    """Respect dataset-provided splits when present; otherwise make a deterministic split."""
    explicit = {"train": [], "val": [], "test": []}
    unspecified: list[Sample] = []
    for sample in paired:
        if sample.source_split in explicit:
            explicit[sample.source_split].append(sample)
        else:
            unspecified.append(sample)

    has_explicit_train = bool(explicit["train"])
    if not has_explicit_train:
        split_map = _assign_random_splits(paired, seed)
    else:
        split_map = explicit
        if unspecified:
            split_map["train"].extend(unspecified)

        # If validation was not supplied, take a small deterministic hold-out from train.
        if not split_map["val"] and len(split_map["train"]) >= 4:
            rng = random.Random(seed)
            rng.shuffle(split_map["train"])
            n_val = max(1, round(len(split_map["train"]) * VAL_FRACTION))
            split_map["val"] = split_map["train"][:n_val]
            split_map["train"] = split_map["train"][n_val:]

        # If no explicit test set exists, reserve one from train.
        if not split_map["test"] and not unpaired_test and len(split_map["train"]) >= 5:
            rng = random.Random(seed + 1)
            rng.shuffle(split_map["train"])
            n_test = max(1, round(len(split_map["train"]) * TEST_FRACTION))
            split_map["test"] = split_map["train"][:n_test]
            split_map["train"] = split_map["train"][n_test:]

    split_map["test"].extend(unpaired_test)
    return split_map


def prepare_dataset(
    dataset_root: Path,
    processed_root: Path,
    size: int = IMAGE_SIZE,
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Convert to grayscale 256x256, create splits, and save a reproducible manifest."""
    processed_root = Path(processed_root)
    if overwrite and processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    paired, unpaired_test = discover_samples(Path(dataset_root))
    split_map = assign_splits(paired, unpaired_test, seed=seed)

    rows: list[dict] = []
    counters = {"train": 0, "val": 0, "test": 0}
    for split, samples in split_map.items():
        for sample in samples:
            idx = counters[split]
            counters[split] += 1
            image_id = f"{split}_{idx:04d}"
            image = load_grayscale(sample.image_path, size=size, is_mask=False)
            out_image = processed_root / split / "images" / f"{image_id}.png"
            save_grayscale(image, out_image, is_mask=False)

            out_mask: Path | None = None
            if sample.mask_paths:
                mask = combine_masks(sample.mask_paths, size=size)
                out_mask = processed_root / split / "masks" / f"{image_id}.png"
                save_grayscale(mask, out_mask, is_mask=True)

            rows.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "image_path": str(out_image.relative_to(processed_root)),
                    "mask_path": str(out_mask.relative_to(processed_root)) if out_mask else "",
                    "source_image": str(sample.image_path),
                    "source_masks": json.dumps([str(p) for p in sample.mask_paths]),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["split", "image_id"]).reset_index(drop=True)
    manifest.to_csv(processed_root / "manifest.csv", index=False)
    return manifest


def load_manifest(processed_root: Path) -> pd.DataFrame:
    path = Path(processed_root) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No processed manifest found at {path}. Run scripts/01_prepare_eda.py first."
        )
    return pd.read_csv(path)
