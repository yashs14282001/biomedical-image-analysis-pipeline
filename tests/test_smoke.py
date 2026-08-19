from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.draw import disk

from src.classical import run_classical
from src.data import prepare_dataset
from src.hybrid import run_hybrid_pipeline
from src.train_utils import train_unet


def _make_synthetic_dataset(root: Path, n: int = 10) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    rng = np.random.default_rng(123)

    for i in range(n):
        image = np.full((64, 64), 0.75, dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        for j in range(3 + (i % 3)):
            cy = int(rng.integers(10, 54))
            cx = int(rng.integers(10, 54))
            radius = int(rng.integers(4, 8))
            rr, cc = disk((cy, cx), radius, shape=image.shape)
            mask[rr, cc] = 1
            image[rr, cc] = 0.15 + 0.05 * rng.random()
        image = np.clip(image + rng.normal(0, 0.03, image.shape), 0, 1)
        Image.fromarray((image * 255).astype(np.uint8)).save(images / f"sample_{i:02d}.png")
        Image.fromarray(mask * 255).save(masks / f"sample_{i:02d}.png")


def test_core_pipeline_without_ollama(tmp_path: Path):
    dataset = tmp_path / "dataset"
    processed = tmp_path / "processed"
    outputs = tmp_path / "outputs"
    models = tmp_path / "models"
    _make_synthetic_dataset(dataset)

    manifest = prepare_dataset(dataset, processed, size=64, overwrite=True)
    assert set(manifest["split"]) == {"train", "val", "test"}

    first = manifest.iloc[0]
    image = np.asarray(Image.open(processed / first["image_path"]).convert("L"), dtype=np.float32) / 255.0
    classical = run_classical(image, min_object_size=5, hole_size=5)
    assert classical.features is not None

    model, history, metrics, checkpoint = train_unet(
        manifest,
        processed,
        outputs / "unet",
        models,
        epochs=1,
        loss_name="bce_dice",
        batch_size=2,
        base_channels=4,
    )
    assert checkpoint.exists()
    assert 0 <= metrics["mean_dice"] <= 1
    assert 0 <= metrics["mean_iou"] <= 1

    frame = run_hybrid_pipeline(
        manifest,
        processed,
        checkpoint,
        outputs / "hybrid",
        use_llm=False,
    )
    assert not frame.empty
    assert {"image_id", "n_objects", "mean_area", "density_class", "quality_flag"}.issubset(frame.columns)
    assert (outputs / "hybrid" / "hybrid_test_records.csv").exists()
