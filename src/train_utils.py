from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, LEARNING_RATE, NUM_WORKERS, RANDOM_SEED
from src.classical import otsu_segment
from src.unet_model import SmallUNet


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NucleiDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, processed_root: Path, augment: bool = False):
        self.rows = rows.reset_index(drop=True)
        self.processed_root = Path(processed_root)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        image = np.asarray(
            Image.open(self.processed_root / row["image_path"]).convert("L"),
            dtype=np.float32,
        ) / 255.0
        mask_path = str(row.get("mask_path", ""))
        if not mask_path or mask_path == "nan":
            raise ValueError(f"Sample {row['image_id']} has no ground-truth mask.")
        mask = (
            np.asarray(Image.open(self.processed_root / mask_path).convert("L"), dtype=np.float32)
            > 0
        ).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if random.random() < 0.5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()

        image_t = torch.from_numpy(image[None, ...]).float()
        mask_t = torch.from_numpy(mask[None, ...]).float()
        return image_t, mask_t, str(row["image_id"])


def dice_coefficient(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    preds = (probabilities >= threshold).float()
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)
    return ((2 * intersection + eps) / (denominator + eps)).mean()


def iou_score(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    preds = (probabilities >= threshold).float()
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    return ((intersection + eps) / (union + eps)).mean()


class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2 * intersection + 1e-7) / (denominator + 1e-7)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + (1 - self.bce_weight) * self.dice(logits, targets)


def get_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name in {"bce_dice", "bce+dice", "combined"}:
        return BCEDiceLoss()
    raise ValueError("loss must be one of: bce, dice, bce_dice")


def make_loaders(manifest: pd.DataFrame, processed_root: Path, batch_size: int = BATCH_SIZE):
    train_rows = manifest[(manifest["split"] == "train") & manifest["mask_path"].fillna("").ne("")]
    val_rows = manifest[(manifest["split"] == "val") & manifest["mask_path"].fillna("").ne("")]
    if train_rows.empty or val_rows.empty:
        raise ValueError("Need paired train and validation images. Re-run preprocessing and inspect manifest.csv.")

    train_ds = NucleiDataset(train_rows, processed_root, augment=True)
    val_ds = NucleiDataset(val_rows, processed_root, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, train_rows, val_rows


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    dice_values, iou_values = [], []
    per_image = []
    for images, masks, ids in loader:
        images, masks = images.to(device), masks.to(device)
        probs = torch.sigmoid(model(images))
        preds = (probs >= 0.5).float()
        for i, image_id in enumerate(ids):
            p = preds[i:i+1]
            t = masks[i:i+1]
            intersection = (p * t).sum().item()
            p_sum = p.sum().item()
            t_sum = t.sum().item()
            dice = (2 * intersection + 1e-7) / (p_sum + t_sum + 1e-7)
            union = p_sum + t_sum - intersection
            iou = (intersection + 1e-7) / (union + 1e-7)
            dice_values.append(dice)
            iou_values.append(iou)
            per_image.append({"image_id": image_id, "dice": dice, "iou": iou})
    return {
        "mean_dice": float(np.mean(dice_values)),
        "mean_iou": float(np.mean(iou_values)),
        "per_image": per_image,
    }


def train_unet(
    manifest: pd.DataFrame,
    processed_root: Path,
    output_dir: Path,
    model_dir: Path,
    epochs: int = 20,
    loss_name: str = "bce_dice",
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    base_channels: int = 16,
) -> tuple[SmallUNet, pd.DataFrame, dict, Path]:
    seed_everything()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, _ = make_loaders(manifest, processed_root, batch_size=batch_size)
    model = SmallUNet(base=base_channels).to(device)
    criterion = get_loss(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_dice = -1.0
    checkpoint_path = model_dir / f"unet_{loss_name}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses, val_dice, val_iou = [], [], []
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                probs = torch.sigmoid(logits)
                val_losses.append(float(loss.item()))
                val_dice.append(float(dice_coefficient(probs, masks).item()))
                val_iou.append(float(iou_score(probs, masks).item()))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "val_dice": float(np.mean(val_dice)),
            "val_iou": float(np.mean(val_iou)),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | train={row['train_loss']:.4f} "
            f"val={row['val_loss']:.4f} dice={row['val_dice']:.4f} iou={row['val_iou']:.4f}"
        )

        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_channels": base_channels,
                    "loss_name": loss_name,
                    "epoch": epoch,
                    "best_val_dice": best_dice,
                },
                checkpoint_path,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / f"training_history_{loss_name}.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model, val_loader, device)
    summary = {
        "loss": loss_name,
        "device": str(device),
        "epochs": epochs,
        "best_epoch": int(checkpoint["epoch"]),
        "mean_dice": metrics["mean_dice"],
        "mean_iou": metrics["mean_iou"],
    }
    with open(output_dir / f"validation_metrics_{loss_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(metrics["per_image"]).to_csv(
        output_dir / f"validation_per_image_{loss_name}.csv", index=False
    )
    return model, history_df, metrics, checkpoint_path


def load_trained_unet(checkpoint_path: Path, device: torch.device | None = None) -> tuple[SmallUNet, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SmallUNet(base=int(checkpoint.get("base_channels", 16))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


@torch.no_grad()
def predict_mask(model: nn.Module, image01: np.ndarray, device: torch.device, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(image01[None, None, ...].astype(np.float32)).to(device)
    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return (probability >= threshold).astype(np.uint8), probability


def binary_dice_np(pred: np.ndarray, true: np.ndarray) -> float:
    pred = pred.astype(bool)
    true = true.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    return float((2 * intersection + 1e-7) / (pred.sum() + true.sum() + 1e-7))


def binary_iou_np(pred: np.ndarray, true: np.ndarray) -> float:
    pred = pred.astype(bool)
    true = true.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float((intersection + 1e-7) / (union + 1e-7))


def evaluate_otsu_on_rows(rows: pd.DataFrame, processed_root: Path) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        image = np.asarray(Image.open(processed_root / row["image_path"]).convert("L"), dtype=np.float32) / 255.0
        true = (np.asarray(Image.open(processed_root / row["mask_path"]).convert("L")) > 0).astype(np.uint8)
        pred, threshold, polarity = otsu_segment(image)
        records.append(
            {
                "image_id": row["image_id"],
                "otsu_dice": binary_dice_np(pred, true),
                "otsu_iou": binary_iou_np(pred, true),
                "otsu_threshold": threshold,
                "otsu_polarity": polarity,
            }
        )
    return pd.DataFrame(records)
